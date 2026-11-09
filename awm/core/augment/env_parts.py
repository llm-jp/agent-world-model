"""Parts that rename tools or parameters by rebuilding the scenario's environment.

The renamed tools are real: the server code is rewritten with the new names
(operation ids, parameters, descriptions, error messages), so nothing the agent
sees still uses an old name. The rebuilt environment becomes a scenario of its
own (<scenario>__<part>) with the original database schema, sample data and
verifiers, and with source tasks rewritten for the new names.

A rewrite of the server code is accepted only if the server starts, lists
exactly the renamed tools and parameters, keeps no old tool name in what it
lists or in its string literals, and still solves the source tasks: their
reference calls, renamed, fail where they failed on the original server and
leave a database the task's verifier grades complete.
"""

import ast
import asyncio
import dataclasses
import difflib
import json
import logging
import os
import random
import re
import string
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import ClassVar

from awm.core.agent import MCPToolExecutor
from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Rejection,
    Variant,
    keep_values,
    properties,
)
from awm.core.augment.refine import Complete, refine
from awm.core.augment.task_parts import (
    accept_task,
    rewrite_step,
)
from awm.core.server import RunningServer, ServerStartError, serve_scenario
from awm.core.tool_view import TaskMeta
from awm.core.verify import run_verifier
from awm.prompts import (
    AUG_ENV_RENAME_CHECK_PROMPT,
    AUG_ENV_RENAME_PROMPT,
    AUG_TASK_RENAME_PROMPT,
    AUG_TOOL_RENAME_PROMPT,
)
from awm.tools import is_rebuilt_env_row, normalize_scenario_name, tools_jsonl_save

logger = logging.getLogger(__name__)

RANDOM_NAME_LENGTH = 8
# names that function-calling APIs accept
TOOL_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
SERVER_START_TIMEOUT = 60.0
CALL_TIMEOUT = 60.0
# longest server output or error text sent back to the LLM
MAX_FEEDBACK_CHARS = 1500
# MCPToolExecutor.call_tool marks a failed call with this prefix
TOOL_ERROR_PREFIX = "Error: "
# environment scenarios are named <scenario><separator><part>
SCENARIO_SEPARATOR = "__"


def to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _word(name: str) -> str:
    return rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"


def mentions(text: str, name: str, ignore_case: bool = False) -> bool:
    """Whether the name appears in the text as a whole word."""
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(_word(name), text, flags) is not None


@dataclass(frozen=True)
class Renames:
    # old tool name -> new tool name
    tools: dict[str, str] = field(default_factory=dict)
    # old tool name -> {old parameter name -> new parameter name}
    params: dict[str, dict[str, str]] = field(default_factory=dict)

    def tool(self, name: str) -> str:
        return self.tools.get(name, name)

    def call(self, call: dict) -> dict:
        """The call made against the renamed tools."""
        name = call["tool_name"]
        params = self.params.get(name, {})
        arguments = call.get("arguments") or {}
        return {
            "tool_name": self.tool(name),
            "arguments": {params.get(k, k): v for k, v in arguments.items()},
        }

    def renamed_params(self) -> dict[str, str]:
        return {old: new for ps in self.params.values() for old, new in ps.items()}

    def mentioned_in(self, text: str) -> dict[str, str]:
        """Old names, with their new names, that the text mentions.

        Tool names match in any case; parameter names only as written, since
        ones like "note" are ordinary words (ParamRename only renames names
        with "_").
        """
        found = {
            old: new for old, new in self.tools.items() if mentions(text, old, True)
        }
        found.update(
            {
                old: new
                for old, new in self.renamed_params().items()
                if mentions(text, old)
            }
        )
        return found

    def as_json(self) -> str:
        return json.dumps(
            {"tools": self.tools, "params": self.params}, ensure_ascii=False
        )


@dataclass
class ScenarioSource:
    """A scenario to rebuild, with the solved tasks that will check the rebuild."""

    scenario: str
    env_row: dict
    tools: list[dict]
    drafts: list[Draft]

    @property
    def code(self) -> str:
        return self.env_row["full_code"]

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]


# ---------------------------------------------------------------- names


def rename_problems(tool_names: list[str], renames: dict[str, str]) -> list[str]:
    """What makes the new tool names unusable."""
    problems = []
    for old, new in renames.items():
        if not TOOL_NAME_PATTERN.fullmatch(new):
            problems.append(
                f"{new!r} must start with a letter and use only letters, digits and _ (at most 64 characters)"
            )
        if new.replace("_", "").lower() == old.replace("_", "").lower():
            problems.append(f"{new!r} only changes the case or separators of {old!r}")
        if new in tool_names and new != old:
            problems.append(f"{new!r} is already the name of another tool")
    if len(set(renames.values())) < len(renames):
        problems.append("two tools got the same new name")
    return problems


class EnvPart:
    """Renames of one kind; subclasses implement renames."""

    name: ClassVar[str]

    def renames(
        self, source: ScenarioSource, rng: random.Random
    ) -> Renames | LLMStep[Renames] | None:
        raise NotImplementedError


class ToolRename(EnvPart):
    """Let the LLM give every tool a new name that still describes what it does."""

    name = "tool_rename"

    def renames(self, source: ScenarioSource, rng: random.Random) -> LLMStep[Renames]:
        docs = {t["name"]: t.get("description", "") for t in source.tools}

        def finish(response: dict) -> Renames | Rejection:
            names = response.get("tools")
            if not isinstance(names, dict) or set(names) != set(docs):
                return Rejection(
                    f'"tools" must map exactly these tool names to new names: {sorted(docs)}'
                )
            if not all(isinstance(v, str) for v in names.values()):
                return Rejection("Every new name must be a string.")
            names = {old: new.strip() for old, new in names.items()}
            if problems := rename_problems(source.tool_names, names):
                return Rejection("; ".join(problems))
            return Renames(tools=names)

        prompt = AUG_TOOL_RENAME_PROMPT.format(
            tools_json=json.dumps(docs, ensure_ascii=False)
        )
        return LLMStep(
            [{"role": "user", "content": prompt}],
            finish,
            label=f"{source.scenario} tool names",
        )


class ToolRenameRandom(EnvPart):
    """Give every tool a random name, so only the descriptions tell them apart."""

    name = "tool_rename_random"

    def renames(self, source: ScenarioSource, rng: random.Random) -> Renames:
        taken = set(source.tool_names)
        names = {}
        for old in source.tool_names:
            new = old
            while new in taken:
                new = "".join(rng.choices(string.ascii_lowercase, k=RANDOM_NAME_LENGTH))
            taken.add(new)
            names[old] = new
        return Renames(tools=names)


class ParamRename(EnvPart):
    """Write the parameters with "_" in camelCase."""

    name = "param_rename"

    def renames(self, source: ScenarioSource, rng: random.Random) -> Renames | None:
        params = {}
        for tool in source.tools:
            names = list(properties(tool))
            mapping = {p: to_camel(p) for p in names if to_camel(p) != p}
            if mapping and not set(mapping.values()) & set(names):
                params[tool["name"]] = mapping
        return Renames(params=params) if params else None


ENV_PARTS: dict[str, EnvPart] = {
    part.name: part for part in [ToolRename(), ToolRenameRandom(), ParamRename()]
}


# --------------------------------------------------------------- replay


@dataclass
class ReplayContext:
    # gen_db.jsonl and gen_sample.jsonl, to create a fresh database
    db_path: str
    sample_path: str
    # directory for the servers and databases of the replays
    work_dir: str


@dataclass
class Replay:
    tools: list[dict]
    # error text of each call, None if it succeeded
    errors: list[str | None]
    reward: str


async def _make_calls(
    url: str, calls: list[dict]
) -> tuple[list[dict], list[str | None]]:
    """The listed tools, and the error text of each call (None if it succeeded)."""
    executor = MCPToolExecutor(url, timeout=CALL_TIMEOUT)
    tools = await executor.list_tools()
    errors: list[str | None] = []
    for call in calls:
        try:
            text = await executor.call_tool(
                call["tool_name"], call.get("arguments") or {}
            )
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
            continue
        errors.append(text if text.startswith(TOOL_ERROR_PREFIX) else None)
    return tools, errors


def replay(
    ctx: ReplayContext,
    code: str,
    db_scenario: str,
    calls: list[dict],
    verifier_code: str,
    final_answer: str,
) -> Replay:
    """Make the calls on a fresh server and database, then grade the database.

    The server runs the way `awm agent` runs it, on the database of db_scenario.
    """
    run_dir = os.path.join(ctx.work_dir, uuid.uuid4().hex)
    os.makedirs(run_dir)
    envs_path = os.path.join(run_dir, "env.jsonl")
    tools_jsonl_save([{"scenario": db_scenario, "full_code": code}], envs_path)

    async def session() -> tuple[RunningServer, list[dict], list[str | None]]:
        async with serve_scenario(
            db_scenario,
            envs_path,
            run_dir,
            db_schema_path=ctx.db_path,
            sample_path=ctx.sample_path,
            timeout=SERVER_START_TIMEOUT,
        ) as server:
            tools, errors = await _make_calls(server.url, calls)
        return server, tools, errors

    server, tools, errors = asyncio.run(session())
    entry = {"verification": {"code": verifier_code}}
    reward, _ = run_verifier(
        entry, "code", server.initial_db_path, server.db_path, final_answer, [], []
    )
    return Replay(tools=tools, errors=errors, reward=reward)


# --------------------------------------------------------- server code


@dataclass
class CheckTask:
    """A source task the rebuilt server has to solve as the original one did."""

    draft: Draft
    # error status of each reference call on the original server
    original_errors: list[bool]


@dataclass
class RebuiltEnv:
    source: ScenarioSource
    part: str
    renames: Renames
    code: str
    tools: list[dict]
    tasks: list[CheckTask]


def rename_operation_ids(code: str, renames: Renames) -> str:
    """The code with operation_id="<old>" literals set to the new tool names."""
    for old, new in renames.tools.items():
        pattern = rf"""(operation_id\s*=\s*)(["']){re.escape(old)}\2"""
        code = re.sub(pattern, lambda m, new=new: f"{m[1]}{m[2]}{new}{m[2]}", code)
    return code


def apply_edits(code: str, edits: object) -> str | Rejection:
    if not isinstance(edits, list) or not edits:
        return Rejection('"edits" must be a non-empty list.')
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict) or not all(
            isinstance(edit.get(k), str) for k in ("old", "new")
        ):
            return Rejection(f'Edit #{i + 1} needs string "old" and "new".')
        count = code.count(edit["old"])
        if count != 1:
            return Rejection(
                f'The "old" text of edit #{i + 1} occurs {count} times in the code after the earlier edits; it must occur exactly once: {edit["old"][:200]!r}'
            )
        code = code.replace(edit["old"], edit["new"])
    return code


def _string_literals(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _listed_text(tool: dict) -> str:
    texts = [tool.get("description") or ""]
    for schema in properties(tool).values():
        texts.append(str(schema.get("description") or ""))
    return "\n".join(texts)


def listing_problems(
    source: ScenarioSource, renames: Renames, tools: list[dict]
) -> list[str]:
    """How the listed tools differ from the renamed original tools."""
    problems = []
    listed = {t["name"]: t for t in tools}
    expected = {renames.tool(t["name"]): t for t in source.tools}
    if set(listed) != set(expected):
        problems.append(
            f"The server lists the tools {sorted(listed)}, expected {sorted(expected)}"
        )
    for name, original in expected.items():
        if name not in listed:
            continue
        params = renames.params.get(original["name"], {})
        want = {params.get(p, p) for p in properties(original)}
        got = set(properties(listed[name]))
        if got != want:
            problems.append(
                f"Tool {name} has the parameters {sorted(got)}, expected {sorted(want)}"
            )
        old = [
            o for o in renames.tools if mentions(_listed_text(listed[name]), o, True)
        ]
        if old:
            problems.append(f"The description of {name} still mentions {old}")
    return problems


def replay_problems(
    ctx: ReplayContext,
    rebuilt_code: str,
    source: ScenarioSource,
    renames: Renames,
    tasks: list[CheckTask],
) -> tuple[list[dict], list[str]]:
    """Run the renamed reference calls; the listed tools and what went wrong."""
    tools: list[dict] = []
    problems = []
    for task in tasks:
        draft = task.draft
        calls = [renames.call(c) for c in draft.reference_calls or []]
        result = replay(
            ctx,
            rebuilt_code,
            source.scenario,
            calls,
            draft.source_verifier_code,
            draft.reference_answer,
        )
        tools = result.tools
        for i, (call, error, was_error) in enumerate(
            zip(calls, result.errors, task.original_errors, strict=True)
        ):
            if (error is not None) != was_error:
                outcome = (
                    f"failed: {error[:MAX_FEEDBACK_CHARS]}" if error else "succeeded"
                )
                expected = "failed" if was_error else "succeeded"
                problems.append(
                    f"Call #{i + 1} {json.dumps(call, ensure_ascii=False)} {outcome}, but it {expected} on the original server"
                )
                break
        else:
            if result.reward != "complete":
                problems.append(
                    f"After the calls {json.dumps(calls, ensure_ascii=False)} the database is not what the task needs (verifier: {result.reward})"
                )
    return tools, problems


def code_step(
    ctx: ReplayContext,
    source: ScenarioSource,
    part: str,
    renames: Renames,
    tasks: list[CheckTask],
) -> LLMStep[RebuiltEnv]:
    """Ask for the edits that rename the server, and check the rebuilt server."""
    base = rename_operation_ids(source.code, renames)

    def finish(response: dict) -> RebuiltEnv | Rejection:
        code = apply_edits(base, response.get("edits"))
        if isinstance(code, Rejection):
            return code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return Rejection(f"The edited code does not parse: {e}")
        old = [
            o
            for o in renames.tools
            if any(mentions(s, o, True) for s in _string_literals(tree))
        ]
        if old:
            return Rejection(f"String literals of the code still mention {old}")
        try:
            tools, problems = replay_problems(ctx, code, source, renames, tasks)
        except ServerStartError as e:
            return Rejection(
                f"The server does not start. Its output ends with:\n{e.output[-MAX_FEEDBACK_CHARS:]}"
            )
        problems = listing_problems(source, renames, tools) + problems
        if problems:
            return Rejection("\n".join(problems))
        return RebuiltEnv(source, part, renames, code, tools, tasks)

    def check_prompt(rebuilt: RebuiltEnv) -> str:
        diff = difflib.unified_diff(
            source.code.splitlines(),
            rebuilt.code.splitlines(),
            "before.py",
            "after.py",
            lineterm="",
        )
        return AUG_ENV_RENAME_CHECK_PROMPT.format(
            renames_json=renames.as_json(), diff="\n".join(diff)
        )

    prompt = AUG_ENV_RENAME_PROMPT.format(renames_json=renames.as_json(), code=base)
    return LLMStep(
        [{"role": "user", "content": prompt}],
        finish,
        label=f"{source.scenario} {part} server",
        check_prompt=check_prompt,
    )


# ---------------------------------------------------------------- tasks


def env_scenario(source_scenario: str, part: str) -> str:
    return f"{source_scenario}{SCENARIO_SEPARATOR}{part}"


def task_step(rebuilt: RebuiltEnv, task: CheckTask) -> Draft | LLMStep[Draft]:
    """The source task for the rebuilt environment, rewritten if it names a renamed tool."""
    source = task.draft
    draft = Draft(
        scenario=env_scenario(rebuilt.source.scenario, rebuilt.part),
        source_task_idx=source.source_task_idx,
        task=source.task,
        source_verifier_code=source.source_verifier_code,
        tools=rebuilt.tools,
        reference_calls=[rebuilt.renames.call(c) for c in source.reference_calls or []],
        reference_answer=source.reference_answer,
        part=rebuilt.part,
    )
    mentioned = rebuilt.renames.mentioned_in(source.task)
    if not mentioned:
        return draft
    rewritten = dataclasses.replace(
        draft,
        banned_mentions=sorted(mentioned),
        required_mentions=sorted(set(mentioned.values())),
    )

    def finish(response: dict) -> Draft | Rejection:
        return accept_task(rewritten, response)

    prompt = AUG_TASK_RENAME_PROMPT.format(
        task=source.task,
        renames=json.dumps(mentioned, ensure_ascii=False),
        keep_values=json.dumps(keep_values(rewritten), ensure_ascii=False),
    )
    return rewrite_step(rewritten, prompt, finish)


def to_variant(draft: Draft, source_scenario: str) -> Variant:
    meta = TaskMeta(
        augmentations=[draft.part],
        source_task_idx=draft.source_task_idx,
        source_scenario=source_scenario,
    )
    return Variant(
        scenario=draft.scenario,
        task=draft.task,
        meta=meta,
        verifier_code=draft.source_verifier_code,
    )


# ------------------------------------------------------------ pipeline


@dataclass
class EnvVariant:
    """A rebuilt environment and its tasks, ready to be added as a scenario."""

    scenario: str
    source_scenario: str
    part: str
    code: str
    variants: list[Variant]


RunTaskLLM = Callable[[list[LLMStep[Draft]]], list[Draft]]


def check_tasks(
    ctx: ReplayContext,
    source: ScenarioSource,
    num_tasks: int,
    rng: random.Random,
    oversample: int = 2,
) -> list[CheckTask]:
    """Up to num_tasks solved tasks whose reference calls solve them on a replay.

    A run of the agent may have depended on something a replay does not
    reproduce (e.g. the time), so tasks are kept only if the original server
    solves them again.
    """
    drafts = [d for d in source.drafts if d.reference_calls]
    rng.shuffle(drafts)
    tasks = []
    for draft in drafts[: num_tasks * oversample]:
        if len(tasks) == num_tasks:
            break
        try:
            result = replay(
                ctx,
                source.code,
                source.scenario,
                draft.reference_calls or [],
                draft.source_verifier_code,
                draft.reference_answer,
            )
        except ServerStartError as e:
            logger.warning(
                f"{source.scenario}: the original server does not start: {e}"
            )
            return []
        if result.reward == "complete":
            tasks.append(CheckTask(draft, [e is not None for e in result.errors]))
    return tasks


def build_env_variants(
    part_names: list[str],
    sources: list[ScenarioSource],
    ctx: ReplayContext,
    complete: Complete,
    run_task_llm: RunTaskLLM,
    num_per_part: int,
    seed: int,
    max_generations: int,
    workers: int,
) -> list[EnvVariant]:
    """A rebuilt environment per scenario and part, with up to num_per_part tasks."""
    if not part_names:
        return []
    os.makedirs(ctx.work_dir, exist_ok=True)

    def tasks_for(source: ScenarioSource) -> list[CheckTask]:
        rng = random.Random(f"{seed}:{normalize_scenario_name(source.scenario)}")
        return check_tasks(ctx, source, num_per_part, rng)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        tasks_of = dict(
            zip(
                (s.scenario for s in sources),
                pool.map(tasks_for, sources),
                strict=True,
            )
        )
    usable = [s for s in sources if tasks_of[s.scenario]]
    logger.info(f"{len(usable)}/{len(sources)} scenarios have tasks to check a rebuild")

    env_variants: list[EnvVariant] = []
    for name in part_names:
        part = ENV_PARTS[name]
        renames: dict[str, Renames] = {}
        name_steps: list[LLMStep[tuple[str, Renames]]] = []
        for source in usable:
            rng = random.Random(
                f"{seed}:{name}:{normalize_scenario_name(source.scenario)}"
            )
            out = part.renames(source, rng)
            if isinstance(out, LLMStep):
                name_steps.append(_tagged(out, source.scenario))
            elif out is not None:
                renames[source.scenario] = out
        renames.update(dict(refine(name_steps, complete, max_generations)))

        steps = [
            code_step(ctx, s, name, renames[s.scenario], tasks_of[s.scenario])
            for s in usable
            if s.scenario in renames
        ]
        rebuilt_envs = refine(steps, complete, max_generations, workers=workers)

        task_steps: list[LLMStep[Draft]] = []
        drafts: list[Draft] = []
        for rebuilt in rebuilt_envs:
            for task in rebuilt.tasks:
                out = task_step(rebuilt, task)
                if isinstance(out, LLMStep):
                    task_steps.append(out)
                else:
                    drafts.append(out)
        drafts.extend(run_task_llm(task_steps))

        for rebuilt in rebuilt_envs:
            scenario = env_scenario(rebuilt.source.scenario, name)
            mine = sorted(
                (d for d in drafts if d.scenario == scenario),
                key=lambda d: d.source_task_idx,
            )
            if mine:
                env_variants.append(
                    EnvVariant(
                        scenario=scenario,
                        source_scenario=rebuilt.source.scenario,
                        part=name,
                        code=rebuilt.code,
                        variants=[to_variant(d, rebuilt.source.scenario) for d in mine],
                    )
                )
        logger.info(
            f"Part {name}: {len(rebuilt_envs)} environments rebuilt from {len(steps)} scenarios"
        )
    return env_variants


def _tagged(step: LLMStep[Renames], scenario: str) -> LLMStep[tuple[str, Renames]]:
    """The step with its result paired with the scenario it belongs to."""

    def finish(response: dict) -> tuple[str, Renames] | Rejection:
        out = step.finish(response)
        return out if isinstance(out, Rejection) else (scenario, out)

    return LLMStep(step.messages, finish, label=step.label)


# ------------------------------------------------------ dataset assembly


def merge_env_variants(
    rows: dict[str, list[dict]],
    env_verifier_rows: list[dict],
    env_variants: list[EnvVariant],
    keep_sources: set[str],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Scenario files and verifier rows of rebuilt environments, replaced.

    rows: rows of the "envs", "db", "sample" and "tasks" files.
    env_verifier_rows: verifier rows of rebuilt environments.
    Rebuilt environments of scenarios in keep_sources (normalized) stay; the
    others are dropped, and env_variants are added.
    """

    def kept(row: dict) -> bool:
        return (
            not is_rebuilt_env_row(row)
            or normalize_scenario_name(row["source_scenario"]) in keep_sources
        )

    new_rows = {kind: [r for r in rs if kept(r)] for kind, rs in rows.items()}
    new_verifiers = [r for r in env_verifier_rows if kept(r)]

    def source_row(kind: str, scenario: str) -> dict:
        norm = normalize_scenario_name(scenario)
        return next(
            r
            for r in rows[kind]
            if not is_rebuilt_env_row(r)
            and normalize_scenario_name(r["scenario"]) == norm
        )

    for env in env_variants:
        marker = {
            "scenario": env.scenario,
            "source_scenario": env.source_scenario,
            "augmentation": env.part,
        }
        new_rows["envs"].append(
            {**source_row("envs", env.source_scenario), **marker, "full_code": env.code}
        )
        for kind in ("db", "sample"):
            new_rows[kind].append({**source_row(kind, env.source_scenario), **marker})
        new_rows["tasks"].append(
            {
                **marker,
                "tasks": [v.task for v in env.variants],
                "task_meta": [v.meta.model_dump() for v in env.variants],
            }
        )
        for idx, variant in enumerate(env.variants):
            new_verifiers.append(
                {
                    "scenario": env.scenario,
                    "task_idx": idx,
                    "task": variant.task,
                    "verification": {"code": variant.verifier_code},
                    "augmented": True,
                    "augmentations": variant.meta.augmentations,
                    "source_task_idx": variant.meta.source_task_idx,
                    "source_scenario": env.source_scenario,
                }
            )
    return new_rows, new_verifiers
