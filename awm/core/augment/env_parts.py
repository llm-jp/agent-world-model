"""Parts that rename tools or parameters by rebuilding the scenario's environment.

The renamed tools are real: the server code is rewritten with the new names
(operation ids, parameters, descriptions, error messages), so nothing the agent
sees still uses an old name. The drafts of a scenario carry the rewritten code,
the tools it lists and their reference calls under the new names, and tasks
that name a renamed tool are rewritten. The recipe's environment keeps the
source's database schema, sample data and verifiers.

A rewrite of the server code is accepted only if the server starts, lists
exactly the renamed tools and parameters, keeps no old tool name in what it
lists or in its string literals, and still solves the source tasks: their
reference calls, renamed, fail where they failed on the original server and
leave a database the task's verifier grades complete.
"""

import ast
import dataclasses
import difflib
import json
import keyword
import logging
import random
import re
import string
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Rejection,
    must_keep,
    properties,
)
from awm.core.augment.part import Part, PartContext
from awm.core.augment.replay import (
    ReplayContext,
    replay,
    replayed_drafts,
    tool_set_problems,
)
from awm.core.augment.task_parts import (
    rewrite_step,
)
from awm.core.server import ServerStartError
from awm.prompts import (
    AUG_ENV_RENAME_CHECK_PROMPT,
    AUG_ENV_RENAME_PROMPT,
    AUG_PARAM_RENAME_PROMPT,
    AUG_TASK_RENAME_PROMPT,
    AUG_TOOL_RENAME_PROMPT,
)
from awm.tools import normalize_scenario_name

logger = logging.getLogger(__name__)

# length range of the names tool_rename_random draws, inclusive
MIN_RANDOM_NAME_LENGTH = 4
MAX_RANDOM_NAME_LENGTH = 14
# names that function-calling APIs accept
TOOL_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
# longest server output or error text sent back to the LLM
MAX_FEEDBACK_CHARS = 1500
# Part.writes entries of the renaming parts
TOOL_NAMES = "tool_names"
PARAM_NAMES = "param_names"


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
        ones like "note" are ordinary words.
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
    """A scenario's environment to rebuild, as the drafts of the recipe have it.

    drafts: solved tasks whose reference calls solve them on a replay of code
    (Draft.call_errors set); they check the rebuild.
    """

    scenario: str
    code: str
    tools: list[dict]
    drafts: list[Draft]

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]


# ---------------------------------------------------------------- names


def name_map(
    response: dict, key: str, expected: set[str], what: str
) -> dict[str, str] | Rejection:
    """The {old name: new name} map under key in the reply, the new names stripped.

    what: the kind of the names ("tool", "parameter"), for the feedback.
    """
    names = response.get(key)
    if not isinstance(names, dict) or set(names) != expected:
        return Rejection(
            f'"{key}" must map exactly these {what} names to new names: {sorted(expected)}'
        )
    if not all(isinstance(v, str) for v in names.values()):
        return Rejection("Every new name must be a string.")
    return {old: new.strip() for old, new in names.items()}


def name_problems(renames: dict[str, str], what: str, keywords: bool) -> list[str]:
    """What makes new names unusable in themselves.

    what: the kind of the names ("tool", "parameter").
    keywords: whether a Python keyword can be a name.
    """
    rule = "must start with a letter and use only letters, digits and _ (at most 64 characters)"
    if not keywords:
        rule = "must start with a letter, use only letters, digits and _ (at most 64 characters) and not be a Python keyword"
    problems = []
    for old, new in renames.items():
        if not TOOL_NAME_PATTERN.fullmatch(new) or (
            not keywords and keyword.iskeyword(new)
        ):
            problems.append(f"{new!r} {rule}")
        if new.replace("_", "").lower() == old.replace("_", "").lower():
            problems.append(f"{new!r} only changes the case or separators of {old!r}")
    if len(set(renames.values())) < len(renames):
        problems.append(f"two {what}s got the same new name")
    return problems


def rename_problems(tool_names: list[str], renames: dict[str, str]) -> list[str]:
    """What makes the new tool names unusable."""
    problems = name_problems(renames, "tool", keywords=True)
    for old, new in renames.items():
        if new in tool_names and new != old:
            problems.append(f"{new!r} is already the name of another tool")
    return problems


class EnvPart(Part):
    """Renames of one kind, built into the server code; subclasses implement renames.

    apply rebuilds each scenario once: it picks the renames, has the LLM edit
    the server code until the rebuilt server passes the checks, and moves the
    scenario's drafts onto the rebuilt server.
    """

    rebuilds_env = True
    needs_reference = True
    needs_tools = True

    @abstractmethod
    def renames(
        self, source: ScenarioSource, rng: random.Random
    ) -> Renames | LLMStep[Renames] | None:
        """The new names, a step that asks the LLM for them, or None if nothing to rename."""

    def apply(self, drafts: list[Draft], ctx: PartContext) -> list[Draft]:
        sources = scenario_sources(self, drafts, ctx)
        if not sources:
            return []

        renames: dict[str, Renames] = {}
        name_steps: list[LLMStep[tuple[str, Renames]]] = []
        for source in sources:
            rng = ctx.rng(self.name, normalize_scenario_name(source.scenario))
            out = self.renames(source, rng)
            if isinstance(out, LLMStep):
                name_steps.append(_tagged(out, source.scenario))
            elif out is not None:
                renames[source.scenario] = out
        renames.update(dict(ctx.run_llm(name_steps)))

        steps = [
            code_step(ctx.replay, s, self.name, renames[s.scenario])
            for s in sources
            if s.scenario in renames
        ]
        rebuilt_envs = ctx.run_llm(steps, workers=ctx.workers)
        logger.info(
            f"Part {self.name}: {len(rebuilt_envs)} environments rebuilt from {len(steps)} scenarios"
        )

        produced: list[Draft] = []
        task_steps: list[LLMStep[Draft]] = []
        for rebuilt in rebuilt_envs:
            for draft in rebuilt.source.drafts:
                out = task_step(rebuilt, draft)
                if isinstance(out, LLMStep):
                    task_steps.append(out)
                else:
                    produced.append(out)
        return produced + ctx.run_llm(task_steps)


def scenario_sources(
    part: Part, drafts: list[Draft], ctx: PartContext
) -> list[ScenarioSource]:
    """The scenarios of the drafts, each with the drafts that check a rebuild.

    Every draft the server solves on a replay is kept, so the later parts of
    the recipe can still choose among them. The drafts of a scenario share its
    server: the source scenario's code, or the code an earlier part of the
    recipe rebuilt. Scenarios whose server solves none of the drafts are left
    out.
    """
    by_scenario: dict[str, list[Draft]] = {}
    for draft in drafts:
        if part.applicable(draft):
            by_scenario.setdefault(normalize_scenario_name(draft.scenario), []).append(
                draft
            )
    candidates = []
    for scenario, members in by_scenario.items():
        code = members[0].code or ctx.env_code.get(scenario)
        if code is None:
            continue
        assert all(d.code == members[0].code for d in members), scenario
        candidates.append((code, members))
    if not candidates:
        return []

    def checked(candidate: tuple[str, list[Draft]]) -> ScenarioSource | None:
        code, members = candidate
        first = members[0]
        try:
            kept = replayed_drafts(ctx.replay, code, members)
        except ServerStartError as e:
            logger.warning(f"{first.scenario}: the server does not start: {e}")
            return None
        if not kept:
            return None
        return ScenarioSource(first.scenario, code, first.tools or [], kept)

    with ThreadPoolExecutor(max_workers=max(1, ctx.workers)) as pool:
        sources = [s for s in pool.map(checked, candidates) if s is not None]
    logger.info(
        f"Part {part.name}: {len(sources)}/{len(candidates)} scenarios have tasks to check a rebuild"
    )
    return sources


class ToolRename(EnvPart):
    """Let the LLM give every tool a new name that still describes what it does."""

    name = "tool_rename"
    writes = frozenset({TOOL_NAMES})

    def renames(self, source: ScenarioSource, rng: random.Random) -> LLMStep[Renames]:
        docs = {t["name"]: t.get("description", "") for t in source.tools}

        def finish(response: dict) -> Renames | Rejection:
            names = name_map(response, "tools", set(docs), "tool")
            if isinstance(names, Rejection):
                return names
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
    writes = frozenset({TOOL_NAMES})

    def renames(self, source: ScenarioSource, rng: random.Random) -> Renames:
        taken = set(source.tool_names)
        names = {}
        for old in source.tool_names:
            new = old
            # a word of the code (e.g. "text") would be taken for the tool name
            while new in taken or mentions(source.code, new, ignore_case=True):
                length = rng.randint(MIN_RANDOM_NAME_LENGTH, MAX_RANDOM_NAME_LENGTH)
                new = "".join(rng.choices(string.ascii_lowercase, k=length))
            taken.add(new)
            names[old] = new
        return Renames(tools=names)


def called_params(source: ScenarioSource) -> dict[str, list[str]]:
    """Parameters, per tool, that the solved tasks' reference calls pass."""
    taken = {t["name"]: properties(t) for t in source.tools}
    used: dict[str, list[str]] = {}
    for draft in source.drafts:
        for call in draft.reference_calls or []:
            params = taken.get(call["tool_name"], {})
            for name in call.get("arguments") or {}:
                if name in params and name not in used.get(call["tool_name"], []):
                    used.setdefault(call["tool_name"], []).append(name)
    return used


def param_rename_problems(
    source: ScenarioSource, used: dict[str, list[str]], renames: dict[str, str]
) -> list[str]:
    """What makes the new parameter names unusable."""
    problems = name_problems(renames, "parameter", keywords=False)
    taken = {t["name"]: set(properties(t)) for t in source.tools}
    for tool, params in used.items():
        clash = sorted(taken[tool] & {renames[p] for p in params})
        if clash:
            problems.append(f"{clash} are already parameter names of {tool}")
    return problems


class ParamRename(EnvPart):
    """Let the LLM give the parameters that the solved tasks pass new names."""

    name = "param_rename"
    writes = frozenset({PARAM_NAMES})

    def renames(
        self, source: ScenarioSource, rng: random.Random
    ) -> LLMStep[Renames] | None:
        used = called_params(source)
        if not used:
            return None
        tools = {t["name"]: t for t in source.tools}
        docs: dict[str, dict[str, str]] = {}
        for tool, params in used.items():
            for param in params:
                schema = properties(tools[tool])[param]
                docs.setdefault(param, {})[tool] = str(schema.get("description") or "")

        def finish(response: dict) -> Renames | Rejection:
            names = name_map(response, "params", set(docs), "parameter")
            if isinstance(names, Rejection):
                return names
            if problems := param_rename_problems(source, used, names):
                return Rejection("; ".join(problems))
            return Renames(
                params={
                    tool: {p: names[p] for p in params} for tool, params in used.items()
                }
            )

        prompt = AUG_PARAM_RENAME_PROMPT.format(
            params_json=json.dumps(docs, ensure_ascii=False)
        )
        return LLMStep(
            [{"role": "user", "content": prompt}],
            finish,
            label=f"{source.scenario} parameter names",
        )


# --------------------------------------------------------- server code


@dataclass
class RebuiltEnv:
    source: ScenarioSource
    part: str
    renames: Renames
    code: str
    tools: list[dict]


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


def _canonical_lines(code: str, renames: Renames) -> list[str]:
    """The code with every mention of a renamed name written as the new name.

    Tool names count in any case and with spaces for underscores ("Create
    order" for create_order), old or new; parameter names only as written,
    with the alias= that accepts a new parameter name dropped. Python function
    names are left out, since the client does not see them. Line breaks stay,
    so line i is line i of the code.
    """
    code = re.sub(r"(\bdef[ \t]+)[A-Za-z_][A-Za-z0-9_]*", r"\1_", code)
    forms: dict[str, str] = {}
    for old, new in renames.tools.items():
        for form in (old, old.replace("_", " "), new, new.replace("_", " ")):
            forms[form.lower()] = new
    # longer forms first, so that a name inside a longer one is left to it
    for form in sorted(forms, key=len, reverse=True):
        code = re.sub(_word(form), forms[form], code, flags=re.IGNORECASE)
    for old, new in renames.renamed_params().items():
        code = re.sub(_word(old), new, code)
        alias = rf"""alias[ \t]*=[ \t]*(["']){re.escape(new)}\1"""
        code = re.sub(rf"{alias}[ \t]*,[ \t]*", "", code)
        code = re.sub(rf",[ \t]*{alias}", "", code)
        code = re.sub(alias, "", code)
    return [line.rstrip() for line in code.splitlines()]


def unrelated_change_problems(
    source_code: str, code: str, renames: Renames
) -> list[str]:
    """Changes of the code that are not a renamed name, as feedback."""
    before = _canonical_lines(source_code, renames)
    after = _canonical_lines(code, renames)
    if before == after:
        return []
    source_lines, lines = source_code.splitlines(), code.splitlines()
    changes = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "equal":
            old = "\n".join(source_lines[i1:i2])
            new = "\n".join(lines[j1:j2])
            changes.append(f"line {i1 + 1}: {old.strip()!r} -> {new.strip()!r}")
    text = "\n".join(changes)
    if len(text) > MAX_FEEDBACK_CHARS:
        text = text[:MAX_FEEDBACK_CHARS] + "\n..."
    head = (
        "These changes are not renames of a tool or parameter name. Undo them: "
        "only the names themselves may change, and other words stay as they were, "
        "even words that are part of a name."
    )
    return [f"{head}\n{text}"]


def _listed_text(tool: dict) -> str:
    texts = [tool.get("description") or ""]
    for schema in properties(tool).values():
        texts.append(str(schema.get("description") or ""))
    return "\n".join(texts)


def listing_problems(
    source: ScenarioSource, renames: Renames, tools: list[dict]
) -> list[str]:
    """How the listed tools differ from the renamed original tools."""
    listed = {t["name"]: t for t in tools}
    expected = {renames.tool(t["name"]): t for t in source.tools}
    problems = tool_set_problems(set(listed), set(expected))
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
) -> tuple[list[dict], list[str]]:
    """Run the renamed reference calls; the listed tools and what went wrong."""
    tools: list[dict] = []
    problems = []
    for draft in source.drafts:
        assert draft.call_errors is not None
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
            zip(calls, result.errors, draft.call_errors, strict=True)
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
        if problems := unrelated_change_problems(source.code, code, renames):
            return Rejection("\n".join(problems))
        try:
            tools, problems = replay_problems(ctx, code, source, renames)
        except ServerStartError as e:
            return Rejection(
                f"The server does not start. Its output ends with:\n{e.output[-MAX_FEEDBACK_CHARS:]}"
            )
        problems = listing_problems(source, renames, tools) + problems
        if problems:
            return Rejection("\n".join(problems))
        return RebuiltEnv(source, part, renames, code, tools)

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


def task_step(rebuilt: RebuiltEnv, draft: Draft) -> Draft | LLMStep[Draft]:
    """The draft on the rebuilt server, its task rewritten if it names a renamed tool."""
    moved = draft.evolve(
        rebuilt.part,
        code=rebuilt.code,
        tools=rebuilt.tools,
        reference_calls=[rebuilt.renames.call(c) for c in draft.reference_calls or []],
    )
    mentioned = rebuilt.renames.mentioned_in(draft.task)
    if not mentioned:
        return moved
    rewritten = dataclasses.replace(
        moved,
        banned_mentions=[*draft.banned_mentions, *sorted(mentioned)],
        required_mentions=[
            *(m for m in draft.required_mentions if m not in mentioned),
            *sorted(set(mentioned.values())),
        ],
    )

    prompt = AUG_TASK_RENAME_PROMPT.format(
        task=draft.task,
        renames=json.dumps(mentioned, ensure_ascii=False),
        keep_values=json.dumps(must_keep(rewritten), ensure_ascii=False),
    )
    return rewrite_step(rewritten, prompt)


def _tagged(step: LLMStep[Renames], scenario: str) -> LLMStep[tuple[str, Renames]]:
    """The step with its result paired with the scenario it belongs to."""

    def finish(response: dict) -> tuple[str, Renames] | Rejection:
        out = step.finish(response)
        return out if isinstance(out, Rejection) else (scenario, out)

    return LLMStep(step.messages, finish, label=step.label)
