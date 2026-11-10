"""Write the dataset with augmented tasks to a directory of its own.

The input files are left as they are. The output directory gets files of the
same names (gen_tasks.jsonl, the verifier file, gen_envs.jsonl, ...), and
together they are a whole dataset so that RL reads a single one: augmented
tasks are appended to each scenario's `tasks`, with their settings in a
parallel `task_meta` list (None for original tasks), and their verifiers follow
the original ones under the same (scenario, task_idx).

Each augmentation is a part applied on its own to a source task. Parts that
rename tools or parameters (env_parts.py) rebuild the environment instead and
add it as a scenario of its own to every scenario file.
"""

import logging
import os
from dataclasses import dataclass

from awm.core.augment.compose import (
    PARTS,
    check_part_names,
    existing_variants,
    generate_variants,
    merge_variants,
)
from awm.core.augment.draft import Draft, LLMStep
from awm.core.augment.env_parts import (
    ENV_PARTS,
    ReplayContext,
    ScenarioSource,
    build_env_variants,
    merge_env_variants,
)
from awm.core.augment.refine import Complete, refine
from awm.gpt import GPTClient
from awm.tools import (
    is_rebuilt_env_row,
    normalize_scenario_name,
    strip_augmented,
    tools_jsonl_load,
    tools_jsonl_save,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # gen_tasks.jsonl
    input_task: str
    # code-mode verifier file
    input_verifier: str
    # gen_reference.jsonl from `awm gen reference`
    input_reference: str
    # gen_tools.jsonl from `awm gen reference`
    input_tools: str
    # scenario files; rebuilt environments are added to their outputs
    envs_path: str = "./outputs/gen_envs.jsonl"
    db_path: str = "./outputs/gen_db.jsonl"
    sample_path: str = "./outputs/gen_sample.jsonl"
    # the dataset with augmented data, under the input file names;
    # defaults to <input_task dir>_aug, e.g. outputs/<model>_ja -> outputs/<model>_ja_aug
    output_dir: str | None = None
    # servers and databases of the replays that check rebuilt environments;
    # defaults to <output_dir>/aug_env_runs
    work_dir: str | None = None
    # comma-separated part names, each applied on its own; defaults to every part
    parts: list[str] | None = None
    # augmented tasks per scenario and part
    num_per_part: int = 2
    seed: int = 0
    model: str = "your-llm-model-name"
    allowed_scenarios: list[str] | None = None
    # replies generated per LLM rewrite before it is dropped, feedback included
    max_generations: int = 3
    # concurrent LLM requests, and concurrent replays of rebuilt environments
    max_concurrency: int = 32

    def input_paths(self) -> list[str]:
        """Inputs that are written again, with augmented data, to output_dir."""
        return [
            self.input_task,
            self.input_verifier,
            self.envs_path,
            self.db_path,
            self.sample_path,
        ]

    def pre_process(self):
        for path in [
            self.input_task,
            self.input_verifier,
            self.input_reference,
            self.input_tools,
            self.envs_path,
            self.db_path,
            self.sample_path,
        ]:
            assert os.path.exists(path), f"{path} not found"
        if self.parts is None:
            self.parts = [*PARTS, *ENV_PARTS]
        check_part_names(self.parts)
        if self.output_dir is None:
            input_dir = os.path.abspath(os.path.dirname(self.input_task))
            self.output_dir = os.path.relpath(f"{input_dir}_aug")
        names = [os.path.basename(p) for p in self.input_paths()]
        assert len(set(names)) == len(names), f"input file names collide: {names}"
        for path in self.input_paths():
            assert os.path.abspath(output_path(path, self.output_dir)) != (
                os.path.abspath(path)
            ), f"output_dir {self.output_dir} would overwrite {path}"
        if self.work_dir is None:
            self.work_dir = os.path.join(self.output_dir, "aug_env_runs")
        if self.allowed_scenarios:
            self.allowed_scenarios = sorted(
                {normalize_scenario_name(name) for name in self.allowed_scenarios}
            )
        if override_model := os.environ.get("AWM_SYN_OVERRIDE_MODEL"):
            self.model = override_model


def load_drafts(config: Config, task_items: list[dict]) -> list[Draft]:
    """A starting draft for every original task that has a verifier."""
    verifiers = {
        (normalize_scenario_name(r["scenario"]), r["task_idx"]): (
            r.get("verification") or {}
        ).get("code")
        for r in tools_jsonl_load(config.input_verifier)
        if not r.get("augmented")
    }
    references = {
        (normalize_scenario_name(r["scenario"]), r["task_idx"]): r
        for r in tools_jsonl_load(config.input_reference)
        if r.get("result") == "complete"
    }
    tools = {
        normalize_scenario_name(r["scenario"]): r["tools"]
        for r in tools_jsonl_load(config.input_tools)
    }

    drafts = []
    for item in task_items:
        scenario = normalize_scenario_name(item["scenario"])
        if config.allowed_scenarios and scenario not in config.allowed_scenarios:
            continue
        other_tools = [t for name, ts in tools.items() if name != scenario for t in ts]
        for task_idx, task in enumerate(strip_augmented(item)):
            code = verifiers.get((scenario, task_idx))
            if not code:
                continue
            reference = references.get((scenario, task_idx)) or {}
            drafts.append(
                Draft(
                    scenario=item["scenario"],
                    source_task_idx=task_idx,
                    task=task,
                    source_verifier_code=code,
                    tools=tools.get(scenario),
                    reference_calls=reference.get("tool_calls"),
                    reference_answer=reference.get("final_answer") or "",
                    other_tools=other_tools,
                )
            )
    return drafts


def scenario_sources(drafts: list[Draft], env_rows: list[dict]) -> list[ScenarioSource]:
    """Scenarios that have an environment, a tool list and solved tasks."""
    envs = {normalize_scenario_name(r["scenario"]): r for r in env_rows}
    by_scenario: dict[str, list[Draft]] = {}
    for draft in drafts:
        by_scenario.setdefault(normalize_scenario_name(draft.scenario), []).append(
            draft
        )
    sources = []
    for scenario, scenario_drafts in by_scenario.items():
        solved = [d for d in scenario_drafts if d.reference_calls]
        tools = scenario_drafts[0].tools
        if scenario in envs and tools and solved:
            sources.append(
                ScenarioSource(
                    scenario_drafts[0].scenario, envs[scenario], tools, solved
                )
            )
    return sources


def llm_complete(config: Config) -> Complete:
    """Send conversations to the LLM in one batch."""
    client: GPTClient | None = None

    def complete(conversations: list[list[dict]]) -> list[str]:
        nonlocal client
        if client is None:
            assert config.model != "your-llm-model-name", (
                "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"
            )
            client = GPTClient(timeout=1200, concurrency_limit=config.max_concurrency)
        return client.batch_chat_completion(
            [
                {
                    "messages": messages,
                    "model": config.model,
                    "temperature": 1.0,
                    "max_tokens": 8192,
                }
                for messages in conversations
            ]
        )

    return complete


def output_path(path: str, output_dir: str) -> str:
    """The input file at path, moved to output_dir."""
    return os.path.join(output_dir, os.path.basename(path))


def load_previous(path: str) -> list[dict]:
    """Rows of an output written by an earlier run, if any."""
    return tools_jsonl_load(path) if os.path.exists(path) else []


def run(config: Config):
    assert config.parts is not None
    assert config.output_dir is not None and config.work_dir is not None
    inputs = {
        "tasks": config.input_task,
        "verifiers": config.input_verifier,
        "envs": config.envs_path,
        "db": config.db_path,
        "sample": config.sample_path,
    }
    outputs = {
        kind: output_path(path, config.output_dir) for kind, path in inputs.items()
    }
    # only original rows are augmented; augmented data already in the inputs is skipped
    rows = {
        kind: [r for r in tools_jsonl_load(path) if not is_rebuilt_env_row(r)]
        for kind, path in inputs.items()
    }
    base_items = rows.pop("tasks")
    base_verifiers = rows.pop("verifiers")

    drafts = load_drafts(config, base_items)
    solved = sum(1 for d in drafts if d.reference_calls)
    logger.info(
        f"{len(drafts)} source tasks with a verifier, {solved} solved by the reference run"
    )

    complete = llm_complete(config)

    def run_llm(steps: list[LLMStep[Draft]]) -> list[Draft]:
        return refine(steps, complete, config.max_generations)

    task_parts = [p for p in config.parts if p in PARTS]
    variants = generate_variants(
        task_parts, drafts, config.num_per_part, config.seed, run_llm
    )
    # scenarios outside the selection keep what an earlier run wrote for them
    previous = {
        kind: load_previous(path) if config.allowed_scenarios else []
        for kind, path in outputs.items()
    }
    keep_sources: set[str] = set()
    if config.allowed_scenarios:
        others = [
            item
            for item in previous["tasks"]
            if not is_rebuilt_env_row(item)
            and normalize_scenario_name(item["scenario"])
            not in config.allowed_scenarios
        ]
        variants = existing_variants(others, previous["verifiers"]) + variants
        keep_sources = {normalize_scenario_name(item["scenario"]) for item in others}
    new_items, new_verifiers = merge_variants(base_items, base_verifiers, variants)

    env_variants = build_env_variants(
        [p for p in config.parts if p in ENV_PARTS],
        scenario_sources(drafts, rows["envs"]),
        ReplayContext(config.db_path, config.sample_path, config.work_dir),
        complete,
        run_llm,
        num_per_part=config.num_per_part,
        seed=config.seed,
        max_generations=config.max_generations,
        workers=config.max_concurrency,
    )

    def rebuilt(kind: str) -> list[dict]:
        return [r for r in previous[kind] if is_rebuilt_env_row(r)]

    new_rows, env_verifiers = merge_env_variants(
        {
            **{kind: rs + rebuilt(kind) for kind, rs in rows.items()},
            "tasks": new_items + rebuilt("tasks"),
        },
        rebuilt("verifiers"),
        env_variants,
        keep_sources,
    )

    new_rows["verifiers"] = new_verifiers + env_verifiers
    for kind, path in outputs.items():
        tools_jsonl_save(new_rows[kind], path)
        logger.info(f"Saved {len(new_rows[kind])} rows to {path}")

    counts: dict[str, int] = {}
    for variant in variants + [v for e in env_variants for v in e.variants]:
        key = variant.meta.augmentations[0]
        counts[key] = counts.get(key, 0) + 1
    logger.info(
        f"Added {sum(counts.values())} augmented tasks ({len(env_variants)} rebuilt environments): {counts}"
    )
