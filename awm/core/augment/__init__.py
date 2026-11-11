"""Write the dataset with augmented tasks to a directory of its own.

The input files are left as they are. The output directory gets files of the
same names (gen_tasks.jsonl, the verifier file, gen_envs.jsonl, ...), and
together they are a whole dataset so that RL reads a single one.

Each augmentation is a recipe: parts applied one after another to a source
task (recipe.py). A recipe's tasks for a source scenario form an environment of
their own, <scenario>__<recipe>, added to every scenario file (dataset.py) with
the settings of each task in a `task_meta` list. Parts that rename tools or
parameters (env_parts.py) also rewrite the environment's server code, and the
parts that change how the tools are presented (tool_parts.py) have the
server present them, so an RL trainer that reads only the scenario files sees
the augmented tools.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from awm.core.augment.dataset import merge_env_variants
from awm.core.augment.draft import Draft, LLMStep
from awm.core.augment.part import PartContext, RunLLM
from awm.core.augment.recipe import PARTS, parse_recipe, run_recipe
from awm.core.augment.refine import Complete, refine
from awm.core.augment.replay import ReplayContext
from awm.gpt import GPTClient
from awm.tools import (
    is_augmented_env_row,
    normalize_scenario_name,
    original_task_rows,
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
    # scenario files; augmented environments are added to their outputs
    envs_path: str = "./outputs/gen_envs.jsonl"
    db_path: str = "./outputs/gen_db.jsonl"
    sample_path: str = "./outputs/gen_sample.jsonl"
    # gen_scenario.jsonl, the list of scenarios an RL trainer offers;
    # defaults to the one next to envs_path
    scenario_path: str | None = None
    # the dataset with augmented data, under the input file names;
    # defaults to <input_task dir>_aug, e.g. outputs/<model>_ja -> outputs/<model>_ja_aug
    output_dir: str | None = None
    # servers and databases of the replays that check rewritten server code;
    # defaults to <output_dir>/aug_env_runs
    work_dir: str | None = None
    # comma-separated recipes, each part names joined by "+" (e.g.
    # "tool_rename+task_ambiguous,miss_param"); defaults to every part on its own
    parts: list[str] | None = None
    # augmented tasks per scenario and recipe
    num_per_part: int = 2
    seed: int = 0
    model: str = "your-llm-model-name"
    allowed_scenarios: list[str] | None = None
    # replies generated per LLM rewrite before it is dropped, feedback included
    max_generations: int = 3
    # concurrent LLM requests, and concurrent replays of rewritten server code
    max_concurrency: int = 32

    def input_paths(self) -> list[str]:
        """Inputs that are written again, with augmented data, to output_dir."""
        assert self.scenario_path is not None
        return [
            self.input_task,
            self.input_verifier,
            self.envs_path,
            self.db_path,
            self.sample_path,
            self.scenario_path,
        ]

    def pre_process(self):
        if self.scenario_path is None:
            self.scenario_path = os.path.join(
                os.path.dirname(self.envs_path), "gen_scenario.jsonl"
            )
        for path in [
            self.scenario_path,
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
            self.parts = list(PARTS)
        for recipe in self.parts:
            parse_recipe(recipe)
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
        for task_idx, task in enumerate(item["tasks"]):
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


def drafts_by_scenario(drafts: list[Draft]) -> dict[str, list[Draft]]:
    grouped: dict[str, list[Draft]] = {}
    for draft in drafts:
        grouped.setdefault(normalize_scenario_name(draft.scenario), []).append(draft)
    return grouped


def llm_runner(complete: Complete, max_generations: int) -> RunLLM:
    """Runs LLM steps, sending problems back until a reply is usable."""

    def run_llm[T](steps: Sequence[LLMStep[T]], workers: int = 1) -> list[T]:
        return refine(steps, complete, max_generations, workers=workers)

    return run_llm


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
    assert config.scenario_path is not None
    inputs = {
        "tasks": config.input_task,
        "verifiers": config.input_verifier,
        "envs": config.envs_path,
        "db": config.db_path,
        "sample": config.sample_path,
        "scenarios": config.scenario_path,
    }
    outputs = {
        kind: output_path(path, config.output_dir) for kind, path in inputs.items()
    }
    # only original rows are augmented; augmented data already in the inputs is skipped
    rows = {
        kind: [r for r in tools_jsonl_load(path) if not is_augmented_env_row(r)]
        for kind, path in inputs.items()
    }
    rows["tasks"] = original_task_rows(rows["tasks"])
    base_verifiers = [r for r in rows.pop("verifiers") if not r.get("augmented")]

    drafts = load_drafts(config, rows["tasks"])
    solved = sum(1 for d in drafts if d.reference_calls)
    logger.info(
        f"{len(drafts)} source tasks with a verifier, {solved} solved by the reference run"
    )

    ctx = PartContext(
        seed=config.seed,
        num_per_part=config.num_per_part,
        run_llm=llm_runner(llm_complete(config), config.max_generations),
        replay=ReplayContext(config.db_path, config.sample_path, config.work_dir),
        env_code={
            normalize_scenario_name(r["scenario"]): r["full_code"]
            for r in rows["envs"]
            if r.get("full_code")
        },
        workers=config.max_concurrency,
    )
    by_scenario = drafts_by_scenario(drafts)
    env_variants = [
        env
        for text in config.parts
        for env in run_recipe(parse_recipe(text), by_scenario, ctx)
    ]

    # scenarios outside the selection keep the environments an earlier run made for them
    previous = {
        kind: [r for r in load_previous(path) if is_augmented_env_row(r)]
        if config.allowed_scenarios
        else []
        for kind, path in outputs.items()
    }
    keep_sources = {
        source
        for r in previous["tasks"]
        if (source := normalize_scenario_name(r["source_scenario"]))
        not in (config.allowed_scenarios or [])
    }
    new_rows, env_verifiers = merge_env_variants(
        {kind: rs + previous[kind] for kind, rs in rows.items()},
        previous["verifiers"],
        env_variants,
        keep_sources,
    )

    new_rows["verifiers"] = base_verifiers + env_verifiers
    for kind, path in outputs.items():
        tools_jsonl_save(new_rows[kind], path)
        logger.info(f"Saved {len(new_rows[kind])} rows to {path}")

    counts: dict[str, int] = {}
    for env in env_variants:
        counts[env.augmentation] = counts.get(env.augmentation, 0) + len(env.variants)
    logger.info(
        f"Added {sum(counts.values())} augmented tasks in {len(env_variants)} environments: {counts}"
    )
