"""Add augmented tasks to gen_tasks.jsonl and their verifiers to the verifier file.

Augmented tasks live next to the original ones so that RL reads a single
dataset: they are appended to each scenario's `tasks`, with their settings in a
parallel `task_meta` list (None for original tasks), and their verifiers are
appended to the code-mode verifier file under the same (scenario, task_idx).

Each augmentation is an independent part (draft.Part). A recipe chains parts
with "+", e.g. "tool_rename+task_paraphrase+miss_param"; a single part name is
a recipe of its own.
"""

import os
import shutil
from dataclasses import dataclass

from loguru import logger

from awm.core.augment.compose import (
    PARTS,
    existing_variants,
    generate_variants,
    merge_variants,
    parse_recipe,
    strip_augmented,
)
from awm.core.augment.draft import Draft, LLMStep
from awm.gpt import GPTClient
from awm.tools import (
    normalize_scenario_name,
    tools_jsonl_load,
    tools_jsonl_save,
    tools_robust_json_loads,
)


@dataclass
class Config:
    # gen_tasks.jsonl, updated in place (first run backs it up to .bak)
    input_task: str
    # code-mode verifier file, updated in place (first run backs it up to .bak)
    input_verifier: str
    # gen_reference.jsonl from `awm gen reference`
    input_reference: str
    # gen_tools.jsonl from `awm gen reference`
    input_tools: str
    # comma-separated recipes, parts chained with "+"; defaults to every part alone
    recipes: list[str] | None = None
    # augmented tasks per scenario and recipe
    num_per_recipe: int = 2
    seed: int = 0
    model: str = "your-llm-model-name"
    allowed_scenarios: list[str] | None = None
    max_retry: int = 2
    max_concurrency: int = 32

    def pre_process(self):
        for path in [
            self.input_task,
            self.input_verifier,
            self.input_reference,
            self.input_tools,
        ]:
            assert os.path.exists(path), f"{path} not found"
        if self.recipes is None:
            self.recipes = list(PARTS)
        for recipe in self.recipes:
            parse_recipe(recipe)
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
        (normalize_scenario_name(r["scenario"]), r["task_idx"]): r["tool_calls"]
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
            drafts.append(
                Draft(
                    scenario=item["scenario"],
                    source_task_idx=task_idx,
                    task=task,
                    source_verifier_code=code,
                    tools=tools.get(scenario),
                    reference_calls=references.get((scenario, task_idx)),
                    other_tools=other_tools,
                )
            )
    return drafts


def llm_runner(config: Config):
    """Send LLM steps in batches, retrying the ones whose reply was unusable."""
    client: GPTClient | None = None

    def run_llm(steps: list[LLMStep]) -> list[Draft]:
        nonlocal client
        if client is None:
            assert config.model != "your-llm-model-name", (
                "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"
            )
            client = GPTClient(timeout=1200, concurrency_limit=config.max_concurrency)
        drafts: list[Draft] = []
        pending = steps
        for _ in range(config.max_retry + 1):
            if not pending:
                break
            responses = client.batch_chat_completion(
                [
                    {
                        "messages": step.messages,
                        "model": config.model,
                        "temperature": 1.0,
                        "max_tokens": 8192,
                    }
                    for step in pending
                ]
            )
            failed = []
            for step, response in zip(pending, responses):
                parsed = tools_robust_json_loads(response) if response else None
                draft = step.finish(parsed) if isinstance(parsed, dict) else None
                if draft is None:
                    failed.append(step)
                else:
                    drafts.append(draft)
            pending = failed
        if pending:
            logger.warning(
                f"Gave up on {len(pending)} LLM rewrites that failed validation"
            )
        return drafts

    return run_llm


def _backup_once(path: str) -> None:
    backup = f"{path}.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        logger.info(f"Backed up {path} to {backup}")


def run(config: Config):
    assert config.recipes is not None
    task_items = tools_jsonl_load(config.input_task)
    verifier_rows = tools_jsonl_load(config.input_verifier)

    drafts = load_drafts(config, task_items)
    solved = sum(1 for d in drafts if d.reference_calls)
    logger.info(
        f"{len(drafts)} source tasks with a verifier, {solved} solved by the reference run"
    )

    variants = generate_variants(
        config.recipes, drafts, config.num_per_recipe, config.seed, llm_runner(config)
    )
    if config.allowed_scenarios:
        # scenarios outside the selection keep the augmented tasks they already have
        others = [
            item
            for item in task_items
            if normalize_scenario_name(item["scenario"]) not in config.allowed_scenarios
        ]
        variants = existing_variants(others, verifier_rows) + variants

    new_items, new_verifiers = merge_variants(task_items, verifier_rows, variants)

    _backup_once(config.input_task)
    _backup_once(config.input_verifier)
    tools_jsonl_save(new_items, config.input_task)
    tools_jsonl_save(new_verifiers, config.input_verifier)

    counts: dict[str, int] = {}
    for variant in variants:
        key = "+".join(variant.meta.augmentations)
        counts[key] = counts.get(key, 0) + 1
    logger.success(f"Added {len(variants)} augmented tasks: {counts}")
