"""Apply recipes (sequences of parts) to source tasks and merge the results into the dataset."""

import random
from collections.abc import Callable

from loguru import logger

from awm.core.augment.draft import Draft, LLMStep, Part, Variant, finalize
from awm.core.augment.followup_parts import MissFunc, MissParam
from awm.core.augment.task_parts import (
    TaskAmbiguous,
    TaskDistractorNames,
    TaskNoToolNames,
    TaskParaphrase,
)
from awm.core.augment.tool_parts import (
    AddParams,
    DistractorTools,
    DropOptionalParams,
    ParamRename,
    Shuffle,
    ToolDocParaphrase,
    ToolRename,
)
from awm.core.tool_view import TaskMeta
from awm.tools import normalize_scenario_name

PARTS: dict[str, Part] = {
    part.name: part
    for part in [
        ToolRename(),
        ParamRename(),
        Shuffle(),
        AddParams(),
        DropOptionalParams(),
        DistractorTools(),
        ToolDocParaphrase(),
        TaskParaphrase(),
        TaskAmbiguous(),
        TaskNoToolNames(),
        TaskDistractorNames(),
        MissFunc(),
        MissParam(),
    ]
}

RECIPE_SEPARATOR = "+"

# runs a batch of LLM steps and returns the drafts whose replies were usable
LLMRunner = Callable[[list[LLMStep]], list[Draft]]


def parse_recipe(recipe: str) -> list[Part]:
    """Parts of a recipe such as "tool_rename+task_paraphrase+miss_param"."""
    names = [n.strip() for n in recipe.split(RECIPE_SEPARATOR) if n.strip()]
    if not names:
        raise ValueError("empty recipe")
    unknown = [n for n in names if n not in PARTS]
    if unknown:
        raise ValueError(
            f"unknown parts {unknown} in recipe '{recipe}', choose from {sorted(PARTS)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"recipe '{recipe}' uses a part more than once")
    return [PARTS[n] for n in names]


def recipe_applicable(parts: list[Part], draft: Draft) -> bool:
    return all(part.applicable(draft) for part in parts)


def part_rng(seed: int, recipe: str, draft: Draft, part: Part) -> random.Random:
    key = f"{seed}:{recipe}:{normalize_scenario_name(draft.scenario)}:{draft.source_task_idx}:{part.name}"
    return random.Random(key)


def apply_recipe(
    recipe: str, drafts: list[Draft], seed: int, run_llm: LLMRunner
) -> list[Draft]:
    """Apply the recipe's parts in order; drafts a part cannot handle are dropped.

    LLM steps of the same part are sent as one batch.
    """
    for part in parse_recipe(recipe):
        produced: list[Draft] = []
        steps: list[LLMStep] = []
        for draft in drafts:
            if not part.applicable(draft):
                continue
            out = part.apply(draft, part_rng(seed, recipe, draft, part))
            if isinstance(out, LLMStep):
                steps.append(out)
            elif out is not None:
                produced.append(out)
        if steps:
            produced.extend(run_llm(steps))
        drafts = produced
    return drafts


def generate_variants(
    recipes: list[str],
    drafts: list[Draft],
    num_per_recipe: int,
    seed: int,
    run_llm: LLMRunner,
    oversample: int = 2,
) -> list[Variant]:
    """Up to num_per_recipe variants per scenario and recipe.

    oversample times as many source tasks are tried, since a part can reject a
    task (nothing to rename, a rewrite that failed validation, ...).
    """
    by_scenario: dict[str, list[Draft]] = {}
    for draft in drafts:
        by_scenario.setdefault(normalize_scenario_name(draft.scenario), []).append(
            draft
        )

    variants: list[Variant] = []
    for recipe in recipes:
        parts = parse_recipe(recipe)
        tried: list[Draft] = []
        for scenario, scenario_drafts in by_scenario.items():
            candidates = [d for d in scenario_drafts if recipe_applicable(parts, d)]
            random.Random(f"{seed}:{recipe}:{scenario}").shuffle(candidates)
            tried.extend(candidates[: num_per_recipe * oversample])

        done = apply_recipe(recipe, tried, seed, run_llm)
        done.sort(
            key=lambda d: (normalize_scenario_name(d.scenario), d.source_task_idx)
        )
        counts: dict[str, int] = {}
        for draft in done:
            scenario = normalize_scenario_name(draft.scenario)
            if counts.get(scenario, 0) < num_per_recipe:
                counts[scenario] = counts.get(scenario, 0) + 1
                variants.append(finalize(draft))
        logger.info(
            f"Recipe {recipe}: {sum(counts.values())} tasks from {len(tried)} tried"
        )
    return variants


# ------------------------------------------------------ dataset assembly


def strip_augmented(item: dict) -> list[str]:
    """Original tasks of a gen_tasks.jsonl row, without previously added ones."""
    metas = item.get("task_meta") or []
    for idx, meta in enumerate(metas):
        if meta is not None:
            return item["tasks"][:idx]
    return list(item["tasks"])


def merge_variants(
    task_items: list[dict], verifier_rows: list[dict], variants: list[Variant]
) -> tuple[list[dict], list[dict]]:
    """New gen_tasks rows and verifier rows with the variants appended."""
    by_scenario: dict[str, list[Variant]] = {}
    for variant in variants:
        by_scenario.setdefault(normalize_scenario_name(variant.scenario), []).append(
            variant
        )

    new_items = []
    new_verifiers = [row for row in verifier_rows if not row.get("augmented")]
    for item in task_items:
        originals = strip_augmented(item)
        added = by_scenario.get(normalize_scenario_name(item["scenario"]), [])
        new_item = {k: v for k, v in item.items() if k not in ("tasks", "task_meta")}
        new_item["tasks"] = originals + [v.task for v in added]
        if added:
            new_item["task_meta"] = [None] * len(originals) + [
                v.meta.model_dump() for v in added
            ]
        new_items.append(new_item)
        for offset, variant in enumerate(added):
            new_verifiers.append(
                {
                    "scenario": item["scenario"],
                    "task_idx": len(originals) + offset,
                    "task": variant.task,
                    "verification": {"code": variant.verifier_code},
                    "augmented": True,
                    "augmentations": variant.meta.augmentations,
                    "source_task_idx": variant.meta.source_task_idx,
                }
            )
    return new_items, new_verifiers


def existing_variants(
    task_items: list[dict], verifier_rows: list[dict]
) -> list[Variant]:
    """Augmented tasks already in the dataset, as variants to merge again."""
    codes = {
        (normalize_scenario_name(r["scenario"]), r["task_idx"]): r["verification"][
            "code"
        ]
        for r in verifier_rows
        if r.get("augmented")
    }
    variants = []
    for item in task_items:
        scenario = normalize_scenario_name(item["scenario"])
        for idx, meta in enumerate(item.get("task_meta") or []):
            if meta is None or (scenario, idx) not in codes:
                continue
            variants.append(
                Variant(
                    scenario=item["scenario"],
                    task=item["tasks"][idx],
                    meta=TaskMeta.model_validate(meta),
                    verifier_code=codes[(scenario, idx)],
                )
            )
    return variants
