"""Apply augmentation parts to source tasks and merge the results into the dataset."""

import logging
import random
from collections.abc import Callable

from pydantic import ValidationError

from awm.core.augment.draft import Draft, LLMStep, Part, Variant, finalize
from awm.core.augment.env_parts import ENV_PARTS
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
    Shuffle,
    ToolDocParaphrase,
)
from awm.core.tool_view import TaskMeta
from awm.tools import normalize_scenario_name, strip_augmented

logger = logging.getLogger(__name__)

PARTS: dict[str, Part] = {
    part.name: part
    for part in [
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

# runs a batch of LLM steps and returns the drafts whose replies were usable
LLMRunner = Callable[[list[LLMStep[Draft]]], list[Draft]]


def check_part_names(names: list[str]) -> None:
    known = set(PARTS) | set(ENV_PARTS)
    if unknown := [n for n in names if n not in known]:
        raise ValueError(f"unknown parts {unknown}, choose from {sorted(known)}")


def part_rng(seed: int, draft: Draft, part: Part) -> random.Random:
    key = f"{seed}:{part.name}:{normalize_scenario_name(draft.scenario)}:{draft.source_task_idx}"
    return random.Random(key)


def apply_part(
    part: Part, drafts: list[Draft], seed: int, run_llm: LLMRunner
) -> list[Draft]:
    """The part applied to each source draft; drafts it cannot handle are dropped.

    LLM steps are sent as one batch.
    """
    produced: list[Draft] = []
    steps: list[LLMStep[Draft]] = []
    for draft in drafts:
        if not part.applicable(draft):
            continue
        out = part.apply(draft, part_rng(seed, draft, part))
        if isinstance(out, LLMStep):
            steps.append(out)
        elif out is not None:
            produced.append(out)
    if steps:
        produced.extend(run_llm(steps))
    return produced


def generate_variants(
    part_names: list[str],
    drafts: list[Draft],
    num_per_part: int,
    seed: int,
    run_llm: LLMRunner,
    oversample: int = 2,
) -> list[Variant]:
    """Up to num_per_part variants per scenario and part.

    oversample times as many source tasks are tried, since a part can reject a
    task (nothing to rename, a rewrite that failed validation, ...).
    """
    by_scenario: dict[str, list[Draft]] = {}
    for draft in drafts:
        by_scenario.setdefault(normalize_scenario_name(draft.scenario), []).append(
            draft
        )

    variants: list[Variant] = []
    for name in part_names:
        part = PARTS[name]
        tried: list[Draft] = []
        for scenario, scenario_drafts in by_scenario.items():
            candidates = [d for d in scenario_drafts if part.applicable(d)]
            random.Random(f"{seed}:{name}:{scenario}").shuffle(candidates)
            tried.extend(candidates[: num_per_part * oversample])

        done = apply_part(part, tried, seed, run_llm)
        done.sort(
            key=lambda d: (normalize_scenario_name(d.scenario), d.source_task_idx)
        )
        counts: dict[str, int] = {}
        for draft in done:
            scenario = normalize_scenario_name(draft.scenario)
            if counts.get(scenario, 0) < num_per_part:
                counts[scenario] = counts.get(scenario, 0) + 1
                variants.append(finalize(draft))
        logger.info(
            f"Part {name}: {sum(counts.values())} tasks from {len(tried)} tried"
        )
    return variants


# ------------------------------------------------------ dataset assembly


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
            try:
                task_meta = TaskMeta.model_validate(meta)
            except ValidationError as e:
                logger.warning(
                    f"Dropping {item['scenario']} task {idx}: its task_meta is from an older format ({e.error_count()} errors)"
                )
                continue
            variants.append(
                Variant(
                    scenario=item["scenario"],
                    task=item["tasks"][idx],
                    meta=task_meta,
                    verifier_code=codes[(scenario, idx)],
                )
            )
    return variants
