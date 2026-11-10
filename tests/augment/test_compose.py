import shutil

import pytest
from helpers import CALLS, TASK, make_db, make_draft, scripted_llm

from awm.core.augment.compose import (
    PARTS,
    apply_part,
    check_part_names,
    generate_variants,
    merge_variants,
)
from awm.core.augment.draft import Variant, finalize, values_in_task
from awm.core.tool_view import PresentedTools, TaskMeta
from awm.core.verify import run_verifier
from awm.tools import strip_augmented


def removals(step) -> list[dict]:
    """Replies that each remove one value, which is what miss_param asks for."""
    return [
        {"task": TASK.replace(v, "it"), "followup": f"It is {v}."}
        for v in values_in_task(TASK, CALLS)
    ]


LLM = scripted_llm(removals)


def grade(variant: Variant, tmp_path, calls: list[dict]) -> str:
    """Reward of a run that kept the first turn unchanged, then ordered 42."""
    run_dir = tmp_path / f"run_{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    initial = make_db(run_dir / "initial.db", [])
    turn0 = run_dir / "turn_0.db"
    shutil.copy2(initial, turn0)
    final = make_db(run_dir / "final.db", [42])
    entry = {"verification": {"code": variant.verifier_code}}
    reward, _ = run_verifier(entry, "code", initial, final, "", [str(turn0)], calls)
    return reward


def test_check_part_names_rejects_unknown_parts():
    check_part_names([*PARTS, "tool_rename", "tool_rename_random"])
    with pytest.raises(ValueError):
        check_part_names(["add_params", "tool_rename+miss_func"])


def test_miss_func_variant_restores_the_tool_after_the_followup(tmp_path):
    [draft] = apply_part(PARTS["miss_func"], [make_draft()], seed=0, run_llm=LLM)
    variant = finalize(draft)
    meta = variant.meta
    assert meta.tool_view is not None
    first = PresentedTools(draft.tools or [], meta.tool_view)
    assert "create_order" not in [t["name"] for t in first.tools]
    [turn] = meta.turns
    assert turn.tool_view is not None
    second = PresentedTools(draft.tools or [], turn.tool_view)
    assert "create_order" in [t["name"] for t in second.tools]

    later = [{"turn": 1, "tool_name": "create_order", "arguments": {}}]
    early = [{"turn": 0, "tool_name": "create_order", "arguments": {}}]
    assert grade(variant, tmp_path, later) == "complete"
    assert grade(variant, tmp_path, early) == "others"


def test_miss_param_variant_gives_the_value_in_the_followup(tmp_path):
    [draft] = apply_part(PARTS["miss_param"], [make_draft()], seed=0, run_llm=LLM)
    variant = finalize(draft)
    assert draft.removed_value is not None
    assert draft.removed_value not in variant.task
    [turn] = variant.meta.turns
    assert draft.removed_value in turn.user
    assert grade(variant, tmp_path, []) == "complete"


def test_part_drops_drafts_it_cannot_handle():
    assert (
        apply_part(PARTS["miss_func"], [make_draft(reference_calls=None)], 0, LLM) == []
    )


def test_generate_variants_limits_tasks_per_scenario_and_part():
    drafts = [make_draft(source_task_idx=i) for i in range(5)]
    drafts.append(make_draft(scenario="Bank", source_task_idx=0))
    variants = generate_variants(["add_params", "miss_func"], drafts, 2, 0, LLM)
    per_key: dict[tuple[str, str], int] = {}
    for v in variants:
        key = (v.scenario, v.meta.augmentations[0])
        per_key[key] = per_key.get(key, 0) + 1
    assert per_key == {
        ("Shop", "add_params"): 2,
        ("Bank", "add_params"): 1,
        ("Shop", "miss_func"): 2,
        ("Bank", "miss_func"): 1,
    }


# ------------------------------------------------- dataset assembly


def variant(task: str) -> Variant:
    return Variant(
        scenario="Shop",
        task=task,
        meta=TaskMeta(augmentations=["add_params"], source_task_idx=0),
        verifier_code="code",
    )


def test_merge_appends_tasks_meta_and_verifiers():
    items = [
        {"scenario": "Shop", "tasks": ["t0", "t1"]},
        {"scenario": "Other", "tasks": ["o"]},
    ]
    verifiers = [{"scenario": "Shop", "task_idx": 0, "verification": {"code": "c"}}]
    new_items, new_verifiers = merge_variants(items, verifiers, [variant("a0")])
    assert new_items[0]["tasks"] == ["t0", "t1", "a0"]
    assert new_items[0]["task_meta"][:2] == [None, None]
    assert "task_meta" not in new_items[1]
    added = new_verifiers[-1]
    assert added["task_idx"] == 2 and added["augmented"]
    assert added["augmentations"] == ["add_params"]


def test_merge_is_idempotent():
    items = [{"scenario": "Shop", "tasks": ["t0"]}]
    once = merge_variants(items, [], [variant("a0")])
    twice = merge_variants(once[0], once[1], [variant("a0")])
    assert once == twice


def test_strip_augmented_without_meta_returns_all_tasks():
    assert strip_augmented({"tasks": ["a", "b"]}) == ["a", "b"]
    assert strip_augmented({"tasks": ["a", "b"], "task_meta": [None, {"x": 1}]}) == [
        "a"
    ]
