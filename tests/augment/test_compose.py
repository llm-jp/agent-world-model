import shutil

import pytest
from helpers import CALLS, make_db, make_draft, scripted_llm

from awm.core.augment.compose import (
    apply_recipe,
    generate_variants,
    merge_variants,
    parse_recipe,
    strip_augmented,
)
from awm.core.augment.draft import Variant, finalize, values_in_task
from awm.core.tool_view import PresentedTools, TaskMeta
from awm.core.verify import run_verifier


def replies(step) -> list[dict]:
    """Removing one value (what miss_param asks for), then a faithful rewrite.

    The faithful rewrite also mentions every presented tool name, which is what
    task_distractor_names asks for.
    """
    task = step.draft.task
    removals = [
        {"task": task.replace(v, "it"), "followup": f"It is {v}."}
        for v in values_in_task(task, CALLS)
    ]
    names = ", ".join(step.draft.presented_names)
    return [*removals, {"task": f"Please: {task} (not needed: {names})"}]


LLM = scripted_llm(replies)


def test_parse_recipe_rejects_unknown_and_repeated_parts():
    assert [p.name for p in parse_recipe("tool_rename + miss_func")] == [
        "tool_rename",
        "miss_func",
    ]
    for bad in ["", "tool_rename+nope", "shuffle+shuffle"]:
        with pytest.raises(ValueError):
            parse_recipe(bad)


def test_combined_recipe_keeps_every_part(tmp_path):
    recipe = "tool_rename+param_rename+task_paraphrase+miss_func+miss_param"
    [draft] = apply_recipe(recipe, [make_draft()], seed=0, run_llm=LLM)
    assert draft.applied == recipe.split("+")
    variant = finalize(draft)
    meta = variant.meta

    # first turn: renamed tools, create_order hidden
    assert meta.tool_view is not None
    first = PresentedTools(draft.tools or [], meta.tool_view)
    presented_order = draft.presented_name("create_order")
    assert presented_order not in [t["name"] for t in first.tools]
    # follow-up turn: the value removed by miss_param and the revealed tool
    [turn] = meta.turns
    assert draft.removed_values[0] in turn.user
    assert presented_order in turn.user
    assert turn.tool_view is not None
    second = PresentedTools(draft.tools or [], turn.tool_view)
    assert second.resolve(presented_order, {"productId": 42}) == (
        "create_order",
        {"product_id": 42},
    )
    assert draft.removed_values[0] not in variant.task

    # reward: waited in the first turn, then completed
    initial = make_db(tmp_path / "initial.db", [])
    turn0 = tmp_path / "turn_0.db"
    shutil.copy2(initial, turn0)
    final = make_db(tmp_path / "final.db", [42])
    calls = [{"turn": 1, "tool_name": "create_order", "arguments": {"product_id": 42}}]
    entry = {"verification": {"code": variant.verifier_code}}
    reward, _ = run_verifier(entry, "code", initial, final, "", [str(turn0)], calls)
    assert reward == "complete"


def test_order_of_parts_changes_what_later_parts_see():
    # distractor tools added first become candidates for distractor names
    [draft] = apply_recipe(
        "distractor_tools+task_distractor_names", [make_draft()], seed=0, run_llm=LLM
    )
    assert set(draft.required_mentions) <= set(draft.presented_names)


def test_rewrite_after_miss_param_must_not_bring_the_value_back():
    def bring_back(step):
        removed = step.draft.removed_values
        return [{"task": step.draft.task + " " + " ".join(removed)}]

    [after_miss] = apply_recipe("miss_param", [make_draft()], seed=0, run_llm=LLM)
    out = apply_recipe("task_paraphrase", [after_miss], 0, scripted_llm(bring_back))
    assert out == []


def test_recipe_drops_drafts_a_part_cannot_handle():
    assert apply_recipe("miss_func", [make_draft(reference_calls=None)], 0, LLM) == []


def test_generate_variants_limits_tasks_per_scenario_and_recipe():
    drafts = [make_draft(source_task_idx=i) for i in range(5)]
    drafts.append(make_draft(scenario="Bank", source_task_idx=0))
    variants = generate_variants(["shuffle", "add_params+miss_func"], drafts, 2, 0, LLM)
    per_key: dict[tuple[str, str], int] = {}
    for v in variants:
        key = (v.scenario, "+".join(v.meta.augmentations))
        per_key[key] = per_key.get(key, 0) + 1
    assert per_key == {
        ("Shop", "shuffle"): 2,
        ("Bank", "shuffle"): 1,
        ("Shop", "add_params+miss_func"): 2,
        ("Bank", "add_params+miss_func"): 1,
    }


# ------------------------------------------------- dataset assembly


def variant(task: str) -> Variant:
    return Variant(
        scenario="Shop",
        task=task,
        meta=TaskMeta(augmentations=["shuffle"], source_task_idx=0),
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
    assert added["augmentations"] == ["shuffle"]


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
