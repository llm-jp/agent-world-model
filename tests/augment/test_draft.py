import shutil

from helpers import CALLS, SOURCE_VERIFIER, TASK, make_db, make_draft

from awm.core.augment.draft import (
    Rejection,
    build_followup_verifier,
    finalize,
    keep_values,
    mentioned_tool_names,
    param_values_in_task,
    rewritten_task,
    values_in_task,
)
from awm.core.tool_view import ToolView
from awm.core.verify import run_verifier


def grade(code, initial, final, turn_dbs, calls):
    entry = {"verification": {"code": code}}
    reward, _ = run_verifier(entry, "code", initial, final, "", turn_dbs, calls)
    return reward


# ------------------------------------------------------------- helpers


def test_values_in_task_ignores_short_bool_and_absent_values():
    calls = [
        {
            "tool_name": "t",
            "arguments": {"a": "blue mug", "b": 4, "c": "red", "d": True},
        }
    ]
    assert values_in_task("buy blue mug x4", calls) == ["blue mug"]
    assert values_in_task("", calls) == []


def test_param_values_in_task():
    assert param_values_in_task(TASK, CALLS) == [
        ("search_products", "query", "blue mug"),
        ("create_order", "product_id", "42"),
        ("create_order", "note", "gift"),
    ]


def test_mentioned_tool_names_matches_spaced_form():
    names = ["create_order", "x_y"]
    assert mentioned_tool_names("please Create Order now", names) == ["create_order"]
    assert mentioned_tool_names("nothing here", names) == []


def test_keep_values_excludes_the_removed_value():
    draft = make_draft(removed_value="gift")
    assert keep_values(draft) == ["blue mug", "42"]


def test_rewritten_task_enforces_the_draft_constraints():
    draft = make_draft(
        removed_value="gift",
        required_mentions=["list_coupons"],
        banned_mentions=["create_order"],
    )
    ok = "Find 'blue mug', ignore list_coupons, buy product 42"
    assert rewritten_task(draft, {"task": ok}) == ok
    for bad in [
        ok + " as a gift",
        "Find 'blue mug', buy product 42",
        ok + " via create order",
        draft.task,
        3,
    ]:
        assert isinstance(rewritten_task(draft, {"task": bad}), Rejection)


def test_rejection_names_the_missing_values():
    rejection = rewritten_task(make_draft(), {"task": "Order a mug"})
    assert isinstance(rejection, Rejection)
    assert "blue mug" in rejection.reason and "42" in rejection.reason


# ------------------------------------------------- follow-up verifier


def snapshot_of(initial, tmp_path) -> str:
    turn0 = tmp_path / "turn_0.db"
    shutil.copy2(initial, turn0)
    return str(turn0)


def test_followup_verifier_rewards_waiting_then_completing(tmp_path):
    code = build_followup_verifier(
        SOURCE_VERIFIER, [{"tool_name": "create_order", "param": None}]
    )
    initial = make_db(tmp_path / "initial.db", [])
    final = make_db(tmp_path / "final.db", [42])
    calls = [
        {"turn": 0, "tool_name": "search_products", "arguments": {"query": "mug"}},
        {"turn": 1, "tool_name": "create_order", "arguments": {"product_id": 42}},
    ]
    turn0 = snapshot_of(initial, tmp_path)
    assert grade(code, initial, final, [turn0], calls) == "complete"


def test_followup_verifier_rejects_db_change_in_first_turn(tmp_path):
    code = build_followup_verifier(SOURCE_VERIFIER, [])
    initial = make_db(tmp_path / "initial.db", [])
    turn0 = make_db(tmp_path / "turn_0.db", [7])
    final = make_db(tmp_path / "final.db", [42])
    assert grade(code, initial, final, [turn0], []) == "others"


def test_followup_verifier_rejects_forbidden_call_with_param(tmp_path):
    code = build_followup_verifier(
        SOURCE_VERIFIER, [{"tool_name": "search_products", "param": "query"}]
    )
    initial = make_db(tmp_path / "initial.db", [])
    final = make_db(tmp_path / "final.db", [42])
    turn0 = snapshot_of(initial, tmp_path)
    without = [{"turn": 0, "tool_name": "search_products", "arguments": {"limit": 3}}]
    with_param = [
        {"turn": 0, "tool_name": "search_products", "arguments": {"query": "x"}}
    ]
    assert grade(code, initial, final, [turn0], without) == "complete"
    assert grade(code, initial, final, [turn0], with_param) == "others"


def test_followup_verifier_without_turn_snapshot_is_others(tmp_path):
    code = build_followup_verifier(SOURCE_VERIFIER, [])
    initial = make_db(tmp_path / "initial.db", [])
    final = make_db(tmp_path / "final.db", [42])
    assert grade(code, initial, final, [], []) == "others"


def test_followup_verifier_fails_when_original_task_not_done(tmp_path):
    code = build_followup_verifier(SOURCE_VERIFIER, [])
    initial = make_db(tmp_path / "initial.db", [])
    final = make_db(tmp_path / "final.db", [1])
    turn0 = snapshot_of(initial, tmp_path)
    assert grade(code, initial, final, [turn0], []) == "others"


# ------------------------------------------------------------- finalize


def test_finalize_without_changes_keeps_the_original_verifier():
    variant = finalize(make_draft(part="task_paraphrase"))
    assert variant.meta.tool_view is None
    assert variant.meta.turns == []
    assert variant.meta.augmentations == ["task_paraphrase"]
    assert variant.verifier_code == SOURCE_VERIFIER


def test_finalize_hides_the_tool_in_the_first_turn_only():
    draft = make_draft(
        part="miss_func",
        hidden_tool="create_order",
        followup="create_order is back.",
        forbidden_calls=[{"tool_name": "create_order", "param": None}],
    )
    variant = finalize(draft)
    assert variant.meta.tool_view == ToolView(hide_tools=["create_order"])
    [turn] = variant.meta.turns
    assert turn.user == "create_order is back."
    assert turn.tool_view == ToolView()
    assert "unchanged_db_before_followup" in variant.verifier_code


def test_finalize_followup_without_hidden_tool_keeps_the_view():
    view = ToolView(tool_descriptions={"create_order": "Place an order."})
    meta = finalize(make_draft(view=view, followup="It is 42.")).meta
    assert meta.tool_view == view
    assert meta.turns[0].tool_view is None
