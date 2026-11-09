import random

from helpers import CALLS, TASK, make_draft

from awm.core.augment.draft import Draft, LLMStep, Rejection
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

RNG = random.Random(0)


def step_of(out) -> LLMStep:
    assert isinstance(out, LLMStep)
    return out


def draft_of(out) -> Draft:
    assert isinstance(out, Draft), out
    return out


# ------------------------------------------------------------ tool parts


def test_shuffle_needs_two_tools():
    assert Shuffle().apply(make_draft(), RNG) is not None
    assert Shuffle().apply(make_draft(tools=[{"name": "a"}]), RNG) is None


def test_add_params_avoids_names_already_taken():
    tool = {
        "name": "t",
        "inputSchema": {"properties": {"request_id": {}, "trace_id": {}}},
    }
    out = draft_of(AddParams().apply(make_draft(tools=[tool]), random.Random(1)))
    assert out.view.extra_params == {
        "t": {"client_version": out.view.extra_params["t"]["client_version"]}
    }


def test_drop_optional_params_keeps_used_and_required():
    out = DropOptionalParams().apply(make_draft(), RNG)
    assert out is not None
    # limit is optional and unused; note is optional but used
    assert out.view.drop_params == {"search_products": ["limit"]}


def test_drop_optional_params_needs_reference():
    assert not DropOptionalParams().applicable(make_draft(reference_calls=None))


def test_distractor_tools_adds_tools_from_other_scenarios():
    out = DistractorTools().apply(make_draft(), RNG)
    assert out is not None
    assert [t["name"] for t in out.view.distractor_tools] == ["send_fax"]
    assert DistractorTools().apply(make_draft(other_tools=[]), RNG) is None


def test_tool_doc_paraphrase_requires_every_tool_and_param():
    step = step_of(ToolDocParaphrase().apply(make_draft(), RNG))
    docs = {
        "search_products": {
            "description": "Find.",
            "parameters": {"query": "q", "limit": "l"},
        },
        "create_order": {
            "description": "Buy.",
            "parameters": {"product_id": "p", "note": "n"},
        },
        "list_coupons": {"description": "Coupons.", "parameters": {}},
    }
    out = draft_of(step.finish({"tools": docs}))
    assert out.view.tool_descriptions["create_order"] == "Buy."
    assert "list_coupons" not in out.view.param_descriptions
    del docs["list_coupons"]
    assert isinstance(step.finish({"tools": docs}), Rejection)


# ------------------------------------------------------------ task parts


def test_paraphrase_rejects_lost_values():
    step = step_of(TaskParaphrase().apply(make_draft(), RNG))
    assert step.check_prompt is not None
    assert isinstance(step.finish({"task": "Order a mug"}), Rejection)
    out = draft_of(
        step.finish({"task": "Order product 42 ('blue mug') with note 'gift'"})
    )
    assert out.part == "task_paraphrase"


def test_ambiguous_uses_its_own_prompt():
    paraphrase = step_of(TaskParaphrase().apply(make_draft(), RNG))
    ambiguous = step_of(TaskAmbiguous().apply(make_draft(), RNG))
    assert paraphrase.messages[1]["content"] != ambiguous.messages[1]["content"]
    out = draft_of(ambiguous.finish({"task": "blue mug, 42, gift - you know"}))
    assert out.part == "task_ambiguous"


def test_no_tool_names_needs_a_mentioned_name_and_bans_it():
    assert TaskNoToolNames().apply(make_draft(), RNG) is None
    draft = make_draft(task="Use create_order to order product 42")
    step = step_of(TaskNoToolNames().apply(draft, RNG))
    assert isinstance(
        step.finish({"task": "Use create order for product 42"}), Rejection
    )
    out = draft_of(step.finish({"task": "Order product 42"}))
    assert out.banned_mentions == ["create_order"]


def test_distractor_names_must_stay_mentioned():
    step = step_of(TaskDistractorNames().apply(make_draft(), RNG))
    assert isinstance(step.finish({"task": TASK + "."}), Rejection)
    out = draft_of(step.finish({"task": TASK + " (no need for list_coupons)"}))
    assert out.required_mentions == ["list_coupons"]


def test_distractor_names_without_unused_tools_is_skipped():
    draft = make_draft(
        tools=[t for t in make_draft().tools or [] if t["name"] != "list_coupons"]
    )
    assert TaskDistractorNames().apply(draft, RNG) is None


# -------------------------------------------------------- follow-up parts


def test_miss_func_hides_the_last_called_tool():
    out = draft_of(MissFunc().apply(make_draft(), RNG))
    assert out.hidden_tool == "create_order"
    assert out.followup is not None and "create_order" in out.followup
    assert out.forbidden_calls == [{"tool_name": "create_order", "param": None}]
    assert MissFunc().apply(make_draft(reference_calls=[]), RNG) is None


def test_miss_param_removes_one_value_and_adds_a_followup():
    step = step_of(MissParam().apply(make_draft(), random.Random(0)))
    assert isinstance(step.finish({"task": TASK, "followup": "x"}), Rejection)
    accepted = []
    for _, _, value in [
        (c["tool_name"], k, str(v)) for c in CALLS for k, v in c["arguments"].items()
    ]:
        out = step.finish(
            {"task": TASK.replace(value, "it"), "followup": f"It is {value}."}
        )
        if isinstance(out, Draft):
            accepted.append((value, out))
    # exactly one candidate value is the one the part removes
    assert len(accepted) == 1
    value, out = accepted[0]
    assert out.removed_value == value
    assert out.followup == f"It is {value}."
    assert out.forbidden_calls[0]["param"] in {"query", "product_id", "note"}
