import random

from helpers import CALLS, TASK, make_draft

from awm.core.augment.draft import Draft, LLMStep, Rejection
from awm.core.augment.followup_parts import MissFunc, MissParam
from awm.core.augment.task_parts import (
    TaskAmbiguous,
    TaskDistractorNames,
    TaskNoToolNames,
    TaskValueFormat,
)
from awm.core.augment.tool_parts import (
    EXTRA_PARAM_POOL,
    MAX_EXTRA_PARAMS,
    MIN_EXTRA_PARAMS,
    AddParams,
    DistractorTools,
    ParamEnum,
    ParamFormat,
    ToolDocParaphrase,
)
from awm.core.tool_view import ToolView

RNG = random.Random(0)


def step_of(out) -> LLMStep:
    assert isinstance(out, LLMStep)
    return out


def draft_of(out) -> Draft:
    assert isinstance(out, Draft), out
    return out


# ------------------------------------------------------------ tool parts


def test_add_params_adds_several_unused_params_to_every_tool():
    tools = [{"name": "a", "inputSchema": {}}, {"name": "b", "inputSchema": {}}]
    out = draft_of(AddParams().apply_one(make_draft(tools=tools), random.Random(1)))
    assert set(out.view.extra_params) == {"a", "b"}
    for params in out.view.extra_params.values():
        assert MIN_EXTRA_PARAMS <= len(params) <= MAX_EXTRA_PARAMS
        assert set(params) <= set(EXTRA_PARAM_POOL)


def test_add_params_avoids_names_already_taken():
    taken = list(EXTRA_PARAM_POOL)[:-1]
    tool = {
        "name": "t",
        "inputSchema": {"properties": {p: {} for p in taken}},
    }
    out = draft_of(AddParams().apply_one(make_draft(tools=[tool]), random.Random(1)))
    last = list(EXTRA_PARAM_POOL)[-1]
    assert out.view.extra_params == {"t": {last: EXTRA_PARAM_POOL[last]}}


def test_add_params_skips_a_task_whose_tools_take_every_name():
    tool = {
        "name": "t",
        "inputSchema": {"properties": {p: {} for p in EXTRA_PARAM_POOL}},
    }
    assert AddParams().apply_one(make_draft(tools=[tool]), random.Random(1)) is None


def test_param_format_writes_a_passed_parameter_in_another_format():
    out = draft_of(ParamFormat().apply_one(make_draft(), RNG))
    # product_id 42 is the only passed value that a format can write
    assert out.view.param_formats == {"create_order": {"product_id": "int_string"}}


def test_param_format_needs_a_value_every_call_can_take():
    calls = [
        {"tool_name": "create_order", "arguments": {"product_id": 42}},
        {"tool_name": "create_order", "arguments": {"product_id": "x"}},
    ]
    assert ParamFormat().apply_one(make_draft(reference_calls=calls), RNG) is None


def test_param_format_keeps_the_earlier_view_and_skips_enum_parameters():
    earlier = ToolView(param_enums={"create_order": {"product_id": ["42", "7"]}})
    assert ParamFormat().apply_one(make_draft(view=earlier), RNG) is None
    extra = ToolView(extra_params={"create_order": {"trace_id": {}}})
    out = draft_of(
        ParamFormat().apply_one(make_draft(view=extra, parts=("add_params",)), RNG)
    )
    assert out.view.extra_params == extra.extra_params
    assert out.view.param_formats == {"create_order": {"product_id": "int_string"}}
    assert out.parts == ("add_params", "param_format")


def test_param_enum_skips_parameters_shown_in_another_format():
    earlier = ToolView(param_formats={"create_order": {"note": "yes_no"}})
    step = step_of(ParamEnum().apply_one(make_draft(view=earlier), RNG))
    assert '"note"' not in step.messages[0]["content"]


def test_param_enum_lists_string_parameters_with_their_passed_values():
    step = step_of(ParamEnum().apply_one(make_draft(), RNG))
    prompt = step.messages[0]["content"]
    assert '"query"' in prompt and '"blue mug"' in prompt and '"note"' in prompt
    assert '"limit"' not in prompt and '"product_id"' not in prompt
    out = draft_of(step.finish({"enums": {"create_order": {"note": ["gift", "none"]}}}))
    assert out.view.param_enums == {"create_order": {"note": ["gift", "none"]}}


def test_param_enum_rejects_unusable_enums():
    step = step_of(ParamEnum().apply_one(make_draft(), RNG))
    for enums in [
        {},
        {"create_order": {"product_id": ["1", "2"]}},
        {"create_order": {"note": ["gift"]}},
        {"create_order": {"note": ["a", "b"]}},
        {"create_order": {"note": ["gift", "gift"]}},
        {"send_fax": {"to": ["a", "b"]}},
    ]:
        assert isinstance(step.finish({"enums": enums}), Rejection), enums


def test_distractor_tools_adds_tools_from_other_scenarios():
    out = draft_of(DistractorTools().apply_one(make_draft(), RNG))
    assert [t["name"] for t in out.view.distractor_tools] == ["send_fax"]
    assert DistractorTools().apply_one(make_draft(other_tools=[]), RNG) is None


def test_tool_doc_paraphrase_requires_every_tool_and_param():
    step = step_of(ToolDocParaphrase().apply_one(make_draft(), RNG))
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


def test_distractor_tools_skip_tools_without_a_description():
    tools = [{"name": "send_fax", "description": "", "inputSchema": {}}]
    assert DistractorTools().apply_one(make_draft(other_tools=tools), RNG) is None


def test_tool_doc_paraphrase_rewords_the_text_before_the_listed_responses():
    responses = "\n\n### Responses:\n\n**200**: OK"
    tools = [{"name": "t", "description": "Old." + responses, "inputSchema": {}}]
    step = step_of(ToolDocParaphrase().apply_one(make_draft(tools=tools), RNG))
    assert "### Responses" not in step.messages[0]["content"]
    out = draft_of(step.finish({"tools": {"t": {"description": "New."}}}))
    assert out.view.tool_descriptions == {"t": "New." + responses}
    reply = {"tools": {"t": {"description": "New." + responses}}}
    assert isinstance(step.finish(reply), Rejection)


# ------------------------------------------------------------ task parts


def test_ambiguous_rejects_lost_values():
    step = step_of(TaskAmbiguous().apply_one(make_draft(), RNG))
    assert step.check_prompt is not None
    assert isinstance(step.finish({"task": "Order a mug"}), Rejection)
    out = draft_of(step.finish({"task": "blue mug, 42, gift - you know"}))
    assert out.parts == ("task_ambiguous",)


REFORMATTED = "Search for 'blue mug' and order product forty-two with the note 'gift'"


def test_value_format_accepts_a_value_written_another_way():
    step = step_of(TaskValueFormat().apply_one(make_draft(), RNG))
    reply = {
        "task": REFORMATTED,
        "rewrites": [{"value": "42", "written_as": "forty-two"}],
    }
    out = draft_of(step.finish(reply))
    assert out.parts == ("task_value_format",) and out.task == REFORMATTED
    assert out.value_rewrites == {"42": "forty-two"}
    # the self-check is told that the value is written another way
    assert step.check_prompt is not None and '"forty-two"' in step.check_prompt(out)


def test_value_format_rejects_replies_that_lose_or_keep_values():
    step = step_of(TaskValueFormat().apply_one(make_draft(), RNG))
    rewrite = [{"value": "42", "written_as": "forty-two"}]
    for reply in [
        {"task": REFORMATTED, "rewrites": []},
        {"task": REFORMATTED, "rewrites": [{"value": "7", "written_as": "forty-two"}]},
        {
            "task": REFORMATTED.replace("forty-two", "42 (forty-two)"),
            "rewrites": rewrite,
        },
        {"task": REFORMATTED.replace("forty-two", "4 2"), "rewrites": rewrite},
        {"task": REFORMATTED.replace("'gift'", "'present'"), "rewrites": rewrite},
    ]:
        assert isinstance(step.finish(reply), Rejection), reply


def test_value_format_rejects_the_original_task():
    step = step_of(TaskValueFormat().apply_one(make_draft(), RNG))
    reply = {"task": TASK, "rewrites": [{"value": "42", "written_as": "forty-two"}]}
    out = step.finish(reply)
    assert isinstance(out, Rejection) and "identical" in out.reason


def test_value_format_keeps_values_that_contain_another_value():
    calls = [{"tool_name": "t", "arguments": {"project": "PRJ-2026-1", "year": "2026"}}]
    draft = make_draft(task="Report PRJ-2026-1 for 2026", reference_calls=calls)
    prompt = step_of(TaskValueFormat().apply_one(draft, RNG)).messages[1]["content"]
    assert 'rewrite: ["PRJ-2026-1"]' in prompt and 'verbatim: ["2026"]' in prompt


def test_value_format_needs_a_value_in_the_task():
    assert TaskValueFormat().apply_one(make_draft(reference_calls=[]), RNG) is None


def test_no_tool_names_needs_a_mentioned_name_and_bans_it():
    assert TaskNoToolNames().apply_one(make_draft(), RNG) is None
    draft = make_draft(task="Use create_order to order product 42")
    step = step_of(TaskNoToolNames().apply_one(draft, RNG))
    assert isinstance(
        step.finish({"task": "Use create order for product 42"}), Rejection
    )
    out = draft_of(step.finish({"task": "Order product 42"}))
    assert out.banned_mentions == ["create_order"]


def test_no_tool_names_also_drops_a_name_an_earlier_part_required():
    draft = make_draft(
        task="Use create_order to order product 42",
        banned_mentions=["placeOrder"],
        required_mentions=["create_order", "list_coupons"],
    )
    step = step_of(TaskNoToolNames().apply_one(draft, RNG))
    out = draft_of(step.finish({"task": "Order product 42, see list_coupons"}))
    assert out.banned_mentions == ["placeOrder", "create_order"]
    assert out.required_mentions == ["list_coupons"]


def test_distractor_names_must_stay_mentioned():
    step = step_of(TaskDistractorNames().apply_one(make_draft(), RNG))
    assert isinstance(step.finish({"task": TASK + "."}), Rejection)
    out = draft_of(step.finish({"task": TASK + " (no need for list_coupons)"}))
    assert out.required_mentions == ["list_coupons"]


def test_distractor_names_keep_the_texts_an_earlier_part_required():
    step = step_of(
        TaskDistractorNames().apply_one(make_draft(required_mentions=["mug"]), RNG)
    )
    assert '"mug"' in step.messages[1]["content"]
    out = draft_of(step.finish({"task": TASK + " (no need for list_coupons)"}))
    assert out.required_mentions == ["mug", "list_coupons"]


def test_distractor_names_without_unused_tools_is_skipped():
    draft = make_draft(
        tools=[t for t in make_draft().tools or [] if t["name"] != "list_coupons"]
    )
    assert TaskDistractorNames().apply_one(draft, RNG) is None


# -------------------------------------------------------- follow-up parts


def test_miss_func_hides_the_last_called_tool():
    out = draft_of(MissFunc().apply_one(make_draft(), RNG))
    assert out.hidden_tool == "create_order"
    assert out.followup is not None and "create_order" in out.followup
    assert out.forbidden_calls == [{"tool_name": "create_order", "param": None}]
    assert MissFunc().apply_one(make_draft(reference_calls=[]), RNG) is None


def test_miss_param_removes_one_value_and_adds_a_followup():
    step = step_of(MissParam().apply_one(make_draft(), random.Random(0)))
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
