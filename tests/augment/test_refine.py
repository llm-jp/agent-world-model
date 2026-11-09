import json
import random

from helpers import TASK, make_draft

from awm.core.augment.followup_parts import MissParam
from awm.core.augment.refine import MAX_CHECK_RETRY, refine
from awm.core.augment.task_parts import TaskParaphrase, dialogue_check_prompt
from awm.core.augment.tool_parts import ToolDocParaphrase

GOOD = json.dumps({"task": "Order product 42 ('blue mug') with note 'gift'"})
LOST_VALUES = json.dumps({"task": "Order a mug"})
CONSISTENT = json.dumps({"consistent": True, "problems": []})


def inconsistent(problem: str) -> str:
    return json.dumps({"consistent": False, "problems": [problem]})


def is_self_check(conversation: list[dict]) -> bool:
    return "Reference tool calls" in conversation[-1]["content"]


class ScriptedLLM:
    """Replies to generations and self-checks from separate scripts, in order."""

    def __init__(self, generations: list[str | None], checks: list[str]):
        self.generations = list(generations)
        self.checks = list(checks)
        self.conversations: list[list[dict]] = []

    def __call__(self, conversations: list[list[dict]]) -> list[str | None]:
        self.conversations.extend(conversations)
        return [
            self.checks.pop(0) if is_self_check(c) else self.generations.pop(0)
            for c in conversations
        ]

    def generation_requests(self) -> list[list[dict]]:
        return [c for c in self.conversations if not is_self_check(c)]


def paraphrase_step():
    return TaskParaphrase().apply(make_draft(), random.Random(0))


def test_reply_is_accepted_after_a_consistent_self_check():
    llm = ScriptedLLM([GOOD], [CONSISTENT])
    [draft] = refine([paraphrase_step()], llm, max_generations=3)
    assert draft.task == json.loads(GOOD)["task"]
    assert len(llm.conversations) == 2


def test_self_check_problems_are_fed_back_to_the_next_generation():
    llm = ScriptedLLM([GOOD, GOOD], [inconsistent("the note is lost"), CONSISTENT])
    assert len(refine([paraphrase_step()], llm, max_generations=3)) == 1
    second = llm.generation_requests()[1]
    assert second[-2] == {"role": "assistant", "content": GOOD}
    assert "the note is lost" in second[-1]["content"]


def test_rejection_reason_is_fed_back_without_a_self_check():
    llm = ScriptedLLM([LOST_VALUES, GOOD], [CONSISTENT])
    assert len(refine([paraphrase_step()], llm, max_generations=3)) == 1
    feedback = llm.generation_requests()[1][-1]["content"]
    assert "blue mug" in feedback
    # only the accepted reply was self-checked
    assert len(llm.conversations) == 3


def test_step_is_dropped_after_max_generations():
    llm = ScriptedLLM([GOOD, LOST_VALUES, GOOD], [inconsistent("a"), inconsistent("b")])
    assert refine([paraphrase_step()], llm, max_generations=3) == []
    assert len(llm.generation_requests()) == 3
    assert llm.checks == []


def test_empty_reply_is_generated_again_without_feedback():
    llm = ScriptedLLM([None, GOOD], [CONSISTENT])
    assert len(refine([paraphrase_step()], llm, max_generations=2)) == 1
    first, second = llm.generation_requests()
    assert first == second


def test_steps_without_self_check_skip_it():
    step = ToolDocParaphrase().apply(make_draft(), random.Random(0))
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
    llm = ScriptedLLM([json.dumps({"tools": docs})], [])
    [draft] = refine([step], llm, max_generations=3)
    assert draft.view.tool_descriptions["create_order"] == "Buy."


def test_slow_finishes_run_in_parallel_with_the_same_results():
    steps = [paraphrase_step() for _ in range(4)]
    llm = ScriptedLLM([GOOD] * 4, [CONSISTENT] * 4)
    assert len(refine(steps, llm, max_generations=3, workers=4)) == 4


def test_unreadable_self_check_is_asked_again_then_dropped():
    llm = ScriptedLLM([GOOD], ["not json"] * (MAX_CHECK_RETRY + 1))
    assert refine([paraphrase_step()], llm, max_generations=3) == []
    assert len(llm.conversations) == 1 + MAX_CHECK_RETRY + 1


def test_self_check_prompt_shows_the_followup_and_forbidden_calls():
    draft = make_draft(
        task="Order product 42",
        followup="The note is gift.",
        forbidden_calls=[{"tool_name": "create_order", "param": "note"}],
    )
    prompt = dialogue_check_prompt(draft)
    assert "The note is gift." in prompt
    assert '"tool_name": "create_order", "param": "note"' in prompt
    assert '"product_id": 42' in prompt


def test_self_check_prompt_without_followup_expects_a_single_turn():
    prompt = dialogue_check_prompt(make_draft())
    assert TASK in prompt
    assert "single turn" in prompt


def test_miss_param_steps_ask_for_a_self_check():
    step = MissParam().apply(make_draft(), random.Random(0))
    assert step is not None and step.check_prompt is not None
