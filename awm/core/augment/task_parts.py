"""Parts that rewrite the task text with an LLM; the expected behaviour stays the same."""

import dataclasses
import json
import random
from collections.abc import Callable

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Part,
    Rejection,
    keep_values,
    mentioned_tool_names,
    rewritten_task,
)
from awm.prompts import (
    AUG_SELF_CHECK_FOLLOWUP_BEHAVIOUR,
    AUG_SELF_CHECK_NO_FOLLOWUP,
    AUG_SELF_CHECK_PROMPT,
    AUG_SELF_CHECK_SINGLE_TURN_BEHAVIOUR,
    AUG_TASK_AMBIGUOUS_PROMPT,
    AUG_TASK_DISTRACTOR_NAMES_PROMPT,
    AUG_TASK_FEEDBACK_PROMPT,
    AUG_TASK_NO_TOOL_NAMES_PROMPT,
    AUG_TASK_PARAPHRASE_PROMPT,
    AUG_TASK_REWRITE_SYSTEM_PROMPT,
)

MAX_DISTRACTOR_NAMES = 2


def dialogue_check_prompt(draft: Draft) -> str:
    """Asks whether the draft's user messages fit what the agent is expected to do."""
    if draft.followup is not None:
        behaviour = AUG_SELF_CHECK_FOLLOWUP_BEHAVIOUR.format(
            forbidden_calls=json.dumps(draft.forbidden_calls, ensure_ascii=False)
        )
    else:
        behaviour = AUG_SELF_CHECK_SINGLE_TURN_BEHAVIOUR
    calls = [
        {"tool_name": c["tool_name"], "arguments": c.get("arguments") or {}}
        for c in draft.reference_calls or []
    ]
    return AUG_SELF_CHECK_PROMPT.format(
        task=draft.task,
        followup=draft.followup or AUG_SELF_CHECK_NO_FOLLOWUP,
        behaviour=behaviour,
        calls=json.dumps(calls, ensure_ascii=False, indent=2),
    )


def rewrite_step(
    draft: Draft, prompt: str, finish: Callable[[dict], Draft | Rejection]
) -> LLMStep[Draft]:
    """A task rewrite, checked against the reference run before it is accepted."""
    messages = [
        {"role": "system", "content": AUG_TASK_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return LLMStep(
        messages,
        finish,
        label=f"{draft.scenario} task {draft.source_task_idx}",
        check_prompt=dialogue_check_prompt,
        feedback_prompt=AUG_TASK_FEEDBACK_PROMPT,
    )


def accept_task(draft: Draft, response: dict) -> Draft | Rejection:
    task = rewritten_task(draft, response)
    if isinstance(task, Rejection):
        return task
    return dataclasses.replace(draft, task=task)


def _dump(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


class TaskParaphrase(Part):
    name = "task_paraphrase"
    needs_reference = True
    template = AUG_TASK_PARAPHRASE_PROMPT

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep[Draft]:
        rewritten = draft.evolve(self.name)

        def finish(response: dict) -> Draft | Rejection:
            return accept_task(rewritten, response)

        prompt = self.template.format(
            task=draft.task, keep_values=_dump(keep_values(draft))
        )
        return rewrite_step(draft, prompt, finish)


class TaskAmbiguous(TaskParaphrase):
    name = "task_ambiguous"
    template = AUG_TASK_AMBIGUOUS_PROMPT


class TaskNoToolNames(Part):
    name = "task_no_tool_names"
    needs_reference = True
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        mentioned = mentioned_tool_names(draft.task, draft.tool_names)
        if not mentioned:
            return None
        rewritten = draft.evolve(self.name, banned_mentions=mentioned)

        def finish(response: dict) -> Draft | Rejection:
            return accept_task(rewritten, response)

        prompt = AUG_TASK_NO_TOOL_NAMES_PROMPT.format(
            task=draft.task,
            tool_names=_dump(mentioned),
            keep_values=_dump(keep_values(rewritten)),
        )
        return rewrite_step(draft, prompt, finish)


class TaskDistractorNames(Part):
    name = "task_distractor_names"
    needs_reference = True
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        used = {c["tool_name"] for c in draft.reference_calls or []}
        unused = [n for n in draft.tool_names if n not in used]
        if not unused:
            return None
        distractors = rng.sample(unused, min(MAX_DISTRACTOR_NAMES, len(unused)))
        rewritten = draft.evolve(self.name, required_mentions=distractors)

        def finish(response: dict) -> Draft | Rejection:
            return accept_task(rewritten, response)

        prompt = AUG_TASK_DISTRACTOR_NAMES_PROMPT.format(
            task=draft.task,
            distractor_names=_dump(distractors),
            keep_values=_dump(keep_values(draft)),
        )
        return rewrite_step(draft, prompt, finish)
