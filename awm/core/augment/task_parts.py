"""Parts that rewrite the task text with an LLM; the expected behaviour stays the same."""

import dataclasses
import json
import random
from collections.abc import Callable

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Rejection,
    keep_values,
    mentioned_tool_names,
    must_keep,
    rewritten_task,
)
from awm.core.augment.part import TaskPart
from awm.prompts import (
    AUG_SELF_CHECK_FOLLOWUP_BEHAVIOUR,
    AUG_SELF_CHECK_NO_FOLLOWUP,
    AUG_SELF_CHECK_PROMPT,
    AUG_SELF_CHECK_SINGLE_TURN_BEHAVIOUR,
    AUG_SELF_CHECK_VALUE_FORMATS,
    AUG_TASK_AMBIGUOUS_PROMPT,
    AUG_TASK_DISTRACTOR_NAMES_PROMPT,
    AUG_TASK_FEEDBACK_PROMPT,
    AUG_TASK_NO_TOOL_NAMES_PROMPT,
    AUG_TASK_REWRITE_SYSTEM_PROMPT,
    AUG_TASK_VALUE_FORMAT_PROMPT,
)

MAX_DISTRACTOR_NAMES = 2
# Part.writes entry of the parts that decide which tool names the task mentions
TOOL_MENTIONS = "tool_mentions"


def dialogue_check_prompt(draft: Draft) -> str:
    """Asks whether the draft's user messages fit what the agent is expected to do."""
    if draft.followup is not None:
        behaviour = AUG_SELF_CHECK_FOLLOWUP_BEHAVIOUR.format(
            forbidden_calls=json.dumps(draft.forbidden_calls, ensure_ascii=False)
        )
    else:
        behaviour = AUG_SELF_CHECK_SINGLE_TURN_BEHAVIOUR
    if draft.value_rewrites:
        behaviour += AUG_SELF_CHECK_VALUE_FORMATS.format(
            value_rewrites=json.dumps(draft.value_rewrites, ensure_ascii=False)
        )
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


def accept_task(draft: Draft, response: dict) -> Draft | Rejection:
    """The draft with the rewritten task in the reply, if the task keeps what it requires."""
    task = rewritten_task(draft, response)
    if isinstance(task, Rejection):
        return task
    return dataclasses.replace(draft, task=task)


def rewrite_step(
    rewritten: Draft,
    prompt: str,
    finish: Callable[[dict], Draft | Rejection] | None = None,
) -> LLMStep[Draft]:
    """A task rewrite, checked against the reference run before it is accepted.

    rewritten: the draft the rewritten task goes into, with what it requires.
    finish: how a reply becomes the draft; by default its task is put into
    rewritten (accept_task).
    """
    messages = [
        {"role": "system", "content": AUG_TASK_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return LLMStep(
        messages,
        finish or (lambda response: accept_task(rewritten, response)),
        label=f"{rewritten.scenario} task {rewritten.source_task_idx}",
        check_prompt=dialogue_check_prompt,
        feedback_prompt=AUG_TASK_FEEDBACK_PROMPT,
    )


def _dump(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


class TaskAmbiguous(TaskPart):
    name = "task_ambiguous"
    needs_reference = True

    def apply_one(self, draft: Draft, rng: random.Random) -> LLMStep[Draft]:
        rewritten = draft.evolve(self.name)

        prompt = AUG_TASK_AMBIGUOUS_PROMPT.format(
            task=draft.task, keep_values=_dump(must_keep(draft))
        )
        return rewrite_step(rewritten, prompt)


class TaskNoToolNames(TaskPart):
    name = "task_no_tool_names"
    writes = frozenset({TOOL_MENTIONS})
    needs_reference = True
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        mentioned = mentioned_tool_names(draft.task, draft.tool_names)
        if not mentioned:
            return None
        # names an earlier part required (e.g. a renamed tool's) go as well
        rewritten = draft.evolve(
            self.name,
            banned_mentions=[*draft.banned_mentions, *mentioned],
            required_mentions=[
                m for m in draft.required_mentions if m not in mentioned
            ],
        )

        prompt = AUG_TASK_NO_TOOL_NAMES_PROMPT.format(
            task=draft.task,
            tool_names=_dump(mentioned),
            keep_values=_dump(must_keep(rewritten)),
        )
        return rewrite_step(rewritten, prompt)


class TaskDistractorNames(TaskPart):
    name = "task_distractor_names"
    writes = frozenset({TOOL_MENTIONS})
    needs_reference = True
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        used = {c["tool_name"] for c in draft.reference_calls or []}
        unused = [n for n in draft.tool_names if n not in used]
        if not unused:
            return None
        distractors = rng.sample(unused, min(MAX_DISTRACTOR_NAMES, len(unused)))
        rewritten = draft.evolve(
            self.name, required_mentions=[*draft.required_mentions, *distractors]
        )

        prompt = AUG_TASK_DISTRACTOR_NAMES_PROMPT.format(
            task=draft.task,
            distractor_names=_dump(distractors),
            keep_values=_dump(must_keep(draft)),
        )
        return rewrite_step(rewritten, prompt)


def rewritten_notations(
    response: dict, rewritable: list[str]
) -> dict[str, str] | Rejection:
    """The {value: notation} rewrites listed in the reply."""
    rewrites = response.get("rewrites")
    if not isinstance(rewrites, list) or not rewrites:
        return Rejection('"rewrites" must list at least one rewritten value.')
    notations: dict[str, str] = {}
    for item in rewrites:
        value = item.get("value") if isinstance(item, dict) else None
        written = item.get("written_as") if isinstance(item, dict) else None
        if value not in rewritable or value in notations:
            return Rejection(
                f"Each rewrite must name a different value you may rewrite: {rewritable}"
            )
        if not isinstance(written, str) or not written.strip() or written == value:
            return Rejection(f'"written_as" of "{value}" must be a new notation.')
        notations[value] = written.strip()
    return notations


class TaskValueFormat(TaskPart):
    """Let the LLM write values of the task in another notation the agent must convert."""

    name = "task_value_format"
    needs_reference = True

    def apply_one(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        values = keep_values(draft)
        # a value inside another one could not leave the task while the other stays
        rewritable = [v for v in values if not any(v != o and v in o for o in values)]
        if not rewritable:
            return None
        rewritten = draft.evolve(self.name)

        def finish(response: dict) -> Draft | Rejection:
            notations = rewritten_notations(response, rewritable)
            if isinstance(notations, Rejection):
                return notations
            kept = [v for v in values if v not in notations]
            task = rewritten_task(rewritten, response, kept)
            if isinstance(task, Rejection):
                return task
            problems = [
                f'"{value}" must not appear as it was'
                for value in notations
                if value in task
            ]
            problems += [
                f'"{written}" must appear in the request'
                for written in notations.values()
                if written not in task
            ]
            if problems:
                return Rejection("; ".join(problems))
            return dataclasses.replace(
                rewritten,
                task=task,
                value_rewrites={**draft.value_rewrites, **notations},
            )

        prompt = AUG_TASK_VALUE_FORMAT_PROMPT.format(
            task=draft.task,
            values=_dump(rewritable),
            keep_values=_dump(
                must_keep(draft, [v for v in values if v not in rewritable])
            ),
        )
        return rewrite_step(rewritten, prompt, finish)
