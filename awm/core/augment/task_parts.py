"""Parts that rewrite the task text with an LLM; the expected behaviour stays the same."""

import json
import random

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Part,
    keep_values,
    mentioned_tool_names,
    rewritten_task,
)
from awm.prompts import (
    AUG_TASK_AMBIGUOUS_PROMPT,
    AUG_TASK_DISTRACTOR_NAMES_PROMPT,
    AUG_TASK_NO_TOOL_NAMES_PROMPT,
    AUG_TASK_PARAPHRASE_PROMPT,
    AUG_TASK_REWRITE_SYSTEM_PROMPT,
)

MAX_DISTRACTOR_NAMES = 2


def rewrite_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": AUG_TASK_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _dump(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


class TaskParaphrase(Part):
    name = "task_paraphrase"
    needs_reference = True
    template = AUG_TASK_PARAPHRASE_PROMPT

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep:
        def finish(response: dict) -> Draft | None:
            task = rewritten_task(draft, response)
            return None if task is None else draft.evolve(self.name, task=task)

        keep = keep_values(draft) + draft.required_mentions
        prompt = self.template.format(task=draft.task, keep_values=_dump(keep))
        return LLMStep(draft, rewrite_messages(prompt), finish)


class TaskAmbiguous(TaskParaphrase):
    name = "task_ambiguous"
    template = AUG_TASK_AMBIGUOUS_PROMPT


class TaskNoToolNames(Part):
    name = "task_no_tool_names"
    needs_reference = True
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep | None:
        names = sorted(set(draft.tool_names) | set(draft.presented_names))
        mentioned = [
            n
            for n in mentioned_tool_names(draft.task, names)
            if n not in draft.required_mentions
        ]
        if not mentioned:
            return None
        banned = sorted(set(draft.banned_mentions) | set(mentioned))
        rewritten = draft.evolve(self.name, banned_mentions=banned)

        def finish(response: dict) -> Draft | None:
            task = rewritten_task(rewritten, response)
            return None if task is None else rewritten.replace(task=task)

        prompt = AUG_TASK_NO_TOOL_NAMES_PROMPT.format(
            task=draft.task,
            tool_names=_dump(mentioned),
            keep_values=_dump(keep_values(rewritten) + draft.required_mentions),
        )
        return LLMStep(draft, rewrite_messages(prompt), finish)


class TaskDistractorNames(Part):
    name = "task_distractor_names"
    needs_reference = True
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep | None:
        used = {
            draft.presented_name(c["tool_name"]) for c in draft.reference_calls or []
        }
        unused = [
            n
            for n in draft.presented_names
            if n not in used
            and n not in draft.required_mentions
            and not mentioned_tool_names(n, draft.banned_mentions)
        ]
        if not unused:
            return None
        distractors = rng.sample(unused, min(MAX_DISTRACTOR_NAMES, len(unused)))
        rewritten = draft.evolve(
            self.name, required_mentions=[*draft.required_mentions, *distractors]
        )

        def finish(response: dict) -> Draft | None:
            task = rewritten_task(rewritten, response)
            return None if task is None else rewritten.replace(task=task)

        prompt = AUG_TASK_DISTRACTOR_NAMES_PROMPT.format(
            task=draft.task,
            distractor_names=_dump(distractors),
            keep_values=_dump(keep_values(draft) + draft.required_mentions),
        )
        return LLMStep(draft, rewrite_messages(prompt), finish)
