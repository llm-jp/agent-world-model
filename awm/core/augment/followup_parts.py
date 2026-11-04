"""Parts after which the agent has to hold back until a scripted follow-up turn."""

import random

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Part,
    param_values_in_task,
    rewritten_task,
)
from awm.core.augment.task_parts import rewrite_messages
from awm.prompts import AUG_MISS_PARAM_PROMPT


class MissFunc(Part):
    """Hide the tool that completes the task until the follow-up turn."""

    name = "miss_func"
    needs_reference = True
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        names = set(draft.tool_names) - set(draft.hidden_tools)
        called = [
            c["tool_name"]
            for c in draft.reference_calls or []
            if c["tool_name"] in names
        ]
        if not called:
            return None
        # the last call is usually the action that completes the task
        hidden = called[-1]
        return draft.evolve(
            self.name,
            hidden_tools=[*draft.hidden_tools, hidden],
            forbidden_calls=[
                *draft.forbidden_calls,
                {"tool_name": hidden, "param": None},
            ],
        )


class MissParam(Part):
    """Remove a value the task needs; the user gives it in the follow-up turn."""

    name = "miss_param"
    needs_reference = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep | None:
        triples = [
            t
            for t in param_values_in_task(draft.task, draft.reference_calls or [])
            if t[2] not in draft.required_mentions
        ]
        if not triples:
            return None
        tool_name, param, value = rng.choice(triples)
        rewritten = draft.evolve(
            self.name,
            removed_values=[*draft.removed_values, value],
            forbidden_calls=[
                *draft.forbidden_calls,
                {"tool_name": tool_name, "param": param},
            ],
        )

        def finish(response: dict) -> Draft | None:
            task = rewritten_task(rewritten, response)
            followup = response.get("followup")
            if task is None or not isinstance(followup, str) or value not in followup:
                return None
            return rewritten.replace(
                task=task, followups=[*rewritten.followups, followup.strip()]
            )

        prompt = AUG_MISS_PARAM_PROMPT.format(
            task=draft.task,
            value=value,
            param=param,
            tool_name=draft.presented_name(tool_name),
        )
        return LLMStep(draft, rewrite_messages(prompt), finish)
