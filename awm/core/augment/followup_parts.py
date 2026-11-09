"""Parts after which the agent has to hold back until a scripted follow-up turn."""

import dataclasses
import random

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Part,
    Rejection,
    param_values_in_task,
    rewritten_task,
)
from awm.core.augment.task_parts import rewrite_step
from awm.prompts import AUG_MISS_FUNC_FOLLOWUP, AUG_MISS_PARAM_PROMPT


class MissFunc(Part):
    """Hide the tool that completes the task until the follow-up turn."""

    name = "miss_func"
    needs_reference = True
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        called = [
            c["tool_name"]
            for c in draft.reference_calls or []
            if c["tool_name"] in draft.tool_names
        ]
        if not called:
            return None
        # the last call is usually the action that completes the task
        hidden = called[-1]
        return draft.evolve(
            self.name,
            hidden_tool=hidden,
            followup=AUG_MISS_FUNC_FOLLOWUP.format(tool_name=hidden),
            forbidden_calls=[{"tool_name": hidden, "param": None}],
        )


class MissParam(Part):
    """Remove a value the task needs; the user gives it in the follow-up turn."""

    name = "miss_param"
    needs_reference = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        triples = param_values_in_task(draft.task, draft.reference_calls or [])
        if not triples:
            return None
        tool_name, param, value = rng.choice(triples)
        rewritten = draft.evolve(
            self.name,
            removed_value=value,
            forbidden_calls=[{"tool_name": tool_name, "param": param}],
        )

        def finish(response: dict) -> Draft | Rejection:
            task = rewritten_task(rewritten, response)
            if isinstance(task, Rejection):
                return task
            followup = response.get("followup")
            if not isinstance(followup, str) or value not in followup:
                return Rejection(
                    f'"followup" must be a string that contains "{value}" verbatim.'
                )
            return dataclasses.replace(rewritten, task=task, followup=followup.strip())

        prompt = AUG_MISS_PARAM_PROMPT.format(
            task=draft.task, value=value, param=param, tool_name=tool_name
        )
        return rewrite_step(draft, prompt, finish)
