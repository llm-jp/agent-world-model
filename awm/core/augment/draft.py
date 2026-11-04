"""The task being augmented, the interface of a part, and turning a draft into data.

A part reads a Draft and returns a new one (or an LLMStep that produces it),
so parts never depend on each other: whatever an earlier part did is visible
only through the draft's fields.
"""

import dataclasses
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from awm.core.tool_view import TaskMeta, ToolView, Turn
from awm.core.verify import find_verifier_function_name
from awm.prompts import AUG_MISS_FUNC_FOLLOWUP

# shortest argument value treated as taken from the task text
MIN_VALUE_LENGTH = 2


@dataclass(frozen=True)
class Draft:
    scenario: str
    source_task_idx: int
    task: str
    source_verifier_code: str
    # tools of the scenario as the server lists them; None if unknown
    tools: list[dict] | None
    # tool calls of a solved reference run of the source task; None if unsolved
    reference_calls: list[dict] | None
    # tools of the other scenarios, a pool for distractors
    other_tools: list[dict] = field(default_factory=list)

    # how the tools are presented in every turn
    view: ToolView = field(default_factory=ToolView)
    # original tool names hidden until the follow-up turn
    hidden_tools: list[str] = field(default_factory=list)
    # user messages of the follow-up turn, in the order the parts added them
    followups: list[str] = field(default_factory=list)
    # [{"tool_name", "param" or None}] the agent must not call before the follow-up
    forbidden_calls: list[dict] = field(default_factory=list)
    # values removed from the task text; later rewrites must not bring them back
    removed_values: list[str] = field(default_factory=list)
    # texts later rewrites must keep (e.g. added tool names)
    required_mentions: list[str] = field(default_factory=list)
    # tool names later rewrites must not mention
    banned_mentions: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)

    def evolve(self, part: str, **changes: Any) -> "Draft":
        """A copy with the changes, recording that the part was applied."""
        return dataclasses.replace(self, applied=[*self.applied, part], **changes)

    def replace(self, **changes: Any) -> "Draft":
        """A copy with the changes, for a part already recorded by evolve."""
        return dataclasses.replace(self, **changes)

    def with_view(self, part: str, **view_changes: Any) -> "Draft":
        view = self.view.model_copy(update=view_changes, deep=True)
        return self.evolve(part, view=view)

    def presented_name(self, tool_name: str) -> str:
        return self.view.rename_tools.get(tool_name, tool_name)

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools or []]

    @property
    def presented_names(self) -> list[str]:
        names = [self.presented_name(n) for n in self.tool_names]
        return names + [t["name"] for t in self.view.distractor_tools]


@dataclass
class LLMStep:
    """A part's request to the LLM and how to turn the parsed reply into a draft."""

    # the draft the part was applied to
    draft: Draft
    messages: list[dict]
    finish: Callable[[dict], Draft | None]


class Part:
    """One augmentation; subclasses set the class attributes and implement apply."""

    name: ClassVar[str]
    needs_reference: ClassVar[bool] = False
    needs_tools: ClassVar[bool] = False

    def applicable(self, draft: Draft) -> bool:
        if self.needs_reference and not draft.reference_calls:
            return False
        return not (self.needs_tools and not draft.tools)

    def apply(self, draft: Draft, rng: random.Random) -> Draft | LLMStep | None:
        raise NotImplementedError


# ---------------------------------------------------------------- helpers


def properties(tool: dict) -> dict:
    return (tool.get("inputSchema") or {}).get("properties", {})


def required(tool: dict) -> list[str]:
    return (tool.get("inputSchema") or {}).get("required", [])


def scalar_values(value: Any) -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, dict):
        return [v for item in value.values() for v in scalar_values(item)]
    if isinstance(value, list):
        return [v for item in value for v in scalar_values(item)]
    return []


def values_in_task(task: str, calls: list[dict]) -> list[str]:
    """Argument values of the reference calls that appear verbatim in the task."""
    found = []
    for call in calls:
        for value in scalar_values(call.get("arguments") or {}):
            if len(value) >= MIN_VALUE_LENGTH and value in task and value not in found:
                found.append(value)
    return found


def param_values_in_task(task: str, calls: list[dict]) -> list[tuple[str, str, str]]:
    """(tool, param, value) of top-level reference arguments taken from the task."""
    found = []
    for call in calls:
        for param, value in (call.get("arguments") or {}).items():
            values = scalar_values(value)
            if len(values) != 1:
                continue
            text = values[0]
            if len(text) >= MIN_VALUE_LENGTH and text in task:
                triple = (call["tool_name"], param, text)
                if triple not in found:
                    found.append(triple)
    return found


def mentioned_tool_names(text: str, tool_names: list[str]) -> list[str]:
    """Tool names that appear in the text, also written with spaces."""
    lowered = text.lower()
    return [
        name
        for name in tool_names
        if name.lower() in lowered or name.replace("_", " ").lower() in lowered
    ]


def keep_values(draft: Draft) -> list[str]:
    """Values a rewrite of the draft's task must keep verbatim."""
    values = values_in_task(draft.task, draft.reference_calls or [])
    return [
        v
        for v in values
        if v not in draft.removed_values
        and not mentioned_tool_names(v, draft.banned_mentions)
    ]


def rewritten_task(draft: Draft, response: dict) -> str | None:
    """The rewritten task in the reply, if it keeps what earlier parts set up."""
    task = response.get("task")
    if not isinstance(task, str) or not task.strip():
        return None
    task = task.strip()
    if task == draft.task:
        return None
    if not all(v in task for v in keep_values(draft) + draft.required_mentions):
        return None
    if any(v in task for v in draft.removed_values):
        return None
    if mentioned_tool_names(task, draft.banned_mentions):
        return None
    return task


# ------------------------------------------------- follow-up verifier


FOLLOWUP_VERIFIER_TEMPLATE = '''def verify_task_completion(initial_db_path, final_db_path, final_answer="", turn_db_paths=None, tool_calls=None):
    """Check the first turn of the dialogue, then run the original task's verifier."""
    import inspect
    import json
    import os
    import re
    import sqlite3

    rules = __RULES__

    def dump(path):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return list(conn.iterdump())
        finally:
            conn.close()

    if rules["unchanged_db_before_followup"]:
        if not turn_db_paths:
            return {"result": "others", "reason": "no database snapshot of the first turn"}
        if dump(initial_db_path) != dump(turn_db_paths[0]):
            return {"result": "others", "reason": "the database changed before the user's follow-up"}

    for call in tool_calls or []:
        if call.get("turn", 0) != 0:
            continue
        for rule in rules["forbidden_calls_before_followup"]:
            if call.get("tool_name") != rule["tool_name"]:
                continue
            if rule.get("param") is None or rule["param"] in (call.get("arguments") or {}):
                return {"result": "others", "reason": "called " + rule["tool_name"] + " before the user's follow-up"}

    namespace = {"sqlite3": sqlite3, "json": json, "os": os, "re": re}
    exec(__SOURCE_CODE__, namespace)
    source = namespace[__SOURCE_FUNCTION__]
    params = inspect.signature(source).parameters
    kwargs = {}
    if "initial_db_path" in params:
        kwargs["initial_db_path"] = initial_db_path
    if "final_db_path" in params:
        kwargs["final_db_path"] = final_db_path
    if "final_answer" in params:
        kwargs["final_answer"] = final_answer or ""
    if not kwargs and len(params) >= 2:
        args = [initial_db_path, final_db_path]
        if len(params) >= 3:
            args.append(final_answer or "")
        return source(*args)
    return source(**kwargs)
'''


def build_followup_verifier(
    source_code: str, forbidden_calls: list[dict], unchanged_db: bool = True
) -> str:
    """Verifier code of a follow-up task, wrapping the original task's verifier.

    forbidden_calls: [{"tool_name": ..., "param": ... or None}] that must not be
    called in the first turn (param None forbids any call of the tool).
    """
    rules = {
        "unchanged_db_before_followup": unchanged_db,
        "forbidden_calls_before_followup": forbidden_calls,
    }
    return (
        FOLLOWUP_VERIFIER_TEMPLATE.replace("__RULES__", repr(rules))
        .replace("__SOURCE_CODE__", repr(source_code))
        .replace(
            "__SOURCE_FUNCTION__",
            repr(find_verifier_function_name(source_code, "code")),
        )
    )


# ---------------------------------------------------------------- output


@dataclass
class Variant:
    """An augmented task ready to be appended to the dataset."""

    scenario: str
    task: str
    meta: TaskMeta
    verifier_code: str


def finalize(draft: Draft) -> Variant:
    """Assemble the task meta and verifier from everything the parts set up."""
    base_view = None if draft.view == ToolView() else draft.view
    first_view = base_view
    if draft.hidden_tools:
        first_view = draft.view.model_copy(
            update={"hide_tools": [*draft.view.hide_tools, *draft.hidden_tools]},
            deep=True,
        )

    messages = list(draft.followups) + [
        AUG_MISS_FUNC_FOLLOWUP.format(tool_name=draft.presented_name(name))
        for name in draft.hidden_tools
    ]
    turns = []
    if messages:
        # the follow-up turn restores the tools hidden in the first one
        turns.append(
            Turn(
                user="\n".join(messages),
                tool_view=(base_view or ToolView()) if draft.hidden_tools else None,
            )
        )

    verifier_code = draft.source_verifier_code
    if turns:
        verifier_code = build_followup_verifier(
            draft.source_verifier_code, draft.forbidden_calls
        )

    meta = TaskMeta(
        augmentations=list(draft.applied),
        source_task_idx=draft.source_task_idx,
        tool_view=first_view,
        turns=turns,
    )
    return Variant(
        scenario=draft.scenario,
        task=draft.task,
        meta=meta,
        verifier_code=verifier_code,
    )
