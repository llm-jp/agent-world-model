"""The task being augmented, LLM requests, and turning a draft into data.

A draft holds everything an augmented task needs: its text, how the tools are
presented, the follow-up turn and, after a part that rebuilds the environment,
the rewritten server code. The parts of a recipe (part.py) pass drafts along,
each one changing what it is about.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from awm.core.tool_view import TaskMeta, ToolView, Turn
from awm.core.verify import find_verifier_function_name
from awm.prompts import AUG_TOOL_FEEDBACK_PROMPT

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
    # final answer of that reference run
    reference_answer: str = ""
    # tools of the other scenarios, a pool for distractors
    other_tools: list[dict] = field(default_factory=list)

    # parts applied so far, in order; empty for a source task
    parts: tuple[str, ...] = ()
    # server code of a rebuilt environment; None keeps the source scenario's
    code: str | None = None
    # whether each reference call fails on the draft's server; None until replayed
    call_errors: tuple[bool, ...] | None = None
    # how the tools are presented in every turn; served by the environment's server
    view: ToolView = field(default_factory=ToolView)
    # tool hidden until the follow-up turn
    hidden_tool: str | None = None
    # user message of the follow-up turn
    followup: str | None = None
    # [{"tool_name", "param" or None}] the agent must not call before the follow-up
    forbidden_calls: list[dict] = field(default_factory=list)
    # value removed from the task text; a rewrite must not keep it
    removed_value: str | None = None
    # texts a rewrite must contain (e.g. tool names it has to mention)
    required_mentions: list[str] = field(default_factory=list)
    # tool names a rewrite must not mention
    banned_mentions: list[str] = field(default_factory=list)
    # argument value -> its notation in the task, for values written another way
    value_rewrites: dict[str, str] = field(default_factory=dict)

    def evolve(self, part: str, **changes: Any) -> "Draft":
        """A copy with the changes, produced by the part."""
        return dataclasses.replace(self, parts=(*self.parts, part), **changes)

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools or []]

    @property
    def has_view(self) -> bool:
        """Whether the tools are presented otherwise than the server lists them."""
        return self.view != ToolView()


@dataclass(frozen=True)
class Rejection:
    """Why a reply was not usable; sent back to the LLM as feedback."""

    reason: str


@dataclass
class LLMStep[T]:
    """A request to the LLM and how to turn the parsed reply into a result."""

    messages: list[dict]
    finish: Callable[[dict], T | Rejection]
    # names the step in logs
    label: str
    # prompt asking the LLM to check an accepted result; None skips the check
    check_prompt: Callable[[T], str] | None = None
    # user message asking for another reply; formatted with {problems}
    feedback_prompt: str = AUG_TOOL_FEEDBACK_PROMPT


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
        if v != draft.removed_value
        and not mentioned_tool_names(v, draft.banned_mentions)
    ]


def must_keep(draft: Draft, values: list[str] | None = None) -> list[str]:
    """Texts a rewrite of the draft's task must keep verbatim.

    values: the argument values to keep; defaults to keep_values(draft). The
    texts the earlier parts of a recipe require are added.
    """
    if values is None:
        values = keep_values(draft)
    texts = [*values, *draft.required_mentions, *draft.value_rewrites.values()]
    return list(dict.fromkeys(texts))


def constraint_problems(
    draft: Draft, task: str, values: list[str] | None = None
) -> list[str]:
    """How a rewrite of the draft's task breaks what the draft requires.

    values: the argument values to keep; defaults to keep_values(draft).
    """
    problems = []
    if missing := [v for v in must_keep(draft, values) if v not in task]:
        problems.append(f"These texts must appear verbatim but are missing: {missing}")
    if draft.removed_value is not None and draft.removed_value in task:
        problems.append(f'The value "{draft.removed_value}" must not appear.')
    if banned := mentioned_tool_names(task, draft.banned_mentions):
        problems.append(f"These tool names must not be mentioned in any form: {banned}")
    return problems


def rewritten_task(
    draft: Draft, response: dict, values: list[str] | None = None
) -> str | Rejection:
    """The rewritten task in the reply, if it keeps what the draft requires.

    values: the argument values to keep; defaults to keep_values(draft).
    """
    task = response.get("task")
    if not isinstance(task, str) or not task.strip():
        return Rejection('"task" must be a non-empty string.')
    task = task.strip()
    if task == draft.task:
        return Rejection("The rewritten request is identical to the original.")
    if problems := constraint_problems(draft, task, values):
        return Rejection("; ".join(problems))
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

    task: str
    meta: TaskMeta
    verifier_code: str


def finalize(draft: Draft) -> Variant:
    """Assemble the task meta and verifier of the recipe's last draft.

    draft.view is served by the environment's server (run_recipe), so the meta
    keeps only the tool hidden until the follow-up turn.
    """
    hidden = [draft.hidden_tool] if draft.hidden_tool is not None else []
    turns = []
    verifier_code = draft.source_verifier_code
    if draft.followup is not None:
        turns.append(Turn(user=draft.followup, shown_tools=hidden))
        verifier_code = build_followup_verifier(
            draft.source_verifier_code, draft.forbidden_calls
        )

    meta = TaskMeta(
        augmentations=list(draft.parts),
        source_task_idx=draft.source_task_idx,
        source_scenario=draft.scenario,
        hidden_tools=hidden,
        turns=turns,
    )
    return Variant(task=draft.task, meta=meta, verifier_code=verifier_code)
