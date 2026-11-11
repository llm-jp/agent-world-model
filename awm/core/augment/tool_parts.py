"""Parts that change how the tools are presented; the expected behaviour stays the same.

Each augmented task gets an environment of its own whose server code serves
the task's view (run_recipe in recipe.py). Renaming tools or parameters is not
a view; see env_parts.py.
"""

import json
import random
from typing import Any

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Rejection,
    properties,
)
from awm.core.augment.part import TaskPart
from awm.core.tool_view import ToolView, schema_types, value_formats
from awm.core.view_server import RESPONSES_MARKER
from awm.prompts import AUG_PARAM_ENUM_PROMPT, AUG_TOOL_DOC_PARAPHRASE_PROMPT

# optional parameters added by add_params; the call drops them before the server
EXTRA_PARAM_POOL = {
    "request_id": {
        "type": "string",
        "description": "Optional client-side identifier of this request.",
    },
    "trace_id": {
        "type": "string",
        "description": "Optional identifier used to trace the request across services.",
    },
    "correlation_id": {
        "type": "string",
        "description": "Optional identifier that links related requests together.",
    },
    "client_version": {
        "type": "string",
        "description": "Optional version string of the calling client.",
    },
    "client_name": {
        "type": "string",
        "description": "Optional name of the application making the request.",
    },
    "user_agent": {
        "type": "string",
        "description": "Optional user agent string of the calling client.",
    },
}
# extra parameters per tool; fewer when the pool has fewer unused names
MIN_EXTRA_PARAMS = 2
MAX_EXTRA_PARAMS = 3

MAX_DISTRACTOR_TOOLS = 3
# parameters param_format writes in another format
MAX_FORMAT_PARAMS = 2
# size of an enum param_enum makes
MIN_ENUM_VALUES = 2
MAX_ENUM_VALUES = 12


def presenting(draft: Draft, part: str, **view: Any) -> Draft:
    """The draft whose tools are also presented with the given view settings.

    A recipe has one part per view field (Part.writes), so the settings of the
    earlier parts stay.
    """
    return draft.evolve(part, view=draft.view.model_copy(update=view))


def view_field(name: str) -> str:
    """The Part.writes entry of a ToolView field."""
    assert name in ToolView.model_fields, name
    return f"view.{name}"


class AddParams(TaskPart):
    name = "add_params"
    writes = frozenset({view_field("extra_params")})
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> Draft | None:
        extra = {}
        for tool in draft.tools or []:
            free = [p for p in EXTRA_PARAM_POOL if p not in properties(tool)]
            count = min(len(free), rng.randint(MIN_EXTRA_PARAMS, MAX_EXTRA_PARAMS))
            if count:
                extra[tool["name"]] = {
                    p: EXTRA_PARAM_POOL[p] for p in rng.sample(free, count)
                }
        if not extra:
            return None
        return presenting(draft, self.name, extra_params=extra)


class DistractorTools(TaskPart):
    name = "distractor_tools"
    writes = frozenset({view_field("distractor_tools")})
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> Draft | None:
        taken = set(draft.tool_names)
        # the server lists a tool without a description by its method and path
        candidates = [
            t
            for t in draft.other_tools
            if t["name"] not in taken and t.get("description")
        ]
        if not candidates:
            return None
        picked = rng.sample(candidates, min(MAX_DISTRACTOR_TOOLS, len(candidates)))
        distractors = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema") or {},
            }
            for t in picked
        ]
        return presenting(draft, self.name, distractor_tools=distractors)


def split_responses(description: str) -> tuple[str, str]:
    """A listed tool description, split before the responses the server appends."""
    head, marker, responses = description.partition(RESPONSES_MARKER)
    return head, marker + responses


class ToolDocParaphrase(TaskPart):
    """Reword the tool and parameter descriptions; the listed responses stay."""

    name = "tool_doc_paraphrase"
    writes = frozenset(
        {view_field("tool_descriptions"), view_field("param_descriptions")}
    )
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> LLMStep[Draft]:
        tools = draft.tools or []
        split = {t["name"]: split_responses(t.get("description", "")) for t in tools}
        docs = {
            t["name"]: {
                "description": split[t["name"]][0],
                "parameters": {
                    p: s.get("description", "") for p, s in properties(t).items()
                },
            }
            for t in tools
        }

        def finish(response: dict) -> Draft | Rejection:
            rewritten = response.get("tools")
            if not isinstance(rewritten, dict) or set(rewritten) != set(docs):
                return Rejection(
                    f'"tools" must have exactly these keys: {sorted(docs)}'
                )
            tool_descriptions = {}
            param_descriptions = {}
            for name, doc in docs.items():
                new = rewritten[name]
                if not isinstance(new, dict) or not isinstance(
                    new.get("description"), str
                ):
                    return Rejection(f'"{name}" needs a string "description".')
                if RESPONSES_MARKER.strip() in new["description"]:
                    return Rejection(
                        f'The description of "{name}" must not have a "{RESPONSES_MARKER.strip()}" section.'
                    )
                new_params = new.get("parameters") or {}
                if not isinstance(new_params, dict) or set(new_params) != set(
                    doc["parameters"]
                ):
                    return Rejection(
                        f'"parameters" of "{name}" must have exactly these keys: {sorted(doc["parameters"])}'
                    )
                if not all(isinstance(v, str) for v in new_params.values()):
                    return Rejection(
                        f'Every parameter description of "{name}" must be a string.'
                    )
                tool_descriptions[name] = new["description"] + split[name][1]
                if new_params:
                    param_descriptions[name] = new_params
            return presenting(
                draft,
                self.name,
                tool_descriptions=tool_descriptions,
                param_descriptions=param_descriptions,
            )

        prompt = AUG_TOOL_DOC_PARAPHRASE_PROMPT.format(
            tools_json=json.dumps(docs, ensure_ascii=False)
        )
        return LLMStep(
            [{"role": "user", "content": prompt}],
            finish,
            label=f"{draft.scenario} task {draft.source_task_idx}",
        )


def passed_values(draft: Draft) -> dict[tuple[str, str], list]:
    """Values the reference calls pass, per (tool, top-level parameter)."""
    found: dict[tuple[str, str], list] = {}
    for call in draft.reference_calls or []:
        for param, value in (call.get("arguments") or {}).items():
            values = found.setdefault((call["tool_name"], param), [])
            if value not in values:
                values.append(value)
    return found


class ParamFormat(TaskPart):
    """Present parameters the reference calls pass with their values in another format."""

    name = "param_format"
    writes = frozenset({view_field("param_formats")})
    needs_reference = True
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> Draft | None:
        view = draft.view
        tools = {t["name"]: t for t in draft.tools or []}
        candidates: dict[tuple[str, str], list[str]] = {}
        for (tool, param), values in passed_values(draft).items():
            schema = properties(tools.get(tool, {})).get(param)
            # an enum lists the values as the server takes them
            if schema is None or param in view.param_enums.get(tool, {}):
                continue
            # a format must write every value the task passes
            formats = [
                f
                for f in value_formats(schema, values[0])
                if all(f in value_formats(schema, v) for v in values[1:])
            ]
            if formats:
                candidates[(tool, param)] = formats
        if not candidates:
            return None
        picked = rng.sample(sorted(candidates), min(MAX_FORMAT_PARAMS, len(candidates)))
        param_formats: dict[str, dict[str, str]] = {}
        for tool, param in picked:
            param_formats.setdefault(tool, {})[param] = rng.choice(
                candidates[(tool, param)]
            )
        return presenting(draft, self.name, param_formats=param_formats)


class ParamEnum(TaskPart):
    """Let the LLM turn the string parameters that take a few fixed values into enums."""

    name = "param_enum"
    writes = frozenset({view_field("param_enums")})
    needs_reference = True
    needs_tools = True

    def apply_one(self, draft: Draft, rng: random.Random) -> LLMStep[Draft] | None:
        passed = passed_values(draft)
        docs: dict[str, dict[str, dict]] = {}
        for tool in draft.tools or []:
            formatted = draft.view.param_formats.get(tool["name"], {})
            for param, schema in properties(tool).items():
                if (
                    "string" not in schema_types(schema)
                    or "enum" in schema
                    or param in formatted
                ):
                    continue
                docs.setdefault(tool["name"], {})[param] = {
                    "description": schema.get("description", ""),
                    "passed_values": [
                        v
                        for v in passed.get((tool["name"], param), [])
                        if isinstance(v, str)
                    ],
                }
        if not docs:
            return None

        def finish(response: dict) -> Draft | Rejection:
            enums = response.get("enums")
            if not isinstance(enums, dict) or not enums:
                return Rejection(
                    '"enums" must map at least one tool to its parameters.'
                )
            problems = []
            for tool, params in enums.items():
                if tool not in docs or not isinstance(params, dict) or not params:
                    problems.append(
                        f'"{tool}" is not a tool with string parameters listed above'
                    )
                    continue
                for param, values in params.items():
                    where = f"{tool}.{param}"
                    if param not in docs[tool]:
                        problems.append(
                            f"{where} is not a string parameter listed above"
                        )
                    elif (
                        not isinstance(values, list)
                        or not all(isinstance(v, str) and v for v in values)
                        or len(set(values)) != len(values)
                    ):
                        problems.append(
                            f"{where} needs a list of different non-empty strings"
                        )
                    elif not MIN_ENUM_VALUES <= len(values) <= MAX_ENUM_VALUES:
                        problems.append(
                            f"{where} needs {MIN_ENUM_VALUES} to {MAX_ENUM_VALUES} values, got {len(values)}"
                        )
                    elif missing := [
                        v for v in docs[tool][param]["passed_values"] if v not in values
                    ]:
                        problems.append(
                            f"{where} must include the passed values {missing}"
                        )
            if problems:
                return Rejection("; ".join(problems))
            return presenting(draft, self.name, param_enums=enums)

        prompt = AUG_PARAM_ENUM_PROMPT.format(
            params_json=json.dumps(docs, ensure_ascii=False),
            min_values=MIN_ENUM_VALUES,
            max_values=MAX_ENUM_VALUES,
        )
        return LLMStep(
            [{"role": "user", "content": prompt}],
            finish,
            label=f"{draft.scenario} task {draft.source_task_idx} enums",
        )
