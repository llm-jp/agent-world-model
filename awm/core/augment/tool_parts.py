"""Parts that change how the tools are presented; the expected behaviour stays the same.

Renaming tools or parameters is not a view; see env_parts.py.
"""

import json
import random
from typing import Any

from awm.core.augment.draft import (
    Draft,
    LLMStep,
    Part,
    Rejection,
    properties,
)
from awm.core.tool_view import ToolView
from awm.prompts import AUG_TOOL_DOC_PARAPHRASE_PROMPT

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
    "client_version": {
        "type": "string",
        "description": "Optional version string of the calling client.",
    },
}

MAX_DISTRACTOR_TOOLS = 3


def presenting(draft: Draft, part: str, **view: Any) -> Draft:
    """The draft whose tools are presented with the given view settings."""
    return draft.evolve(part, view=ToolView(**view))


class AddParams(Part):
    name = "add_params"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        extra = {}
        for tool in draft.tools or []:
            free = [p for p in EXTRA_PARAM_POOL if p not in properties(tool)]
            if free:
                param = rng.choice(free)
                extra[tool["name"]] = {param: EXTRA_PARAM_POOL[param]}
        if not extra:
            return None
        return presenting(draft, self.name, extra_params=extra)


class DistractorTools(Part):
    name = "distractor_tools"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        taken = set(draft.tool_names)
        candidates = [t for t in draft.other_tools if t["name"] not in taken]
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


class ToolDocParaphrase(Part):
    name = "tool_doc_paraphrase"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep[Draft]:
        docs = {
            t["name"]: {
                "description": t.get("description", ""),
                "parameters": {
                    p: s.get("description", "") for p, s in properties(t).items()
                },
            }
            for t in draft.tools or []
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
                tool_descriptions[name] = new["description"]
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
