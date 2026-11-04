"""Parts that change how the tools are presented; the expected behaviour stays the same."""

import json
import random

from awm.core.augment.draft import Draft, LLMStep, Part, properties, required
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


def to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def to_pascal(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


class ToolRename(Part):
    name = "tool_rename"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        if draft.view.rename_tools:
            return None
        style = rng.choice([to_camel, to_pascal])
        renames = {n: style(n) for n in draft.tool_names if style(n) != n}
        presented = [renames.get(n, n) for n in draft.tool_names] + [
            t["name"] for t in draft.view.distractor_tools
        ]
        if not renames or len(set(presented)) < len(presented):
            return None
        return draft.with_view(self.name, rename_tools=renames)


class ParamRename(Part):
    name = "param_rename"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        if draft.view.rename_params:
            return None
        renames = {}
        for tool in draft.tools or []:
            params = list(properties(tool))
            mapping = {p: to_camel(p) for p in params if to_camel(p) != p}
            extra = set(draft.view.extra_params.get(tool["name"], {}))
            if mapping and not set(mapping.values()) & (set(params) | extra):
                renames[tool["name"]] = mapping
        if not renames:
            return None
        return draft.with_view(self.name, rename_params=renames)


class Shuffle(Part):
    name = "shuffle"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        if draft.view.shuffle_seed is not None or len(draft.presented_names) < 2:
            return None
        return draft.with_view(self.name, shuffle_seed=rng.randrange(1 << 30))


class AddParams(Part):
    name = "add_params"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        extra = {k: dict(v) for k, v in draft.view.extra_params.items()}
        added = False
        for tool in draft.tools or []:
            name = tool["name"]
            renames = draft.view.rename_params.get(name, {})
            taken = {renames.get(p, p) for p in properties(tool)} | set(
                extra.get(name, {})
            )
            free = [p for p in EXTRA_PARAM_POOL if p not in taken]
            if free:
                param = rng.choice(free)
                extra.setdefault(name, {})[param] = EXTRA_PARAM_POOL[param]
                added = True
        if not added:
            return None
        return draft.with_view(self.name, extra_params=extra)


class DropOptionalParams(Part):
    name = "drop_optional_params"
    needs_tools = True
    needs_reference = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        used: dict[str, set[str]] = {}
        for call in draft.reference_calls or []:
            used.setdefault(call["tool_name"], set()).update(
                call.get("arguments") or {}
            )
        drop = {k: list(v) for k, v in draft.view.drop_params.items()}
        dropped = False
        for tool in draft.tools or []:
            name = tool["name"]
            optional = [
                p
                for p in properties(tool)
                if p not in required(tool)
                and p not in used.get(name, set())
                and p not in drop.get(name, [])
            ]
            if optional:
                drop.setdefault(name, []).extend(optional)
                dropped = True
        if not dropped:
            return None
        return draft.with_view(self.name, drop_params=drop)


class DistractorTools(Part):
    name = "distractor_tools"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> Draft | None:
        taken = set(draft.presented_names) | set(draft.tool_names)
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
        return draft.with_view(
            self.name, distractor_tools=[*draft.view.distractor_tools, *distractors]
        )


class ToolDocParaphrase(Part):
    name = "tool_doc_paraphrase"
    needs_tools = True

    def apply(self, draft: Draft, rng: random.Random) -> LLMStep | None:
        if draft.view.tool_descriptions:
            return None
        docs = {
            t["name"]: {
                "description": t.get("description", ""),
                "parameters": {
                    p: s.get("description", "") for p, s in properties(t).items()
                },
            }
            for t in draft.tools or []
        }

        def finish(response: dict) -> Draft | None:
            rewritten = response.get("tools")
            if not isinstance(rewritten, dict) or set(rewritten) != set(docs):
                return None
            tool_descriptions = {}
            param_descriptions = {}
            for name, doc in docs.items():
                new = rewritten[name]
                if not isinstance(new, dict) or not isinstance(
                    new.get("description"), str
                ):
                    return None
                new_params = new.get("parameters") or {}
                if not isinstance(new_params, dict) or set(new_params) != set(
                    doc["parameters"]
                ):
                    return None
                if not all(isinstance(v, str) for v in new_params.values()):
                    return None
                tool_descriptions[name] = new["description"]
                if new_params:
                    param_descriptions[name] = new_params
            return draft.with_view(
                self.name,
                tool_descriptions=tool_descriptions,
                param_descriptions=param_descriptions,
            )

        prompt = AUG_TOOL_DOC_PARAPHRASE_PROMPT.format(
            tools_json=json.dumps(docs, ensure_ascii=False)
        )
        return LLMStep(draft, [{"role": "user", "content": prompt}], finish)
