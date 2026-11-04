"""Change how MCP tools are presented to the agent without touching the server.

A ToolView rewrites the `list_tools` result (names, descriptions, parameters,
extra and hidden tools, order) and maps the agent's calls back to the tool
names and arguments the server knows. All keys of a ToolView refer to the
original tool and parameter names.
"""

import copy
import random

from pydantic import BaseModel, ConfigDict, Field


class ToolView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hide_tools: list[str] = Field(default_factory=list)
    rename_tools: dict[str, str] = Field(default_factory=dict)
    # tool -> {param -> presented param}
    rename_params: dict[str, dict[str, str]] = Field(default_factory=dict)
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    # tool -> {param -> presented description}
    param_descriptions: dict[str, dict[str, str]] = Field(default_factory=dict)
    # tool -> {presented param -> JSON schema}; always optional, dropped on call
    extra_params: dict[str, dict[str, dict]] = Field(default_factory=dict)
    # tool -> params removed from the presented schema
    drop_params: dict[str, list[str]] = Field(default_factory=dict)
    # tools in presented form (name, description, inputSchema); calls fail
    distractor_tools: list[dict] = Field(default_factory=list)
    shuffle_seed: int | None = None


class ToolCallRejected(Exception):
    """The agent called a tool that the presented view does not offer."""


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # user message appended when the agent ends its turn with a text answer
    user: str
    # replaces the current view from this turn on; None keeps the current one
    tool_view: ToolView | None = None


class TaskMeta(BaseModel):
    """Per-task settings of an augmented task, stored in gen_tasks.jsonl."""

    model_config = ConfigDict(extra="forbid")

    # augmentation parts applied to the source task, in order
    augmentations: list[str]
    source_task_idx: int
    tool_view: ToolView | None = None
    turns: list[Turn] = Field(default_factory=list)


def _schema_of(tool: dict) -> dict:
    return tool.get("inputSchema") or tool.get("input_schema") or {}


class PresentedTools:
    """The tools the agent sees under a view, and the way back to the server."""

    def __init__(self, tools: list[dict], view: ToolView | None):
        self.view = view
        self.original_names = [t["name"] for t in tools]
        self.tools = tools if view is None else self._present(tools, view)
        self._to_original: dict[str, str] = {}
        # presented names of the server's own parameters, per original tool
        self._real_params: dict[str, set[str]] = {}
        if view is not None:
            for tool in tools:
                name = tool["name"]
                if name in view.hide_tools:
                    continue
                self._to_original[view.rename_tools.get(name, name)] = name
                renames = view.rename_params.get(name, {})
                self._real_params[name] = {
                    renames.get(p, p) for p in _schema_of(tool).get("properties", {})
                }

    @staticmethod
    def _present(tools: list[dict], view: ToolView) -> list[dict]:
        presented = []
        for tool in tools:
            name = tool["name"]
            if name in view.hide_tools:
                continue
            schema = copy.deepcopy(_schema_of(tool))
            properties: dict = schema.get("properties", {})
            required: list[str] = list(schema.get("required", []))

            for param in view.drop_params.get(name, []):
                properties.pop(param, None)
                if param in required:
                    required.remove(param)
            for param, desc in view.param_descriptions.get(name, {}).items():
                if param in properties:
                    properties[param]["description"] = desc
            renames = view.rename_params.get(name, {})
            properties = {renames.get(p, p): s for p, s in properties.items()}
            required = [renames.get(p, p) for p in required]
            for param, param_schema in view.extra_params.get(name, {}).items():
                properties.setdefault(param, copy.deepcopy(param_schema))

            if view.shuffle_seed is not None:
                items = list(properties.items())
                random.Random(f"{view.shuffle_seed}:{name}").shuffle(items)
                properties = dict(items)

            if properties or "properties" in schema:
                schema["properties"] = properties
            if required or "required" in schema:
                schema["required"] = required
            presented.append(
                {
                    "name": view.rename_tools.get(name, name),
                    "description": view.tool_descriptions.get(
                        name, tool.get("description", "")
                    ),
                    "inputSchema": schema,
                }
            )

        presented.extend(copy.deepcopy(view.distractor_tools))
        if view.shuffle_seed is not None:
            random.Random(view.shuffle_seed).shuffle(presented)
        return presented

    def resolve(self, tool_name: str, arguments: dict) -> tuple[str, dict]:
        """Map a call on a presented tool to the server's tool name and arguments."""
        view = self.view
        if view is None:
            return tool_name, arguments

        original = self._to_original.get(tool_name)
        if original is None:
            raise ToolCallRejected(
                f"Unknown tool '{tool_name}'. Call list_tools to see the available tools."
            )

        back = {p: o for o, p in view.rename_params.get(original, {}).items()}
        extra = set(view.extra_params.get(original, {})) - self._real_params[original]
        resolved = {back.get(k, k): v for k, v in arguments.items() if k not in extra}
        return original, resolved
