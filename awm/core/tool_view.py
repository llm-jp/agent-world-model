"""How an augmented task presents the MCP tools to the agent.

A ToolView changes the `list_tools` result (descriptions, extra parameters,
allowed values and value formats of parameters, distractor tools). An
augmented environment serves its view from the server itself (bake_view,
awm/core/view_server.py), so every MCP client sees it, an RL trainer included.
Renamed tools are not a view: their server code is rewritten
(awm/core/augment/env_parts.py).

What changes between the turns of a task cannot be served that way: a task's
meta lists the tools hidden until a follow-up turn, and `awm agent` hides
them (PresentedTools).
"""

import inspect
import json
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from awm.core import view_server
from awm.core.view_server import FORMAT_ERRORS, VALUE_FORMATS


class ToolView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    # tool -> {param -> presented description}
    param_descriptions: dict[str, dict[str, str]] = Field(default_factory=dict)
    # tool -> {presented param -> JSON schema}; always optional, ignored on call
    extra_params: dict[str, dict[str, dict]] = Field(default_factory=dict)
    # tools in presented form (name, description, inputSchema); calls fail
    distractor_tools: list[dict] = Field(default_factory=list)
    # tool -> {param -> allowed values}, shown as the parameter's enum
    param_enums: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    # tool -> {param -> name of a VALUE_FORMATS entry}; calls are converted back
    param_formats: dict[str, dict[str, str]] = Field(default_factory=dict)


class ToolCallRejected(Exception):
    """The agent called a tool that is hidden in the current turn."""


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # user message appended when the agent ends its turn with a text answer
    user: str
    # hidden tools the agent can see and call from this turn on
    shown_tools: list[str] = Field(default_factory=list)


class TaskMeta(BaseModel):
    """Per-task settings of an augmented task, stored in gen_tasks.jsonl."""

    model_config = ConfigDict(extra="forbid")

    # augmentation parts applied to the source task, in order
    augmentations: list[str]
    source_task_idx: int
    # scenario of the source task; augmented tasks live in environments of their own
    source_scenario: str | None = None
    # tools the server lists but the agent does not see until a turn shows them
    hidden_tools: list[str] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _served_view(cls, data: Any) -> Any:
        if isinstance(data, dict) and "tool_view" in data:
            raise ValueError(
                "task_meta.tool_view is no longer read: the environment's server "
                "presents the tools; run `awm gen aug` again"
            )
        return data


def schema_types(schema: dict) -> set[str]:
    """JSON schema types a parameter takes, apart from null."""
    found = schema.get("type")
    types = {found} if isinstance(found, str) else set(found or [])
    for alternative in schema.get("anyOf", []) + schema.get("oneOf", []):
        types |= schema_types(alternative)
    return types - {"null"}


def value_formats(schema: dict, value: Any) -> list[str]:
    """Formats that can present the parameter and write this value of it."""
    found = []
    for name, value_format in VALUE_FORMATS.items():
        if value_format.server_type not in schema_types(schema):
            continue
        try:
            if value_format.to_server(value_format.to_agent(value)) == value:
                found.append(name)
        except FORMAT_ERRORS:
            continue
    return found


class PresentedTools:
    """The tools the agent sees in the current turn: the server's, but the hidden ones."""

    def __init__(self, tools: list[dict], hidden: Iterable[str] = ()):
        self._all = tools
        self.hidden = set(hidden)

    @property
    def tools(self) -> list[dict]:
        return [t for t in self._all if t["name"] not in self.hidden]

    def show(self, names: Iterable[str]) -> None:
        self.hidden -= set(names)

    def check(self, tool_name: str) -> None:
        """Raises ToolCallRejected if the tool is hidden in the current turn."""
        if tool_name in self.hidden:
            raise ToolCallRejected(
                f"Unknown tool '{tool_name}'. Call list_tools to see the available tools."
            )


# the block bake_view adds to the server code; it defines a single name
BAKED_VIEW_TEMPLATE = """

def _install_tool_view():
    import json
    import sys
    import types

    module = types.ModuleType("awm_view_server")
    sys.modules[module.__name__] = module
    exec(compile({source!r}, module.__name__, "exec"), module.__dict__)
    module.install(app, json.loads({view!r}))


_install_tool_view()

"""
MAIN_GUARD = re.compile(r"""^if __name__ == ["']__main__["']\s*:""", re.MULTILINE)
# lines the server launchers rewrite wherever these texts appear in them
LAUNCHER_MARKERS = ("create_engine(", "uvicorn.run(app")


def bake_view(code: str, view: ToolView) -> str:
    """Server code that presents its tools as the view says, to any MCP client.

    The block goes before the `if __name__ == "__main__":` guard, where the
    launchers mount the MCP layer, so the app has every route by then.
    """
    source = inspect.getsource(view_server)
    # JSON allows any character to be escaped, so no view text reaches the code as written
    view_json = json.dumps(view.model_dump()).replace("(", "\\u0028")
    block = BAKED_VIEW_TEMPLATE.format(source=source, view=view_json)
    if found := [m for m in LAUNCHER_MARKERS if m in block]:
        raise ValueError(f"the view block contains {found}, which launchers rewrite")
    guard = MAIN_GUARD.search(code)
    if guard is None:
        return code.rstrip() + "\n" + block
    return code[: guard.start()] + block.lstrip("\n") + code[guard.start() :]
