import ast
import asyncio
import inspect
import json
import types
from typing import Any

import httpx
import pytest
from fastapi.openapi import utils as openapi_utils

from awm.core import view_server
from awm.core.mcp_bridge.server import (
    API_BASE_URL,
    ToolCallError,
    build_tools,
    execute_operation,
)
from awm.core.tool_view import LAUNCHER_MARKERS, ToolView, bake_view

SERVER = """
from fastapi import Body, FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Shop")


class Order(BaseModel):
    product_id: int
    day: str
    note: str = ""


@app.post("/orders", summary="Create an order.", operation_id="create_order")
def create_order(order: Order = Body(...), urgent: bool = Query(False, description="Rush it.")):
    return {"order": order.model_dump(), "urgent": urgent}


@app.get("/days/{day}", summary="Look up a day.", operation_id="get_day")
def get_day(day: str):
    return {"day": day}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

DISTRACTOR: dict[str, Any] = {
    "name": "send_fax",
    "description": "Send a fax.",
    "inputSchema": {
        "type": "object",
        "properties": {"to": {"type": "string", "title": "to"}},
        "required": ["to"],
    },
}


class Loaded:
    """A server loaded in this process, with the MCP bridge's view of it."""

    def __init__(self, app) -> None:
        self.app = app
        tools, self.operations = build_tools(app)
        self.tools = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in tools
        ]

    def call(self, tool: str, arguments: dict) -> dict:
        async def run() -> str:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url=API_BASE_URL
            ) as client:
                return await execute_operation(client, tool, arguments, self.operations)

        return json.loads(asyncio.run(run()))


@pytest.fixture
def load(monkeypatch):
    """Loads server code in-process; the get_openapi it wraps is restored after."""
    original = openapi_utils.get_openapi

    def run(code: str) -> Loaded:
        monkeypatch.setattr(openapi_utils, "get_openapi", original)
        module = types.ModuleType("server")
        exec(compile(code, "server", "exec"), module.__dict__)  # noqa: S102
        return Loaded(module.app)

    return run


def full_view(base: list[dict]) -> ToolView:
    order = next(t for t in base if t["name"] == "create_order")
    _, responses = order["description"].split("\n\n", 1)
    return ToolView(
        tool_descriptions={"create_order": "Order a product.\n\n" + responses},
        param_descriptions={"create_order": {"note": "A note.", "urgent": "Hurry."}},
        extra_params={
            "create_order": {"trace_id": {"type": "string", "description": "Trace."}}
        },
        param_enums={"create_order": {"note": ["gift", "none"]}},
        param_formats={
            "create_order": {
                "product_id": "int_string",
                "day": "date_compact",
                "urgent": "bool_yes_no",
            },
            "get_day": {"day": "date_slash"},
        },
        distractor_tools=[DISTRACTOR],
    )


# ---------------------------------------------------------------- listing


def test_baked_server_lists_the_tools_as_the_view_presents_them(load):
    base = {t["name"]: t for t in load(SERVER).tools}
    view = full_view(list(base.values()))
    listed = {t["name"]: t for t in load(bake_view(SERVER, view)).tools}
    assert sorted(listed) == ["create_order", "get_day", "send_fax"]

    order = listed["create_order"]
    assert order["description"] == view.tool_descriptions["create_order"]
    props = order["inputSchema"]["properties"]
    assert props["note"]["description"] == "A note."
    assert props["note"]["enum"] == ["gift", "none"]
    assert props["urgent"]["description"] == 'Hurry. Write "yes" or "no".'
    assert props["urgent"]["enum"] == ["yes", "no"]
    assert props["product_id"]["type"] == "string"
    assert props["product_id"]["pattern"] == r"^-?\d+$"
    assert props["day"]["pattern"] == r"^\d{8}$"
    assert props["trace_id"]["description"] == "Trace."
    assert "trace_id" not in order["inputSchema"].get("required", [])
    # the presented tools keep what the view does not change
    assert listed["get_day"]["description"] == base["get_day"]["description"]
    assert (
        order["inputSchema"]["required"]
        == base["create_order"]["inputSchema"]["required"]
    )

    fax = listed["send_fax"]
    assert fax["description"] == DISTRACTOR["description"]
    assert fax["inputSchema"]["properties"] == DISTRACTOR["inputSchema"]["properties"]
    assert fax["inputSchema"]["required"] == ["to"]


def test_unbaked_server_lists_its_own_tools(load):
    base = load(SERVER).tools
    load(bake_view(SERVER, full_view(base)))
    assert load(SERVER).tools == base


# ------------------------------------------------------------------ calls


def test_baked_server_takes_presented_values_and_drops_extra_params(load):
    server = load(bake_view(SERVER, full_view(load(SERVER).tools)))
    out = server.call(
        "create_order",
        {
            "product_id": "42",
            "day": "20260520",
            "urgent": "yes",
            "note": "gift",
            "trace_id": "t-1",
        },
    )
    assert out == {
        "order": {"product_id": 42, "day": "2026-05-20", "note": "gift"},
        "urgent": True,
    }


def test_baked_server_takes_a_path_value_written_with_slashes(load):
    server = load(bake_view(SERVER, full_view(load(SERVER).tools)))
    assert server.call("get_day", {"day": "2026/05/20"}) == {"day": "2026-05-20"}


def test_baked_server_rejects_a_value_in_the_servers_own_format(load):
    server = load(bake_view(SERVER, full_view(load(SERVER).tools)))
    with pytest.raises(ToolCallError, match="Write the date as YYYYMMDD"):
        server.call("create_order", {"product_id": "42", "day": "2026-05-20"})


def test_calling_a_distractor_tool_fails(load):
    server = load(bake_view(SERVER, full_view(load(SERVER).tools)))
    with pytest.raises(ToolCallError, match="send_fax is not available"):
        server.call("send_fax", {"to": "03-1234"})


# --------------------------------------------------------------- bake_view


def test_bake_view_adds_one_name_before_the_main_guard():
    code = bake_view(SERVER, ToolView(distractor_tools=[DISTRACTOR]))
    before, guard, _ = code.partition('if __name__ == "__main__":')
    assert guard and before.startswith(SERVER.split("if __name__")[0])
    added = ast.parse(before[len(SERVER.split("if __name__")[0]) :])
    names = {n.name for n in added.body if isinstance(n, ast.FunctionDef)}
    assert names == {"_install_tool_view"}


def test_bake_view_appends_to_code_without_a_main_guard(load):
    code = SERVER.split("if __name__")[0]
    baked = bake_view(code, ToolView(distractor_tools=[DISTRACTOR]))
    assert baked.startswith(code.rstrip())
    assert "send_fax" in [t["name"] for t in load(baked).tools]


def test_view_texts_never_reach_lines_the_launchers_rewrite(load):
    text = "Like create_engine(...) or uvicorn.run(app)."
    tool = dict(DISTRACTOR, description=text)
    baked = bake_view(SERVER, ToolView(distractor_tools=[tool]))
    added = [line for line in baked.splitlines() if line not in SERVER.splitlines()]
    assert not [m for m in LAUNCHER_MARKERS for line in added if m in line]
    listed = {t["name"]: t for t in load(baked).tools}
    assert listed["send_fax"]["description"] == text


def test_view_server_runs_without_awm_on_python_3_10():
    source = inspect.getsource(view_server)
    tree = ast.parse(source, feature_version=(3, 10))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported <= {
        "collections",
        "copy",
        "dataclasses",
        "fastapi",
        "json",
        "re",
        "starlette",
        "typing",
        "urllib",
    }
