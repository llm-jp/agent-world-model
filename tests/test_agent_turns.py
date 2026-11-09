import asyncio
import json
from typing import ClassVar

from awm.core import agent
from awm.core.tool_view import TaskMeta, ToolView, Turn

TOOLS = [
    {"name": "create_order", "description": "Order.", "inputSchema": {}},
    {"name": "list_products", "description": "List.", "inputSchema": {}},
]


class FakeMCP:
    calls: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.timeout = timeout

    async def list_tools(self) -> list[dict]:
        return TOOLS

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        FakeMCP.calls.append((tool_name, arguments))
        return "ok"


def tool_call(name: str, arguments) -> tuple[str, list[dict]]:
    return "", [{"id": "c", "name": name, "arguments": arguments}]


def run_scripted(tmp_path, monkeypatch, meta: TaskMeta, replies: list) -> dict:
    tasks_path = tmp_path / "gen_tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "scenario": "shop",
                "tasks": ["order product 42", "order product 42"],
                "task_meta": [None, meta.model_dump()],
            }
        )
        + "\n"
    )
    script = iter(replies)

    async def fake_generate(*args, **kwargs):
        return next(script)

    FakeMCP.calls = []
    monkeypatch.setattr(agent, "MCPToolExecutor", FakeMCP)
    monkeypatch.setattr(agent, "generate_response", fake_generate)
    config = agent.Config(
        scenario="shop",
        task_id=1,
        mcp_url="http://unused/mcp",
        model="m",
        api_url="http://unused/v1",
        tasks_path=str(tasks_path),
        output_dir=str(tmp_path / "run"),
        verbose=False,
    )
    asyncio.run(agent.run_agent(config))
    return json.loads((tmp_path / "run" / "trajectory.json").read_text())


def test_miss_func_dialogue_hides_then_reveals_the_tool(tmp_path, monkeypatch):
    meta = TaskMeta(
        augmentations=["miss_func"],
        source_task_idx=0,
        tool_view=ToolView(hide_tools=["create_order"]),
        turns=[Turn(user="create_order is available now", tool_view=ToolView())],
    )
    order = {"tool_name": "create_order", "arguments": '{"product_id": 42}'}
    data = run_scripted(
        tmp_path,
        monkeypatch,
        meta,
        [
            tool_call("list_tools", None),
            tool_call("call_tool", order),
            ("I cannot place orders with the available tools.", []),
            tool_call("call_tool", order),
            ("Ordered product 42.", []),
        ],
    )

    assert FakeMCP.calls == [("create_order", {"product_id": 42})]
    assert [(c["turn"], c.get("rejected", False)) for c in data["tool_calls"]] == [
        (0, True),
        (1, False),
    ]
    users = [m["content"] for m in data["messages"] if m["role"] == "user"]
    assert users == ["order product 42", "create_order is available now"]
    assert data["task_meta"]["augmentations"] == ["miss_func"]
    # without a database file (external MCP server) no snapshot can be taken
    assert data["turn_dbs"] == []


def test_extra_params_are_dropped_before_the_server(tmp_path, monkeypatch):
    meta = TaskMeta(
        augmentations=["add_params"],
        source_task_idx=0,
        tool_view=ToolView(
            extra_params={"create_order": {"request_id": {"type": "string"}}}
        ),
    )
    run_scripted(
        tmp_path,
        monkeypatch,
        meta,
        [
            tool_call(
                "call_tool",
                {
                    "tool_name": "mcp_tool_create_order",
                    "arguments": '{"request_id": "r"}',
                },
            ),
            ("done", []),
        ],
    )
    assert FakeMCP.calls == [("create_order", {})]
