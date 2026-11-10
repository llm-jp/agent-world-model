import pytest

from awm.core.tool_view import PresentedTools, ToolCallRejected, ToolView

TOOLS = [
    {
        "name": "create_order",
        "description": "Create an order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "Product."},
                "note": {"type": "string", "description": "Note."},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "list_products",
        "description": "List products.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def names(tools: list[dict]) -> list[str]:
    return [t["name"] for t in tools]


def test_without_view_tools_and_calls_pass_through():
    presented = PresentedTools(TOOLS, None)
    assert presented.tools == TOOLS
    assert presented.resolve("anything", {"a": 1}) == ("anything", {"a": 1})


def test_unknown_tool_is_rejected():
    presented = PresentedTools(TOOLS, ToolView())
    with pytest.raises(ToolCallRejected):
        presented.resolve("createOrder", {})


def test_hidden_and_distractor_tools_are_rejected():
    distractor = {"name": "send_fax", "description": "Fax.", "inputSchema": {}}
    view = ToolView(hide_tools=["create_order"], distractor_tools=[distractor])
    presented = PresentedTools(TOOLS, view)
    assert names(presented.tools) == ["list_products", "send_fax"]
    for name in ["create_order", "send_fax"]:
        with pytest.raises(ToolCallRejected):
            presented.resolve(name, {})


def test_extra_params_are_dropped_but_real_params_kept():
    view = ToolView(
        extra_params={
            "create_order": {
                "request_id": {"type": "string"},
                # same name as a real parameter: the server's parameter wins
                "note": {"type": "integer"},
            }
        }
    )
    presented = PresentedTools(TOOLS, view)
    props = presented.tools[0]["inputSchema"]["properties"]
    assert props["note"]["type"] == "string"
    assert "request_id" not in presented.tools[0]["inputSchema"]["required"]
    assert presented.resolve(
        "create_order", {"product_id": 1, "note": "n", "request_id": "r"}
    ) == ("create_order", {"product_id": 1, "note": "n"})


def test_descriptions():
    view = ToolView(
        tool_descriptions={"create_order": "Place an order."},
        param_descriptions={"create_order": {"product_id": "ID of the product."}},
    )
    order = PresentedTools(TOOLS, view).tools[0]
    assert order["description"] == "Place an order."
    assert list(order["inputSchema"]["properties"]) == ["product_id", "note"]
    assert (
        order["inputSchema"]["properties"]["product_id"]["description"]
        == "ID of the product."
    )


def test_view_does_not_modify_the_original_tools():
    before = repr(TOOLS)
    PresentedTools(
        TOOLS,
        ToolView(
            param_descriptions={"create_order": {"note": "Changed."}},
            extra_params={"create_order": {"request_id": {"type": "string"}}},
        ),
    )
    assert repr(TOOLS) == before
