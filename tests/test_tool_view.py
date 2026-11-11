from typing import Any

import pytest
from pydantic import ValidationError

from awm.core.tool_view import (
    VALUE_FORMATS,
    PresentedTools,
    TaskMeta,
    ToolCallRejected,
    Turn,
    value_formats,
)
from awm.core.view_server import formatted_schema

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


def test_without_hidden_tools_every_tool_is_seen_and_called():
    presented = PresentedTools(TOOLS)
    assert presented.tools == TOOLS
    presented.check("create_order")


def test_a_hidden_tool_is_not_seen_until_it_is_shown():
    presented = PresentedTools(TOOLS, ["create_order"])
    assert names(presented.tools) == ["list_products"]
    with pytest.raises(ToolCallRejected, match="Unknown tool 'create_order'"):
        presented.check("create_order")
    presented.show(["create_order"])
    assert names(presented.tools) == ["create_order", "list_products"]
    presented.check("create_order")


def test_task_meta_lists_hidden_tools_and_the_turn_that_shows_them():
    meta = TaskMeta.model_validate(
        {
            "augmentations": ["miss_func"],
            "source_task_idx": 0,
            "hidden_tools": ["create_order"],
            "turns": [{"user": "Go on.", "shown_tools": ["create_order"]}],
        }
    )
    assert meta.turns == [Turn(user="Go on.", shown_tools=["create_order"])]


def test_task_meta_with_a_view_for_the_client_is_refused():
    with pytest.raises(ValidationError, match="run `awm gen aug` again"):
        TaskMeta.model_validate(
            {"augmentations": [], "source_task_idx": 0, "tool_view": None}
        )


# ------------------------------------------------------- values of parameters

DATED: list[dict[str, Any]] = [
    {
        "name": "book_room",
        "description": "Book a room.",
        "inputSchema": {
            "properties": {
                "day": {
                    "anyOf": [{"type": "string", "format": "date"}, {"type": "null"}],
                    "description": "Day of the stay.",
                    "default": None,
                },
                "nights": {
                    "type": "integer",
                    "description": "Nights.",
                    "minimum": 1,
                    "maximum": 30,
                    "example": 2,
                },
                "smoking": {"type": "boolean", "description": "Smoking room."},
                "plan": {"type": "string", "description": "Plan."},
            }
        },
    }
]


@pytest.mark.parametrize(
    ("fmt", "server", "agent"),
    [
        ("date_compact", "2026-05-20", "20260520"),
        ("date_slash", "2026-05-20", "2026/05/20"),
        ("bool_yes_no", True, "yes"),
        ("bool_yes_no", False, "no"),
        ("int_string", -3, "-3"),
    ],
)
def test_value_formats_convert_both_ways(fmt, server, agent):
    value_format = VALUE_FORMATS[fmt]
    assert value_format.to_agent(server) == agent
    assert value_format.to_server(agent) == server


def test_value_formats_fit_the_parameter_type_and_value():
    props = DATED[0]["inputSchema"]["properties"]
    assert value_formats(props["day"], "2026-05-20") == ["date_compact", "date_slash"]
    assert value_formats(props["nights"], 2) == ["int_string"]
    assert value_formats(props["smoking"], False) == ["bool_yes_no"]
    assert value_formats(props["plan"], "standard") == []
    assert value_formats(props["nights"], True) == []


def test_formatted_schema_writes_the_value_in_the_format():
    day = formatted_schema(DATED[0]["inputSchema"]["properties"]["day"], "date_compact")
    assert day == {
        "type": "string",
        "pattern": r"^\d{8}$",
        "description": "Day of the stay. Write the date as YYYYMMDD, e.g. 20260520.",
    }


def test_formatted_schema_drops_constraints_of_the_server_type():
    nights = formatted_schema(
        DATED[0]["inputSchema"]["properties"]["nights"], "int_string"
    )
    assert nights == {
        "type": "string",
        "pattern": r"^-?\d+$",
        "example": "2",
        "description": 'Nights. Write the number as a string of digits, e.g. "42".',
    }
