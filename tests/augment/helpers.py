import dataclasses
import sqlite3
from typing import Any

from awm.core.augment.draft import Draft

# original verifier: complete when an order for product 42 exists
SOURCE_VERIFIER = """
def verify_task_completion(initial_db_path, final_db_path, final_answer=""):
    conn = sqlite3.connect(final_db_path)
    count = conn.execute("SELECT COUNT(*) FROM orders WHERE product_id = 42").fetchone()[0]
    conn.close()
    return {"result": "complete" if count else "others"}
"""

TOOLS = [
    {
        "name": "search_products",
        "description": "Search products.",
        "inputSchema": {
            "properties": {
                "query": {"type": "string", "description": "Words."},
                "limit": {"type": "integer", "description": "Max."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_order",
        "description": "Create an order.",
        "inputSchema": {
            "properties": {
                "product_id": {"type": "integer", "description": "Product."},
                "note": {"type": "string", "description": "Note."},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "list_coupons",
        "description": "List coupons.",
        "inputSchema": {"properties": {}},
    },
]

TASK = "Search for 'blue mug' and order product 42 with the note 'gift'"
CALLS: list[dict[str, Any]] = [
    {"tool_name": "search_products", "arguments": {"query": "blue mug"}},
    {"tool_name": "create_order", "arguments": {"product_id": 42, "note": "gift"}},
]
OTHER_TOOLS = [{"name": "send_fax", "description": "Fax.", "inputSchema": {}}]


BASE_DRAFT = Draft(
    scenario="Shop",
    source_task_idx=0,
    task=TASK,
    source_verifier_code=SOURCE_VERIFIER,
    tools=TOOLS,
    reference_calls=CALLS,
    other_tools=OTHER_TOOLS,
)


def make_draft(**changes: Any) -> Draft:
    return dataclasses.replace(BASE_DRAFT, **changes)


def make_db(path, orders: list[int]) -> str:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER)")
    conn.executemany(
        "INSERT INTO orders (product_id) VALUES (?)", [(o,) for o in orders]
    )
    conn.commit()
    conn.close()
    return str(path)


def first_accepted(step, candidates: list[dict]):
    """The draft from the first candidate reply the step accepts, else None."""
    for reply in candidates:
        draft = step.finish(reply)
        if draft is not None:
            return draft
    return None


def scripted_llm(reply_candidates):
    """An LLM runner whose replies are reply_candidates(step) tried in order."""

    def run(steps):
        drafts = [first_accepted(step, reply_candidates(step)) for step in steps]
        return [d for d in drafts if d is not None]

    return run
