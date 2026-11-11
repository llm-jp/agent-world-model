import dataclasses
import sqlite3
import types
from typing import Any

from awm.core.augment.draft import Draft, Rejection
from awm.core.augment.part import PartContext, RunLLM
from awm.core.augment.replay import ReplayContext
from awm.core.mcp_bridge.server import build_tools

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
    """The result of the first candidate reply the step accepts, else None."""
    for reply in candidates:
        result = step.finish(reply)
        if not isinstance(result, Rejection):
            return result
    return None


def scripted_llm(reply_candidates) -> RunLLM:
    """An LLM runner whose replies are reply_candidates(step) tried in order."""

    def run(steps, workers=1):
        drafts = [first_accepted(step, reply_candidates(step)) for step in steps]
        return [d for d in drafts if d is not None]

    return run


def make_ctx(run_llm: RunLLM, num_per_part: int = 2, **changes: Any) -> PartContext:
    """A context for parts that do not replay server code, unless a replay is given."""
    changes.setdefault(
        "replay", ReplayContext("gen_db.jsonl", "gen_sample.jsonl", "aug_env_runs")
    )
    return PartContext(seed=0, num_per_part=num_per_part, run_llm=run_llm, **changes)


# a server that serves TOOLS, small enough to load in-process
SHOP_SERVER = """
from fastapi import FastAPI, Query

app = FastAPI(title="Shop")


@app.get("/products", summary="Search products.", operation_id="search_products")
def search_products(query: str = Query(..., description="Words."), limit: int = Query(10, description="Max.")):
    return {"ids": [42]}


@app.post("/orders", summary="Create an order.", operation_id="create_order")
def create_order(product_id: int = Query(..., description="Product."), note: str = Query("", description="Note.")):
    return {"product_id": product_id, "note": note}


@app.get("/coupons", summary="List coupons.", operation_id="list_coupons")
def list_coupons():
    return {"coupons": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""


def in_process_tools(code: str) -> list[dict]:
    """The tools the server code lists, loaded in this process.

    The server code of an augmented environment wraps get_openapi; the
    wrapper is dropped again afterwards.
    """
    from fastapi.openapi import utils

    original = utils.get_openapi
    try:
        module = types.ModuleType("server")
        exec(compile(code, "server", "exec"), module.__dict__)  # noqa: S102
        tools, _ = build_tools(module.app)
    finally:
        utils.get_openapi = original
    return [
        {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
        for t in tools
    ]
