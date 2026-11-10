import json
import random

import pytest
from helpers import SOURCE_VERIFIER, TASK, make_draft

from awm.core.augment.draft import Draft, LLMStep, Rejection, Variant
from awm.core.augment.env_parts import (
    CheckTask,
    EnvVariant,
    RebuiltEnv,
    Renames,
    ReplayContext,
    ScenarioSource,
    ToolRename,
    ToolRenameRandom,
    apply_edits,
    code_step,
    env_scenario,
    listing_problems,
    merge_env_variants,
    rename_operation_ids,
    rename_problems,
    replay,
    task_step,
)
from awm.core.augment.refine import refine
from awm.core.tool_view import TaskMeta

# a server like the generated ones: one route per tool, the database via create_engine
SERVER = """
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///placeholder.db", connect_args={"check_same_thread": False})
app = FastAPI(title="Shop")


@app.post("/orders", summary="Create an order. Use search_products to find the id first.", operation_id="create_order")
def create_order(product_id: int = Query(..., description="Product.")):
    if product_id <= 0:
        raise HTTPException(status_code=400, detail="create_order needs a positive product_id")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO orders (product_id) VALUES (:p)"), {"p": product_id})
    return {"ok": True}


@app.get("/products", summary="Search products.", operation_id="search_products")
def search_products(query: str = Query(..., description="Words.")):
    return {"ids": [42]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

SERVER_TOOLS = [
    {
        "name": "create_order",
        "description": "Create an order. Use search_products to find the id first.",
        "inputSchema": {"properties": {"product_id": {"type": "integer"}}},
    },
    {
        "name": "search_products",
        "description": "Search products.",
        "inputSchema": {"properties": {"query": {"type": "string"}}},
    },
]

SERVER_CALLS = [
    {"tool_name": "search_products", "arguments": {"query": "mug"}},
    {"tool_name": "create_order", "arguments": {"product_id": 0}},
    {"tool_name": "create_order", "arguments": {"product_id": 42}},
]

RENAMES = Renames(
    tools={"create_order": "placeOrder", "search_products": "findItems"},
    params={"create_order": {"product_id": "productId"}},
)

# the edits that rename SERVER after rename_operation_ids
GOOD_EDITS = [
    {"old": "Use search_products to find", "new": "Use findItems to find"},
    {
        "old": 'product_id: int = Query(..., description="Product.")',
        "new": 'product_id: int = Query(..., alias="productId", description="Product.")',
    },
    {
        "old": '"create_order needs a positive product_id"',
        "new": '"placeOrder needs a positive productId"',
    },
]


def make_source(**changes) -> ScenarioSource:
    draft = make_draft(
        tools=SERVER_TOOLS,
        reference_calls=SERVER_CALLS,
        source_verifier_code=SOURCE_VERIFIER,
    )
    fields = {
        "scenario": "Shop",
        "env_row": {"scenario": "Shop", "full_code": SERVER},
        "tools": SERVER_TOOLS,
        "drafts": [draft],
        **changes,
    }
    return ScenarioSource(**fields)


@pytest.fixture
def ctx(tmp_path) -> ReplayContext:
    db = tmp_path / "gen_db.jsonl"
    sample = tmp_path / "gen_sample.jsonl"
    ddl = "CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER)"
    db.write_text(
        json.dumps(
            {
                "scenario": "Shop",
                "db_schema": {"tables": [{"name": "orders", "ddl": ddl}]},
            }
        )
    )
    sample.write_text(json.dumps({"scenario": "Shop", "sample_data": {"tables": []}}))
    return ReplayContext(str(db), str(sample), str(tmp_path / "runs"))


# --------------------------------------------------------------- names


def test_renames_rename_calls_and_find_mentions():
    assert RENAMES.call(SERVER_CALLS[2]) == {
        "tool_name": "placeOrder",
        "arguments": {"productId": 42},
    }
    text = "Call Create_Order with product_id, not create_orders or productid"
    assert RENAMES.mentioned_in(text) == {
        "create_order": "placeOrder",
        "product_id": "productId",
    }


def test_rename_problems():
    names = ["create_order", "search_products"]
    assert rename_problems(names, {"create_order": "placeOrder"}) == []
    for bad in ["CreateOrder", "place order", "search_products"]:
        assert rename_problems(names, {"create_order": bad})
    assert rename_problems(names, {"create_order": "x", "search_products": "x"})


def test_tool_rename_asks_the_llm_and_checks_the_names():
    step = ToolRename().renames(make_source(), random.Random(0))
    assert isinstance(step, LLMStep)
    names = {"create_order": "placeOrder", "search_products": "findItems"}
    assert step.finish({"tools": names}) == Renames(tools=names)
    assert isinstance(step.finish({"tools": {"create_order": "placeOrder"}}), Rejection)


def test_tool_rename_random_is_distinct_letters_and_deterministic():
    first = ToolRenameRandom().renames(make_source(), random.Random(3))
    again = ToolRenameRandom().renames(make_source(), random.Random(3))
    assert first == again
    names = list(first.tools.values())
    assert len(set(names)) == 2
    assert all(n.isalpha() and n.islower() and len(n) == 8 for n in names)


# ---------------------------------------------------------- code edits


def test_rename_operation_ids_only_touches_operation_id_literals():
    code = rename_operation_ids(SERVER, RENAMES)
    assert 'operation_id="placeOrder"' in code
    assert 'operation_id="findItems"' in code
    assert "def create_order(" in code
    assert "Use search_products" in code


def test_apply_edits_needs_each_old_text_exactly_once():
    assert apply_edits("a b a", [{"old": "b", "new": "c"}]) == "a c a"
    assert isinstance(apply_edits("a b a", [{"old": "a", "new": "c"}]), Rejection)
    assert isinstance(apply_edits("a", [{"old": "x", "new": "y"}]), Rejection)
    assert isinstance(apply_edits("a", []), Rejection)


def test_listing_problems_compare_with_the_renamed_tools():
    renamed = [
        {
            "name": "placeOrder",
            "description": "Create an order. Use findItems first.",
            "inputSchema": {"properties": {"productId": {}}},
        },
        {
            "name": "findItems",
            "description": "Search.",
            "inputSchema": {"properties": {"query": {}}},
        },
    ]
    assert listing_problems(make_source(), RENAMES, renamed) == []
    stale = [dict(renamed[0], description="Use search_products first."), renamed[1]]
    assert listing_problems(make_source(), RENAMES, stale)
    missing = [renamed[0]]
    assert listing_problems(make_source(), RENAMES, missing)


# --------------------------------------------- servers (start processes)


def test_replay_grades_the_database_and_reports_failed_calls(ctx):
    result = replay(ctx, SERVER, "Shop", SERVER_CALLS, SOURCE_VERIFIER, "")
    assert result.reward == "complete"
    assert [e is not None for e in result.errors] == [False, True, False]
    assert {t["name"] for t in result.tools} == {"create_order", "search_products"}


def check_task() -> CheckTask:
    return CheckTask(make_source().drafts[0], [False, True, False])


def test_code_step_accepts_a_server_that_behaves_the_same(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES, [check_task()])
    rebuilt = step.finish({"edits": GOOD_EDITS})
    assert not isinstance(rebuilt, Rejection), rebuilt
    assert {t["name"] for t in rebuilt.tools} == {"placeOrder", "findItems"}
    assert step.check_prompt is not None
    # the self-check sees the edits as a diff
    assert '+def create_order(product_id: int = Query(..., alias="productId"' in (
        step.check_prompt(rebuilt)
    )


def test_code_step_rejects_stale_names_and_changed_parameters(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES, [check_task()])
    stale = step.finish({"edits": GOOD_EDITS[1:]})
    assert isinstance(stale, Rejection) and "search_products" in stale.reason
    not_aliased = step.finish({"edits": [GOOD_EDITS[0], GOOD_EDITS[2]]})
    assert isinstance(not_aliased, Rejection) and "productId" in not_aliased.reason


def test_code_step_reports_a_server_that_does_not_start(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES, [check_task()])
    broken = [*GOOD_EDITS, {"old": "app = FastAPI(", "new": "app = missing_name("}]
    rejection = step.finish({"edits": broken})
    assert isinstance(rejection, Rejection)
    assert "does not start" in rejection.reason and "missing_name" in rejection.reason


def test_refine_feeds_server_problems_back(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES, [check_task()])
    replies = iter(
        [
            json.dumps({"edits": GOOD_EDITS[1:]}),
            json.dumps({"edits": GOOD_EDITS}),
            json.dumps({"consistent": True, "problems": []}),
        ]
    )
    seen: list[list[dict]] = []

    def complete(conversations):
        seen.extend(conversations)
        return [next(replies) for _ in conversations]

    [rebuilt] = refine([step], complete, max_generations=3)
    assert rebuilt.code.count("productId") >= 2
    assert "search_products" in seen[1][-1]["content"]


# ---------------------------------------------------------------- tasks


def rebuilt_env(task: str) -> RebuiltEnv:
    source = make_source(drafts=[make_draft(task=task, reference_calls=SERVER_CALLS)])
    return RebuiltEnv(source, "tool_rename", RENAMES, SERVER, [], [check_task()])


def test_task_without_old_names_is_kept_as_is():
    rebuilt = rebuilt_env(TASK)
    draft = task_step(rebuilt, CheckTask(rebuilt.source.drafts[0], [False] * 3))
    assert isinstance(draft, Draft)
    assert draft.scenario == env_scenario("Shop", "tool_rename") == "Shop__tool_rename"
    assert draft.reference_calls is not None
    assert draft.reference_calls[2]["tool_name"] == "placeOrder"


def test_task_naming_an_old_tool_is_rewritten_and_checked():
    task = "Use create_order to order product 42"
    rebuilt = rebuilt_env(task)
    step = task_step(rebuilt, CheckTask(rebuilt.source.drafts[0], [False] * 3))
    assert isinstance(step, LLMStep) and step.check_prompt is not None
    stale = step.finish({"task": "Use create order to order product 42"})
    assert isinstance(stale, Rejection)
    out = step.finish({"task": "Use placeOrder to order product 42"})
    assert isinstance(out, Draft) and out.task == "Use placeOrder to order product 42"
    assert "placeOrder" in step.check_prompt(out)


# ------------------------------------------------------ dataset assembly


def env_variant(task: str) -> EnvVariant:
    meta = TaskMeta(
        augmentations=["tool_rename"], source_task_idx=3, source_scenario="Shop"
    )
    variant = Variant("Shop__tool_rename", task, meta, "code")
    return EnvVariant("Shop__tool_rename", "Shop", "tool_rename", "new code", [variant])


def scenario_rows() -> dict[str, list[dict]]:
    return {
        "envs": [{"scenario": "Shop", "full_code": "old code", "api_spec": {}}],
        "db": [{"scenario": "Shop", "db_schema": {}}],
        "sample": [{"scenario": "Shop", "sample_data": {}}],
        "tasks": [{"scenario": "Shop", "tasks": ["t0"]}],
    }


def test_merge_adds_the_environment_to_every_scenario_file():
    rows, verifiers = merge_env_variants(scenario_rows(), [], [env_variant("a")], set())
    env = rows["envs"][-1]
    assert env["scenario"] == "Shop__tool_rename" and env["full_code"] == "new code"
    assert env["api_spec"] == {} and env["source_scenario"] == "Shop"
    assert rows["db"][-1]["db_schema"] == {} and rows["sample"][-1]["sample_data"] == {}
    tasks = rows["tasks"][-1]
    assert tasks["tasks"] == ["a"]
    assert tasks["task_meta"][0]["source_scenario"] == "Shop"
    assert verifiers == [
        {
            "scenario": "Shop__tool_rename",
            "task_idx": 0,
            "task": "a",
            "verification": {"code": "code"},
            "augmented": True,
            "augmentations": ["tool_rename"],
            "source_task_idx": 3,
            "source_scenario": "Shop",
        }
    ]


def test_merge_replaces_earlier_environments_unless_kept():
    once = merge_env_variants(scenario_rows(), [], [env_variant("a")], set())
    again = merge_env_variants(once[0], once[1], [env_variant("b")], set())
    assert [r["tasks"] for r in again[0]["tasks"]] == [["t0"], ["b"]]
    assert len(again[0]["envs"]) == 2 and len(again[1]) == 1
    kept = merge_env_variants(once[0], once[1], [], {"shop"})
    assert kept == once
