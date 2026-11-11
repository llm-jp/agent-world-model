import json
import random
from pathlib import Path

import pytest
from helpers import SOURCE_VERIFIER, TASK, make_ctx, make_draft, scripted_llm

from awm.core.augment.draft import Draft, LLMStep, Rejection
from awm.core.augment.env_parts import (
    MAX_RANDOM_NAME_LENGTH,
    MIN_RANDOM_NAME_LENGTH,
    ParamRename,
    RebuiltEnv,
    Renames,
    ScenarioSource,
    ToolRename,
    ToolRenameRandom,
    apply_edits,
    called_params,
    code_step,
    listing_problems,
    rename_operation_ids,
    rename_problems,
    task_step,
    unrelated_change_problems,
)
from awm.core.augment.refine import refine
from awm.core.augment.replay import ReplayContext, replay, replayed_drafts

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
        call_errors=(False, True, False),
    )
    fields = {
        "scenario": "Shop",
        "code": SERVER,
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
    assert all(n.isalpha() and n.islower() for n in names)
    assert all(
        MIN_RANDOM_NAME_LENGTH <= len(n) <= MAX_RANDOM_NAME_LENGTH for n in names
    )


def test_tool_rename_random_varies_the_length():
    tools = [{"name": f"tool_{i}", "inputSchema": {}} for i in range(40)]
    renames = ToolRenameRandom().renames(make_source(tools=tools), random.Random(0))
    assert len({len(n) for n in renames.tools.values()}) > 1


def test_tool_rename_random_avoids_words_of_the_code(monkeypatch):
    # the first draw is a word of the server code, so it is drawn again
    draws = iter([list("text"), list("zzzzq")])
    monkeypatch.setattr(random.Random, "choices", lambda self, *a, **k: next(draws))
    tools = [{"name": "only_tool", "inputSchema": {}}]
    renames = ToolRenameRandom().renames(make_source(tools=tools), random.Random(0))
    assert renames.tools == {"only_tool": "zzzzq"}


def test_called_params_are_the_parameters_the_reference_calls_pass():
    assert called_params(make_source()) == {
        "search_products": ["query"],
        "create_order": ["product_id"],
    }


def test_param_rename_maps_each_name_once_for_every_tool():
    step = ParamRename().renames(make_source(), random.Random(0))
    assert isinstance(step, LLMStep)
    assert '"product_id"' in step.messages[0]["content"]
    out = step.finish({"params": {"query": "keywords", "product_id": "itemId"}})
    assert out == Renames(
        params={
            "search_products": {"query": "keywords"},
            "create_order": {"product_id": "itemId"},
        }
    )


def test_param_rename_rejects_unusable_names():
    tools = [
        dict(
            SERVER_TOOLS[0], inputSchema={"properties": {"product_id": {}, "note": {}}}
        ),
        SERVER_TOOLS[1],
    ]
    step = ParamRename().renames(make_source(tools=tools), random.Random(0))
    assert isinstance(step, LLMStep)
    for names in [
        {"query": "keywords"},
        {"query": "Query", "product_id": "itemId"},
        {"query": "class", "product_id": "itemId"},
        {"query": "itemId", "product_id": "itemId"},
        {"query": "keywords", "product_id": "note"},
    ]:
        assert isinstance(step.finish({"params": names}), Rejection), names


def test_param_rename_needs_a_parameter_the_reference_calls_pass():
    source = make_source(drafts=[make_draft(tools=SERVER_TOOLS, reference_calls=[])])
    assert ParamRename().renames(source, random.Random(0)) is None


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


def renamed_server(*extra_edits: dict) -> str:
    code = apply_edits(
        rename_operation_ids(SERVER, RENAMES), [*GOOD_EDITS, *extra_edits]
    )
    assert isinstance(code, str), code
    return code


def test_unrelated_change_problems_accepts_edits_of_names_only():
    assert unrelated_change_problems(SERVER, renamed_server(), RENAMES) == []
    # a name written with spaces may change, or stay, and so may function names
    spaced = {"old": 'summary="Search products."', "new": 'summary="findItems."'}
    function = {"old": "def search_products(", "new": "def find_items("}
    assert (
        unrelated_change_problems(SERVER, renamed_server(spaced, function), RENAMES)
        == []
    )


def test_unrelated_change_problems_reports_reworded_text():
    reworded = {"old": 'description="Product."', "new": 'description="Item."'}
    problems = unrelated_change_problems(SERVER, renamed_server(reworded), RENAMES)
    assert len(problems) == 1
    assert "Product." in problems[0] and "Item." in problems[0]
    line = SERVER.splitlines().index(
        next(x for x in SERVER.splitlines() if 'description="Product."' in x)
    )
    assert f"line {line + 1}:" in problems[0]


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
    # the server's code and databases are removed after the replay
    assert list(Path(ctx.work_dir).iterdir()) == []


def test_code_step_accepts_a_server_that_behaves_the_same(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES)
    rebuilt = step.finish({"edits": GOOD_EDITS})
    assert not isinstance(rebuilt, Rejection), rebuilt
    assert {t["name"] for t in rebuilt.tools} == {"placeOrder", "findItems"}
    assert step.check_prompt is not None
    # the self-check sees the edits as a diff
    assert '+def create_order(product_id: int = Query(..., alias="productId"' in (
        step.check_prompt(rebuilt)
    )


def test_code_step_rejects_stale_names_and_changed_parameters(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES)
    stale = step.finish({"edits": GOOD_EDITS[1:]})
    assert isinstance(stale, Rejection) and "search_products" in stale.reason
    not_aliased = step.finish({"edits": [GOOD_EDITS[0], GOOD_EDITS[2]]})
    assert isinstance(not_aliased, Rejection) and "productId" in not_aliased.reason


def test_code_step_renames_parameters_only(ctx):
    renames = Renames(params={"create_order": {"product_id": "productId"}})
    step = code_step(ctx, make_source(), "param_rename", renames)
    edits = [
        GOOD_EDITS[1],
        {
            "old": '"create_order needs a positive product_id"',
            "new": '"create_order needs a positive productId"',
        },
    ]
    rebuilt = step.finish({"edits": edits})
    assert not isinstance(rebuilt, Rejection), rebuilt
    tool = next(t for t in rebuilt.tools if t["name"] == "create_order")
    assert set(tool["inputSchema"]["properties"]) == {"productId"}


def test_unrelated_change_problems_drops_an_alias_given_alone():
    renames = Renames(params={"t": {"q": "words"}})
    before = "def t(q: str = Query()):\n    return q\n"
    after = 'def t(q: str = Query(alias="words")):\n    return q\n'
    assert unrelated_change_problems(before, after, renames) == []


def test_code_step_rejects_text_reworded_beyond_the_names(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES)
    reworded = [*GOOD_EDITS, {"old": "Search products.", "new": "Look up items."}]
    rejection = step.finish({"edits": reworded})
    assert isinstance(rejection, Rejection)
    assert "not renames" in rejection.reason and "Look up items." in rejection.reason


def test_code_step_reports_a_server_that_does_not_start(ctx):
    broken = SERVER.replace("app = FastAPI(", "app = missing_name(")
    source = make_source(code=broken)
    step = code_step(ctx, source, "tool_rename", RENAMES)
    rejection = step.finish({"edits": GOOD_EDITS})
    assert isinstance(rejection, Rejection)
    assert "does not start" in rejection.reason and "missing_name" in rejection.reason


def test_refine_feeds_server_problems_back(ctx):
    step = code_step(ctx, make_source(), "tool_rename", RENAMES)
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
    return RebuiltEnv(source, "tool_rename", RENAMES, SERVER, [])


def test_task_without_old_names_is_kept_as_is():
    rebuilt = rebuilt_env(TASK)
    draft = task_step(rebuilt, rebuilt.source.drafts[0])
    assert isinstance(draft, Draft)
    assert draft.scenario == "Shop" and draft.parts == ("tool_rename",)
    assert draft.code == SERVER and draft.task == TASK
    assert draft.reference_calls is not None
    assert draft.reference_calls[2]["tool_name"] == "placeOrder"


def test_task_naming_an_old_tool_is_rewritten_and_checked():
    task = "Use create_order to order product 42"
    rebuilt = rebuilt_env(task)
    step = task_step(rebuilt, rebuilt.source.drafts[0])
    assert isinstance(step, LLMStep) and step.check_prompt is not None
    stale = step.finish({"task": "Use create order to order product 42"})
    assert isinstance(stale, Rejection)
    out = step.finish({"task": "Use placeOrder to order product 42"})
    assert isinstance(out, Draft) and out.task == "Use placeOrder to order product 42"
    assert "placeOrder" in step.check_prompt(out)
    assert out.banned_mentions == ["create_order"]
    assert out.required_mentions == ["placeOrder"]


# ------------------------------------------------- drafts that check a rebuild


def test_replayed_drafts_keep_every_solved_draft_from_known_replays(tmp_path):
    ctx = ReplayContext("gen_db.jsonl", "gen_sample.jsonl", str(tmp_path))
    ctx.source_replays[("shop", 1)] = None
    ctx.source_replays[("shop", 2)] = (False, True)
    ctx.source_replays[("shop", 3)] = (False,)
    rebuilt = make_draft(source_task_idx=0, code="rebuilt", call_errors=(False,))
    unsolved = make_draft(source_task_idx=1)
    solved = make_draft(source_task_idx=2)
    later = make_draft(source_task_idx=3)
    kept = replayed_drafts(ctx, SERVER, [rebuilt, unsolved, solved, later])
    assert [(d.source_task_idx, d.call_errors) for d in kept] == [
        (0, (False,)),
        (2, (False, True)),
        (3, (False,)),
    ]
    assert replayed_drafts(ctx, SERVER, [unsolved]) == []
    assert list(tmp_path.iterdir()) == []


def test_env_part_skips_scenarios_without_server_code():
    ctx = make_ctx(scripted_llm(lambda step: []))
    drafts = [make_draft(tools=SERVER_TOOLS, reference_calls=SERVER_CALLS)]
    assert ToolRenameRandom().apply(drafts, ctx) == []
    assert ToolRenameRandom().apply([make_draft(reference_calls=None)], ctx) == []
