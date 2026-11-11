import json
import shutil

import pytest
from helpers import (
    CALLS,
    SHOP_SERVER,
    TASK,
    in_process_tools,
    make_ctx,
    make_db,
    make_draft,
    scripted_llm,
)

from awm.core.augment import recipe as recipe_module
from awm.core.augment import replay as replay_module
from awm.core.augment.dataset import EnvVariant
from awm.core.augment.draft import Draft, Variant, finalize, values_in_task
from awm.core.augment.recipe import (
    PARTS,
    parse_recipe,
    presented_calls,
    run_recipe,
    view_listing_problems,
    view_problems,
)
from awm.core.augment.replay import Replay, ReplayContext
from awm.core.augment.tool_parts import (
    EXTRA_PARAM_POOL,
    MIN_EXTRA_PARAMS,
    split_responses,
)
from awm.core.tool_view import ToolView, Turn, bake_view
from awm.core.verify import run_verifier


def removals(step) -> list[dict]:
    """Replies that each remove one value, which is what miss_param asks for."""
    return [
        {"task": TASK.replace(v, "it"), "followup": f"It is {v}."}
        for v in values_in_task(TASK, CALLS)
    ]


LLM = scripted_llm(removals)


def grade(variant: Variant, tmp_path, calls: list[dict]) -> str:
    """Reward of a run that kept the first turn unchanged, then ordered 42."""
    run_dir = tmp_path / f"run_{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    initial = make_db(run_dir / "initial.db", [])
    turn0 = run_dir / "turn_0.db"
    shutil.copy2(initial, turn0)
    final = make_db(run_dir / "final.db", [42])
    entry = {"verification": {"code": variant.verifier_code}}
    reward, _ = run_verifier(entry, "code", initial, final, "", [str(turn0)], calls)
    return reward


def apply(name: str, drafts, llm=LLM):
    return PARTS[name].apply(drafts, make_ctx(llm))


# ------------------------------------------------------------------ parse


def test_parse_recipe_reads_single_parts_and_chains():
    assert parse_recipe("add_params").name == "add_params"
    recipe = parse_recipe("tool_rename + param_rename+add_params+miss_param")
    assert [p.name for p in recipe.parts] == [
        "tool_rename",
        "param_rename",
        "add_params",
        "miss_param",
    ]
    assert recipe.name == "tool_rename+param_rename+add_params+miss_param"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("add_params+tool_rename+miss_func", "must come before add_params"),
        ("add_params+nope", "unknown parts"),
        ("task_ambiguous+task_ambiguous", "more than once"),
        ("miss_func+miss_param", "both set ['followup']"),
        ("tool_rename+tool_rename_random", "both set ['tool_names']"),
        ("task_no_tool_names+task_distractor_names", "both set ['tool_mentions']"),
    ],
)
def test_parse_recipe_rejects_recipes_that_cannot_run(text, problem):
    with pytest.raises(ValueError, match=problem.replace("[", r"\[")):
        parse_recipe(text)


# ------------------------------------------------------------- task parts


def test_miss_func_variant_restores_the_tool_after_the_followup(tmp_path):
    [draft] = apply("miss_func", [make_draft()])
    variant = finalize(draft)
    meta = variant.meta
    assert meta.hidden_tools == ["create_order"]
    [turn] = meta.turns
    assert turn.shown_tools == ["create_order"]

    later = [{"turn": 1, "tool_name": "create_order", "arguments": {}}]
    early = [{"turn": 0, "tool_name": "create_order", "arguments": {}}]
    assert grade(variant, tmp_path, later) == "complete"
    assert grade(variant, tmp_path, early) == "others"


def test_miss_param_variant_gives_the_value_in_the_followup(tmp_path):
    [draft] = apply("miss_param", [make_draft()])
    variant = finalize(draft)
    assert draft.removed_value is not None
    assert draft.removed_value not in variant.task
    [turn] = variant.meta.turns
    assert draft.removed_value in turn.user
    assert grade(variant, tmp_path, []) == "complete"


def test_part_drops_drafts_it_cannot_handle():
    assert apply("miss_func", [make_draft(reference_calls=None)]) == []


# ----------------------------------------------------------------- recipes


ENV_CODE = {"shop": SHOP_SERVER, "bank": SHOP_SERVER}


@pytest.fixture(autouse=True)
def served(monkeypatch) -> list[str]:
    """Environments whose served view is checked; every check passes."""
    checked = []

    def no_problems(ctx, env, draft):
        checked.append(env.scenario)
        return []

    monkeypatch.setattr(recipe_module, "view_problems", no_problems)
    return checked


def run(
    text: str, drafts, num_per_part: int = 2, llm=LLM, env_code=ENV_CODE
) -> list[EnvVariant]:
    by_scenario: dict = {}
    for d in drafts:
        by_scenario.setdefault(d.scenario.lower(), []).append(d)
    ctx = make_ctx(llm, num_per_part, env_code=env_code)
    return run_recipe(parse_recipe(text), by_scenario, ctx)


def test_run_recipe_makes_an_environment_per_task_with_limited_tasks():
    drafts = [make_draft(source_task_idx=i) for i in range(5)]
    drafts.append(make_draft(scenario="Bank", source_task_idx=0))
    envs = run("add_params", drafts)
    assert [(e.scenario, e.source_scenario, len(e.variants)) for e in envs] == [
        ("Bank__add_params__0", "Bank", 1),
        ("Shop__add_params__0", "Shop", 1),
        ("Shop__add_params__1", "Shop", 1),
    ]
    assert {e.augmentation for e in envs} == {"add_params"}
    shop = [v.meta for e in envs[1:] for v in e.variants]
    assert [(m.source_scenario, m.source_task_idx) for m in shop] == [
        ("Shop", 0),
        ("Shop", 1),
    ]


def test_tasks_of_one_scenario_get_servers_of_their_own_views():
    envs = run("add_params", [make_draft(source_task_idx=i) for i in range(2)])
    assert len({e.code for e in envs}) == 2


def test_run_recipe_is_deterministic():
    drafts = [make_draft(source_task_idx=i) for i in range(6)]
    first = run("add_params+param_format", drafts)
    assert first == run("add_params+param_format", drafts)


def test_run_recipe_without_applicable_drafts_is_empty():
    assert run("miss_func", [make_draft(reference_calls=None)]) == []


def test_run_recipe_serves_the_view_from_the_environments_server(served):
    [env] = run("add_params", [make_draft()], 1)
    assert env.code is not None
    listed = {t["name"]: t for t in in_process_tools(env.code)}
    assert sorted(listed) == ["create_order", "list_coupons", "search_products"]
    for tool in listed.values():
        extra = set(tool["inputSchema"]["properties"]) & set(EXTRA_PARAM_POOL)
        assert len(extra) >= MIN_EXTRA_PARAMS
    assert env.variants[0].meta.hidden_tools == []
    assert served == ["Shop__add_params__0"]


def test_run_recipe_keeps_the_server_of_a_recipe_without_a_view():
    [env] = run(
        "task_distractor_names",
        [make_draft()],
        1,
        scripted_llm(lambda step: [{"task": TASK + " (not with list_coupons)"}]),
    )
    assert env.code is None


def test_run_recipe_skips_a_view_without_server_code(served):
    assert run("add_params", [make_draft()], 1, env_code={}) == []
    assert served == []


def test_run_recipe_drops_environments_whose_server_fails_the_check(monkeypatch):
    monkeypatch.setattr(
        recipe_module, "view_problems", lambda ctx, env, draft: ["broken"]
    )
    assert run("add_params", [make_draft()], 1) == []


def test_a_dropped_environment_leaves_room_for_the_next_task(monkeypatch):
    def first_fails(ctx, env, draft):
        return ["broken"] if draft.source_task_idx == 0 else []

    monkeypatch.setattr(recipe_module, "view_problems", first_fails)
    drafts = [make_draft(source_task_idx=i) for i in range(2)]
    [env] = run("add_params", drafts, 1)
    assert env.scenario == "Shop__add_params__1"


def test_chained_views_add_up_on_the_server_and_the_followup_hides_a_tool():
    [env] = run("add_params+param_format+miss_func", [make_draft()], 1)
    [variant] = env.variants
    meta = variant.meta
    assert env.scenario == "Shop__add_params+param_format+miss_func__0"
    assert meta.augmentations == ["add_params", "param_format", "miss_func"]
    assert meta.hidden_tools == ["create_order"]
    assert meta.turns == [Turn(user=meta.turns[0].user, shown_tools=["create_order"])]
    assert env.code is not None
    listed = {t["name"]: t for t in in_process_tools(env.code)}
    product_id = listed["create_order"]["inputSchema"]["properties"]["product_id"]
    assert product_id["type"] == "string"
    assert "unchanged_db_before_followup" in variant.verifier_code


# ------------------------------------------------------------ the served view


def shop_draft(**changes):
    """A draft of SHOP_SERVER, listing its tools as the server does."""
    return make_draft(tools=in_process_tools(SHOP_SERVER), **changes)


def fake_replay(monkeypatch, rewards: dict[str, str]) -> list[list[dict]]:
    """Replays that load the server in-process; graded by whether it serves a view."""
    calls_made = []

    def replay(ctx, code, db_scenario, calls, verifier_code, final_answer):
        calls_made.append(calls)
        reward = rewards["baked" if "_install_tool_view" in code else "source"]
        return Replay(in_process_tools(code), [None] * len(calls), reward)

    monkeypatch.setattr(recipe_module, "replay", replay)
    monkeypatch.setattr(replay_module, "replay", replay)
    return calls_made


def coupons_responses() -> str:
    """What the server lists of list_coupons after its description."""
    tool = next(t for t in in_process_tools(SHOP_SERVER) if t["name"] == "list_coupons")
    return split_responses(tool["description"])[1]


# as tool_doc_paraphrase writes it, the listed responses stay
VIEW = ToolView(
    tool_descriptions={"list_coupons": "Coupons you can use." + coupons_responses()},
    extra_params={
        "create_order": {"trace_id": {"type": "string", "description": "T."}}
    },
    param_formats={"create_order": {"product_id": "int_string"}},
    distractor_tools=[{"name": "send_fax", "description": "Fax.", "inputSchema": {}}],
)


def served_env(code: str) -> EnvVariant:
    return EnvVariant("Shop__x", "Shop", "x", code, [])


def test_view_problems_accepts_a_server_that_presents_the_view(monkeypatch):
    calls_made = fake_replay(monkeypatch, {"baked": "complete", "source": "complete"})
    ctx = make_ctx(LLM, env_code=ENV_CODE)
    draft = shop_draft(view=VIEW)
    assert view_problems(ctx, served_env(bake_view(SHOP_SERVER, VIEW)), draft) == []
    # the reference calls are made as an agent makes them on the presented tools
    assert calls_made == [presented_calls(CALLS, VIEW)]


def test_view_problems_reports_a_server_that_does_not_present_it(monkeypatch):
    fake_replay(monkeypatch, {"baked": "complete", "source": "complete"})
    ctx = make_ctx(LLM, env_code=ENV_CODE)
    problems = view_problems(ctx, served_env(SHOP_SERVER), shop_draft(view=VIEW))
    assert any("send_fax" in p for p in problems)
    assert any("['list_coupons'] are not listed with their new" in p for p in problems)
    assert any(
        "create_order does not list the parameters ['trace_id']" in p for p in problems
    )


def test_view_problems_blames_the_view_only_if_the_source_solves_the_task(monkeypatch):
    env = served_env(bake_view(SHOP_SERVER, VIEW))
    fake_replay(monkeypatch, {"baked": "others", "source": "others"})
    ctx = make_ctx(LLM, env_code=ENV_CODE)
    assert view_problems(ctx, env, shop_draft(view=VIEW)) == []
    fake_replay(monkeypatch, {"baked": "others", "source": "complete"})
    ctx = make_ctx(LLM, env_code=ENV_CODE)
    assert view_problems(ctx, env, shop_draft(view=VIEW)) == [
        "the presented reference calls are graded others"
    ]


def test_view_problems_replays_the_source_once_per_task(monkeypatch):
    calls_made = fake_replay(monkeypatch, {"baked": "others", "source": "complete"})
    ctx = make_ctx(LLM, env_code=ENV_CODE)
    env = served_env(bake_view(SHOP_SERVER, VIEW))
    view_problems(ctx, env, shop_draft(view=VIEW))
    view_problems(ctx, env, shop_draft(view=VIEW))
    # two replays on the baked server, one on the source server
    assert len(calls_made) == 3


def test_view_problems_has_nothing_to_check_without_a_view(monkeypatch):
    calls_made = fake_replay(monkeypatch, {"baked": "others", "source": "others"})
    ctx = make_ctx(LLM, env_code=ENV_CODE)
    assert view_problems(ctx, served_env(SHOP_SERVER), shop_draft()) == []
    assert calls_made == []


def test_presented_calls_write_formatted_values_as_presented():
    calls = presented_calls(CALLS, VIEW)
    assert calls[1]["arguments"] == {"product_id": "42", "note": "gift"}
    assert CALLS[1]["arguments"]["product_id"] == 42


def test_view_listing_problems_name_what_the_server_does_not_show():
    tools = in_process_tools(bake_view(SHOP_SERVER, VIEW))
    draft = shop_draft(view=VIEW)
    assert view_listing_problems(tools, draft) == []
    unlisted = [t for t in tools if t["name"] != "send_fax"]
    [problem] = view_listing_problems(unlisted, draft)
    assert "expected" in problem and "send_fax" in problem


# ------------------------------------------- the served view (start servers)


@pytest.fixture
def replay_ctx(tmp_path) -> ReplayContext:
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


ORDERING_SERVER = """
from fastapi import FastAPI, Query
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///placeholder.db", connect_args={"check_same_thread": False})
app = FastAPI(title="Shop")


@app.post("/orders", summary="Create an order.", operation_id="create_order")
def create_order(product_id: int = Query(..., description="Product.")):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO orders (product_id) VALUES (:p)"), {"p": product_id})
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

ORDERING_VIEW = ToolView(
    extra_params={
        "create_order": {"trace_id": {"type": "string", "description": "T."}}
    },
    param_formats={"create_order": {"product_id": "int_string"}},
    distractor_tools=[{"name": "send_fax", "description": "Fax.", "inputSchema": {}}],
)


def ordering_draft() -> Draft:
    return make_draft(
        tools=in_process_tools(ORDERING_SERVER),
        reference_calls=[
            {"tool_name": "create_order", "arguments": {"product_id": 42}}
        ],
        view=ORDERING_VIEW,
    )


def test_a_started_server_serves_the_view_it_was_baked_with(replay_ctx):
    ctx = make_ctx(LLM, replay=replay_ctx, env_code={"shop": ORDERING_SERVER})
    env = served_env(bake_view(ORDERING_SERVER, ORDERING_VIEW))
    assert view_problems(ctx, env, ordering_draft()) == []


def test_a_started_server_without_the_view_fails_the_check(replay_ctx):
    ctx = make_ctx(LLM, replay=replay_ctx, env_code={"shop": ORDERING_SERVER})
    problems = view_problems(ctx, served_env(ORDERING_SERVER), ordering_draft())
    assert any("send_fax" in p for p in problems)
