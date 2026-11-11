import pytest

from awm.core.augment.dataset import (
    EnvVariant,
    env_scenario,
    merge_env_variants,
)
from awm.core.augment.draft import Variant
from awm.core.tool_view import TaskMeta


def env_variant(task: str, code: str | None = "new code") -> EnvVariant:
    meta = TaskMeta(
        augmentations=["tool_rename"], source_task_idx=3, source_scenario="Shop"
    )
    moved = Variant(task, meta, "code")
    return EnvVariant("Shop__tool_rename", "Shop", "tool_rename", code, [moved])


def scenario_rows() -> dict[str, list[dict]]:
    return {
        "envs": [{"scenario": "Shop", "full_code": "old code", "api_spec": {}}],
        "db": [{"scenario": "Shop", "db_schema": {}}],
        "sample": [{"scenario": "Shop", "sample_data": {}}],
        "tasks": [{"scenario": "Shop", "tasks": ["t0"]}],
    }


def test_env_scenario_joins_the_source_the_recipe_and_the_task():
    assert env_scenario("Shop", "add_params", 3) == "Shop__add_params__3"
    assert (
        env_scenario("Shop", "add_params+miss_func", 0)
        == "Shop__add_params+miss_func__0"
    )


def test_merge_adds_the_environment_to_every_scenario_file():
    rows, verifiers = merge_env_variants(scenario_rows(), [], [env_variant("a")], set())
    env = rows["envs"][-1]
    assert env["scenario"] == "Shop__tool_rename" and env["full_code"] == "new code"
    assert env["api_spec"] == {} and env["source_scenario"] == "Shop"
    assert rows["db"][-1]["db_schema"] == {} and rows["sample"][-1]["sample_data"] == {}
    assert rows["db"][-1]["scenario"] == "Shop__tool_rename"
    tasks = rows["tasks"][-1]
    assert tasks["tasks"] == ["a"]
    assert tasks["task_meta"][0]["source_scenario"] == "Shop"
    # the source scenario keeps only its own tasks
    assert rows["tasks"][0] == {"scenario": "Shop", "tasks": ["t0"]}
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


def test_merge_lists_the_environment_as_a_scenario_of_its_own():
    rows = {**scenario_rows(), "scenarios": [{"name": "Shop", "description": "d"}]}
    once, _ = merge_env_variants(rows, [], [env_variant("a")], set())
    assert once["scenarios"] == [
        {"name": "Shop", "description": "d"},
        {
            "name": "Shop__tool_rename",
            "description": "d",
            "source_scenario": "Shop",
            "augmentation": "tool_rename",
        },
    ]
    again, _ = merge_env_variants(once, [], [], set())
    assert [r["name"] for r in again["scenarios"]] == ["Shop"]


def test_merge_copies_the_source_code_when_the_environment_keeps_it():
    rows, _ = merge_env_variants(
        scenario_rows(), [], [env_variant("a", code=None)], set()
    )
    assert rows["envs"][-1]["full_code"] == "old code"
    assert rows["envs"][-1]["scenario"] == "Shop__tool_rename"


def test_merge_replaces_earlier_environments_unless_kept():
    once = merge_env_variants(scenario_rows(), [], [env_variant("a")], set())
    again = merge_env_variants(once[0], once[1], [env_variant("b")], set())
    assert [r["tasks"] for r in again[0]["tasks"]] == [["t0"], ["b"]]
    assert len(again[0]["envs"]) == 2 and len(again[1]) == 1
    kept = merge_env_variants(once[0], once[1], [], {"shop"})
    assert kept == once


def test_merge_rejects_an_environment_that_shares_a_database_file():
    rows = scenario_rows()
    for kind in ("envs", "db", "sample", "tasks"):
        rows[kind].append({**rows[kind][0], "scenario": "Shop Tool Rename"})
    with pytest.raises(ValueError, match="database file"):
        merge_env_variants(rows, [], [env_variant("a")], set())
