import os

import pytest
from helpers import CALLS, SHOP_SERVER, SOURCE_VERIFIER, TASK, TOOLS

from awm.core.augment import Config, output_path, run
from awm.core.augment import recipe as recipe_module
from awm.tools import tools_jsonl_load, tools_jsonl_save

SCENARIOS = ["Shop", "Cafe"]


@pytest.fixture(autouse=True)
def no_view_problems(monkeypatch):
    """Servers are not started; every served view passes its check."""
    monkeypatch.setattr(recipe_module, "view_problems", lambda ctx, env, draft: [])


def write_inputs(tmp_path) -> dict[str, str]:
    (tmp_path / "data").mkdir()
    paths = {
        name: str(tmp_path / "data" / f"{name}.jsonl")
        for name in [
            "gen_tasks",
            "gen_verifier.pure_code",
            "gen_reference",
            "gen_tools",
            "gen_envs",
            "gen_db",
            "gen_sample",
            "gen_scenario",
        ]
    }
    tools_jsonl_save(
        [{"scenario": s, "tasks": [TASK]} for s in SCENARIOS], paths["gen_tasks"]
    )
    tools_jsonl_save(
        [
            {"scenario": s, "task_idx": 0, "verification": {"code": SOURCE_VERIFIER}}
            for s in SCENARIOS
        ],
        paths["gen_verifier.pure_code"],
    )
    tools_jsonl_save(
        [
            {"scenario": s, "task_idx": 0, "result": "complete", "tool_calls": CALLS}
            for s in SCENARIOS
        ],
        paths["gen_reference"],
    )
    tools_jsonl_save(
        [{"scenario": s, "tools": TOOLS} for s in SCENARIOS], paths["gen_tools"]
    )
    tools_jsonl_save(
        [{"scenario": s, "full_code": SHOP_SERVER} for s in SCENARIOS],
        paths["gen_envs"],
    )
    for name in ["gen_db", "gen_sample"]:
        tools_jsonl_save([{"scenario": s} for s in SCENARIOS], paths[name])
    tools_jsonl_save(
        [{"name": s, "description": f"{s} API"} for s in SCENARIOS],
        paths["gen_scenario"],
    )
    return paths


def augment(
    paths: dict[str, str],
    tmp_path,
    parts: list[str] | None = None,
    **changes,
) -> None:
    config = Config(
        input_task=paths["gen_tasks"],
        input_verifier=paths["gen_verifier.pure_code"],
        input_reference=paths["gen_reference"],
        input_tools=paths["gen_tools"],
        envs_path=paths["gen_envs"],
        db_path=paths["gen_db"],
        sample_path=paths["gen_sample"],
        parts=parts or ["add_params"],
        num_per_part=1,
        **changes,
    )
    config.pre_process()
    run(config)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def aug(path: str) -> str:
    """Where the default output_dir puts the input at path."""
    return path.replace("/data/", "/data_aug/")


def test_output_dir_defaults_to_the_input_dir_with_aug(tmp_path):
    paths = write_inputs(tmp_path)
    config = Config(
        input_task=paths["gen_tasks"],
        input_verifier=paths["gen_verifier.pure_code"],
        input_reference=paths["gen_reference"],
        input_tools=paths["gen_tools"],
        envs_path=paths["gen_envs"],
        db_path=paths["gen_db"],
        sample_path=paths["gen_sample"],
    )
    config.pre_process()
    assert config.output_dir is not None
    assert os.path.abspath(config.output_dir) == str(tmp_path / "data_aug")
    assert output_path(paths["gen_tasks"], config.output_dir) == os.path.join(
        config.output_dir, "gen_tasks.jsonl"
    )


def test_output_dir_must_not_be_the_input_dir(tmp_path):
    paths = write_inputs(tmp_path)
    with pytest.raises(AssertionError):
        augment(paths, tmp_path, output_dir=str(tmp_path / "data"))


def test_run_writes_the_whole_dataset_to_the_aug_dir_and_keeps_the_inputs(tmp_path):
    paths = write_inputs(tmp_path)
    before = {name: read(path) for name, path in paths.items()}

    augment(paths, tmp_path)

    assert {name: read(path) for name, path in paths.items()} == before
    tasks = {r["scenario"]: r for r in tools_jsonl_load(aug(paths["gen_tasks"]))}
    assert sorted(tasks) == [
        "Cafe",
        "Cafe__add_params__0",
        "Shop",
        "Shop__add_params__0",
    ]
    for source in SCENARIOS:
        assert tasks[source] == {"scenario": source, "tasks": [TASK]}
        env = tasks[f"{source}__add_params__0"]
        assert env["source_scenario"] == source
        assert len(env["tasks"]) == 1
        assert env["task_meta"][0]["augmentations"] == ["add_params"]
        assert env["task_meta"][0]["source_scenario"] == source
    verifiers = tools_jsonl_load(aug(paths["gen_verifier.pure_code"]))
    assert sorted((r["scenario"], r["task_idx"]) for r in verifiers) == [
        ("Cafe", 0),
        ("Cafe__add_params__0", 0),
        ("Shop", 0),
        ("Shop__add_params__0", 0),
    ]
    scenarios = tools_jsonl_load(aug(paths["gen_scenario"]))
    assert scenarios[:2] == tools_jsonl_load(paths["gen_scenario"])
    assert sorted((r["name"], r["description"]) for r in scenarios[2:]) == [
        ("Cafe__add_params__0", "Cafe API"),
        ("Shop__add_params__0", "Shop API"),
    ]
    for name in ["gen_envs", "gen_db", "gen_sample"]:
        rows = tools_jsonl_load(aug(paths[name]))
        assert rows[:2] == tools_jsonl_load(paths[name])
        assert sorted((r["scenario"], r["source_scenario"]) for r in rows[2:]) == [
            ("Cafe__add_params__0", "Cafe"),
            ("Shop__add_params__0", "Shop"),
        ]


def test_rerun_on_some_scenarios_keeps_what_the_others_have(tmp_path):
    paths = write_inputs(tmp_path)
    augment(paths, tmp_path)
    rebuilt = {
        "scenario": "Cafe__tool_rename_random__0",
        "source_scenario": "Cafe",
        "augmentation": "tool_rename_random",
    }
    envs_out = aug(paths["gen_envs"])
    tools_jsonl_save([*tools_jsonl_load(envs_out), rebuilt], envs_out)

    augment(paths, tmp_path, allowed_scenarios=["Shop"])

    tasks = {r["scenario"]: r for r in tools_jsonl_load(aug(paths["gen_tasks"]))}
    assert sorted(tasks) == [
        "Cafe",
        "Cafe__add_params__0",
        "Shop",
        "Shop__add_params__0",
    ]
    envs = [r["scenario"] for r in tools_jsonl_load(envs_out)]
    assert rebuilt in tools_jsonl_load(envs_out)
    assert envs.count("Shop__add_params__0") == envs.count("Cafe__add_params__0") == 1


def test_rerun_on_all_scenarios_drops_earlier_environments(tmp_path):
    paths = write_inputs(tmp_path)
    augment(paths, tmp_path)
    augment(paths, tmp_path)

    envs = [r["scenario"] for r in tools_jsonl_load(aug(paths["gen_envs"]))]
    assert sorted(envs) == [
        "Cafe",
        "Cafe__add_params__0",
        "Shop",
        "Shop__add_params__0",
    ]


def test_run_writes_a_chained_recipe_as_one_environment(tmp_path):
    paths = write_inputs(tmp_path)
    augment(paths, tmp_path, parts=["add_params+miss_func", "add_params"])

    tasks = {r["scenario"]: r for r in tools_jsonl_load(aug(paths["gen_tasks"]))}
    chained = tasks["Shop__add_params+miss_func__0"]
    assert chained["augmentation"] == "add_params+miss_func"
    [meta] = chained["task_meta"]
    assert meta["augmentations"] == ["add_params", "miss_func"]
    assert meta["hidden_tools"] == ["create_order"]
    assert meta["turns"][0]["shown_tools"] == ["create_order"]
    envs = {r["scenario"]: r for r in tools_jsonl_load(aug(paths["gen_envs"]))}
    assert envs["Shop"]["full_code"] == SHOP_SERVER
    assert "_install_tool_view()" in envs["Shop__add_params+miss_func__0"]["full_code"]
    assert "Shop__add_params__0" in tasks
    verifiers = {
        r["scenario"]: r for r in tools_jsonl_load(aug(paths["gen_verifier.pure_code"]))
    }
    chained_verifier = verifiers["Shop__add_params+miss_func__0"]
    assert chained_verifier["augmentations"] == ["add_params", "miss_func"]
    assert "unchanged_db_before_followup" in chained_verifier["verification"]["code"]


def test_recipes_that_cannot_run_are_refused_before_the_run(tmp_path):
    paths = write_inputs(tmp_path)
    with pytest.raises(ValueError, match="must come before"):
        augment(paths, tmp_path, parts=["add_params+tool_rename"])
    assert not os.path.exists(tmp_path / "data_aug")
