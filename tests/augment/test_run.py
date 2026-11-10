import os

import pytest
from helpers import CALLS, SOURCE_VERIFIER, TASK, TOOLS

from awm.core.augment import Config, output_path, run
from awm.tools import tools_jsonl_load, tools_jsonl_save

SCENARIOS = ["Shop", "Cafe"]


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
    for name in ["gen_envs", "gen_db", "gen_sample"]:
        tools_jsonl_save([{"scenario": s} for s in SCENARIOS], paths[name])
    return paths


def augment(paths: dict[str, str], tmp_path, **changes) -> None:
    config = Config(
        input_task=paths["gen_tasks"],
        input_verifier=paths["gen_verifier.pure_code"],
        input_reference=paths["gen_reference"],
        input_tools=paths["gen_tools"],
        envs_path=paths["gen_envs"],
        db_path=paths["gen_db"],
        sample_path=paths["gen_sample"],
        parts=["add_params"],
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
    tasks = tools_jsonl_load(aug(paths["gen_tasks"]))
    for item in tasks:
        assert item["tasks"][0] == TASK
        assert len(item["tasks"]) == 2
        assert item["task_meta"][0] is None
        assert item["task_meta"][1]["augmentations"] == ["add_params"]
    verifiers = tools_jsonl_load(aug(paths["gen_verifier.pure_code"]))
    assert sorted((r["scenario"], r["task_idx"]) for r in verifiers) == [
        ("Cafe", 0),
        ("Cafe", 1),
        ("Shop", 0),
        ("Shop", 1),
    ]
    for name in ["gen_envs", "gen_db", "gen_sample"]:
        assert tools_jsonl_load(aug(paths[name])) == (tools_jsonl_load(paths[name]))


def test_rerun_on_some_scenarios_keeps_what_the_others_have(tmp_path):
    paths = write_inputs(tmp_path)
    augment(paths, tmp_path)
    rebuilt = {
        "scenario": "Cafe__tool_rename_random",
        "source_scenario": "Cafe",
        "augmentation": "tool_rename_random",
    }
    envs_out = aug(paths["gen_envs"])
    tools_jsonl_save([*tools_jsonl_load(envs_out), rebuilt], envs_out)

    augment(paths, tmp_path, allowed_scenarios=["Shop"])

    tasks = {r["scenario"]: r for r in tools_jsonl_load(aug(paths["gen_tasks"]))}
    assert len(tasks["Cafe"]["tasks"]) == 2
    assert len(tasks["Shop"]["tasks"]) == 2
    assert rebuilt in tools_jsonl_load(envs_out)
