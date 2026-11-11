import json

from awm.core.report import (
    NOT_RUN,
    ORIGINAL,
    Config,
    ReferenceRow,
    TaskRow,
    format_table,
    run,
    summarize,
)
from awm.tools import tools_jsonl_save

TASK_ROWS = [
    TaskRow(scenario="Shop", tasks=["t0", "t1"]),
    TaskRow(
        scenario="Shop__add_params",
        source_scenario="Shop",
        augmentation="add_params",
        tasks=["a0", "a1"],
        task_meta=[
            {"augmentations": ["add_params"], "source_task_idx": 0},
            {"augmentations": ["add_params"], "source_task_idx": 1},
        ],
    ),
]


def verdicts(*rows: tuple[str, int, str]) -> list[ReferenceRow]:
    return [ReferenceRow(scenario=s, task_idx=i, result=r) for s, i, r in rows]


def test_summarize_counts_solved_tasks_per_part_and_source():
    summary = summarize(
        TASK_ROWS,
        verdicts(
            ("shop", 0, "complete"),
            ("shop", 1, "incomplete"),
            ("shop_add_params", 0, "complete"),
            ("shop_add_params", 1, "complete"),
        ),
    )
    assert summary["overall"]["solved"] == 3 and summary["overall"]["ran"] == 4
    assert summary["overall"]["solve_rate"] == 0.75
    assert summary["by_part"][ORIGINAL]["solve_rate"] == 0.5
    assert summary["by_part"]["add_params"]["solve_rate"] == 1.0
    assert summary["by_source_scenario"]["Shop"]["tasks"] == 4


def test_robustness_only_counts_augmented_tasks_whose_source_was_solved():
    summary = summarize(
        TASK_ROWS,
        verdicts(
            ("shop", 0, "complete"),
            ("shop", 1, "incomplete"),
            ("shop_add_params", 0, "incomplete"),
            ("shop_add_params", 1, "complete"),
        ),
    )
    # only a0 comes from a solved source task (t0), and a0 failed
    assert summary["robustness_by_part"]["add_params"]["ran"] == 1
    assert summary["robustness_by_part"]["add_params"]["solve_rate"] == 0.0


def test_tasks_without_a_verdict_are_not_run_and_left_out_of_the_rate():
    summary = summarize(TASK_ROWS, verdicts(("shop", 0, "agent_error")))
    overall = summary["overall"]
    assert overall["tasks"] == 4 and overall["ran"] == 1
    assert overall["solve_rate"] == 0.0
    assert overall["results"] == {"agent_error": 1, NOT_RUN: 3}
    assert summary["by_part"]["add_params"]["solve_rate"] is None


def test_format_table_shows_a_dash_for_groups_that_did_not_run():
    summary = summarize(TASK_ROWS, verdicts(("shop", 0, "complete")))
    table = format_table("part", summary["by_part"])
    assert "original" in table and "100.0%" in table
    assert next(line for line in table.splitlines() if "add_params" in line).count("-")


def test_run_writes_the_summary_as_json(tmp_path):
    tasks = tmp_path / "gen_tasks.jsonl"
    references = tmp_path / "gen_reference.jsonl"
    tools_jsonl_save([r.model_dump() for r in TASK_ROWS], str(tasks))
    tools_jsonl_save(
        [{"scenario": "shop", "task_idx": 0, "result": "complete", "tool_calls": []}],
        str(references),
    )
    config = Config(input_task=str(tasks), input_reference=str(references))
    config.pre_process()
    run(config)
    assert config.output == str(tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["overall"]["solved"] == 1
