import json

from awm.core.augment.compose import existing_variants
from awm.core.reference import pending_tasks
from awm.tools import (
    is_rebuilt_env_row,
    original_task_rows,
    with_rebuilt_env_rows,
)

TASK_ROWS = [
    {
        "scenario": "Shop",
        "tasks": ["t0", "t1", "augmented"],
        "task_meta": [
            None,
            None,
            {"augmentations": ["add_params"], "source_task_idx": 0},
        ],
    },
    {"scenario": "Bank", "tasks": ["b0"]},
    {
        "scenario": "Shop__tool_rename",
        "source_scenario": "Shop",
        "augmentation": "tool_rename",
        "tasks": ["renamed"],
        "task_meta": [{"augmentations": ["tool_rename"], "source_task_idx": 1}],
    },
]


def test_is_rebuilt_env_row():
    assert is_rebuilt_env_row(TASK_ROWS[2])
    assert not is_rebuilt_env_row(TASK_ROWS[0])


def test_original_task_rows_drop_augmented_tasks_and_rebuilt_environments():
    assert original_task_rows(TASK_ROWS) == [
        {"scenario": "Shop", "tasks": ["t0", "t1"]},
        {"scenario": "Bank", "tasks": ["b0"]},
    ]
    # the rows read from the file are left as they are
    assert TASK_ROWS[0]["tasks"] == ["t0", "t1", "augmented"]


def test_with_rebuilt_env_rows_keeps_them_from_the_output_file(tmp_path):
    output = tmp_path / "gen_envs.jsonl"
    assert with_rebuilt_env_rows([{"scenario": "Shop"}], str(output)) == [
        {"scenario": "Shop"}
    ]
    old = [
        {"scenario": "Shop", "full_code": "old"},
        {"scenario": "Shop__tool_rename", "source_scenario": "Shop", "full_code": "r"},
    ]
    output.write_text("".join(json.dumps(r) + "\n" for r in old))
    new = [{"scenario": "Shop", "full_code": "new"}]
    assert with_rebuilt_env_rows(new, str(output)) == [*new, old[1]]
    # a result for the same scenario replaces the kept row
    replaced = [*new, {"scenario": "shop__tool_rename", "source_scenario": "Shop"}]
    assert with_rebuilt_env_rows(replaced, str(output)) == replaced


def test_reference_stage_only_solves_original_tasks():
    assert pending_tasks(TASK_ROWS, done={("shop", 1)}, allowed_scenarios=None) == [
        ("shop", 0, "t0"),
        ("bank", 0, "b0"),
    ]


def test_existing_variants_drop_task_meta_of_an_older_format():
    rows = [
        {
            "scenario": "Shop",
            "tasks": ["t0", "old", "new"],
            "task_meta": [
                None,
                {
                    "augmentations": ["tool_rename"],
                    "source_task_idx": 0,
                    "tool_view": {"rename_tools": {"a": "b"}},
                },
                {"augmentations": ["add_params"], "source_task_idx": 0},
            ],
        }
    ]
    verifiers = [
        {
            "scenario": "Shop",
            "task_idx": i,
            "augmented": True,
            "verification": {"code": "c"},
        }
        for i in (1, 2)
    ]
    assert [v.task for v in existing_variants(rows, verifiers)] == ["new"]
