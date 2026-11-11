"""Summarize how well a model solves a dataset, from its graded runs.

`awm gen reference --include_augmented true` (scripts/run_gen_traj.sh) runs
the model on every task and stores one verifier verdict per task in
gen_reference.jsonl, next to the trajectories. This command reads those
verdicts with the dataset's gen_tasks.jsonl and reports the share of tasks
solved ("complete"), overall, per augmentation part and per source scenario,
and how often an augmented task is solved when its source task is.
"""

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from awm.tools import normalize_scenario_name, tools_jsonl_load

logger = logging.getLogger(__name__)

SOLVED = "complete"
# part name of the tasks of the original scenarios
ORIGINAL = "original"
# verdict of a task the run has not reached
NOT_RUN = "not_run"


@dataclass
class Config:
    # gen_tasks.jsonl of the dataset the model solved
    input_task: str
    # gen_reference.jsonl written by `awm gen reference`
    input_reference: str
    # the summary as JSON; defaults to <input_reference dir>/report.json
    output: str | None = None

    def pre_process(self):
        for path in [self.input_task, self.input_reference]:
            assert os.path.exists(path), f"{path} not found"
        if self.output is None:
            self.output = os.path.join(
                os.path.dirname(self.input_reference) or ".", "report.json"
            )


class TaskRow(BaseModel):
    """A line of gen_tasks.jsonl; augmented environments carry source_scenario."""

    model_config = ConfigDict(extra="ignore")

    scenario: str
    tasks: list[str]
    task_meta: list[dict | None] | None = None
    source_scenario: str | None = None
    augmentation: str | None = None


class ReferenceRow(BaseModel):
    """A line of gen_reference.jsonl."""

    model_config = ConfigDict(extra="ignore")

    scenario: str
    task_idx: int
    result: str


@dataclass(frozen=True)
class Entry:
    """A task of the dataset with the verdict of its run."""

    part: str
    source_scenario: str
    # index of the original task it comes from (itself for an original task)
    source_task_idx: int
    result: str

    @property
    def solved(self) -> bool:
        return self.result == SOLVED


def entries(task_rows: list[TaskRow], references: list[ReferenceRow]) -> list[Entry]:
    """Every task of the dataset, NOT_RUN where no verdict exists."""
    verdicts = {
        (normalize_scenario_name(r.scenario), r.task_idx): r.result for r in references
    }
    out = []
    for row in task_rows:
        scenario = normalize_scenario_name(row.scenario)
        metas = row.task_meta or [None] * len(row.tasks)
        for idx, meta in enumerate(metas):
            source_idx = (meta or {}).get("source_task_idx", idx)
            out.append(
                Entry(
                    part=row.augmentation or ORIGINAL,
                    source_scenario=row.source_scenario or row.scenario,
                    source_task_idx=source_idx,
                    result=verdicts.get((scenario, idx), NOT_RUN),
                )
            )
    return out


def rate(group: Iterable[Entry]) -> dict:
    """Solved share of the tasks that ran, and the count of every verdict."""
    group = list(group)
    ran = [e for e in group if e.result != NOT_RUN]
    results: dict[str, int] = {}
    for e in group:
        results[e.result] = results.get(e.result, 0) + 1
    solved = sum(e.solved for e in ran)
    return {
        "tasks": len(group),
        "ran": len(ran),
        "solved": solved,
        "solve_rate": solved / len(ran) if ran else None,
        "results": dict(sorted(results.items())),
    }


def by(key, group: list[Entry]) -> dict[str, dict]:
    groups: dict[str, list[Entry]] = {}
    for e in group:
        groups.setdefault(key(e), []).append(e)
    return {name: rate(members) for name, members in sorted(groups.items())}


def robustness(group: list[Entry]) -> dict[str, dict]:
    """Per part: of the augmented tasks whose source task was solved, the share solved."""
    solved_sources = {
        (normalize_scenario_name(e.source_scenario), e.source_task_idx)
        for e in group
        if e.part == ORIGINAL and e.solved
    }
    kept = [
        e
        for e in group
        if e.part != ORIGINAL
        and (normalize_scenario_name(e.source_scenario), e.source_task_idx)
        in solved_sources
    ]
    return by(lambda e: e.part, kept)


def summarize(task_rows: list[TaskRow], references: list[ReferenceRow]) -> dict:
    group = entries(task_rows, references)
    return {
        "overall": rate(group),
        "by_part": by(lambda e: e.part, group),
        "by_source_scenario": by(lambda e: e.source_scenario, group),
        "robustness_by_part": robustness(group),
    }


def format_table(title: str, stats: dict[str, dict]) -> str:
    """A text table of solve rates, one line per group."""
    width = max([len(name) for name in stats] + [len(title)])
    lines = [f"{title:<{width}}  solved/ran   rate  not_run"]
    for name, s in stats.items():
        share = "-" if s["solve_rate"] is None else f"{s['solve_rate']:6.1%}"
        not_run = s["results"].get(NOT_RUN, 0)
        lines.append(
            f"{name:<{width}}  {s['solved']:>4}/{s['ran']:<5} {share:>6}  {not_run:>7}"
        )
    return "\n".join(lines)


def run(config: Config):
    assert config.output is not None
    task_rows = [TaskRow.model_validate(r) for r in tools_jsonl_load(config.input_task)]
    references = [
        ReferenceRow.model_validate(r) for r in tools_jsonl_load(config.input_reference)
    ]
    summary = summarize(task_rows, references)
    with open(config.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(format_table("overall", {"all tasks": summary["overall"]}))
    print(f"verdicts: {summary['overall']['results']}")
    print()
    print(format_table("part", summary["by_part"]))
    print()
    print(format_table("augmented, source task solved", summary["robustness_by_part"]))
    print()
    print(format_table("source scenario", summary["by_source_scenario"]))
    logger.info(f"Saved the report to {config.output}")
