"""Add augmented environments to the dataset as scenarios of their own.

Every augmented task goes to an environment of its own,
<scenario>__<recipe>__<source task index>, so that tasks augmented in
different ways never share a server or a name. The environment copies the
source's rows of the scenario files (with the server code rewritten by the
parts that rename tools, and serving the task's tool view), so its server,
database and verifiers do not depend on the source scenario.
"""

from dataclasses import dataclass

from awm.core.augment.draft import Variant
from awm.tools import is_augmented_env_row, normalize_scenario_name

SCENARIO_SEPARATOR = "__"
# field naming the scenario of a row, per kind of scenario file
NAME_KEY = {"scenarios": "name"}


def row_name(kind: str, row: dict) -> str:
    return row[NAME_KEY.get(kind, "scenario")]


def env_scenario(source_scenario: str, recipe: str, source_task_idx: int) -> str:
    return SCENARIO_SEPARATOR.join([source_scenario, recipe, str(source_task_idx)])


@dataclass
class EnvVariant:
    """An augmented environment and its tasks, ready to be added as a scenario."""

    scenario: str
    source_scenario: str
    # name of the recipe that made the tasks
    augmentation: str
    # server code of the environment; None keeps the source's
    code: str | None
    variants: list[Variant]


def merge_env_variants(
    rows: dict[str, list[dict]],
    env_verifier_rows: list[dict],
    env_variants: list[EnvVariant],
    keep_sources: set[str],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Scenario files and verifier rows of augmented environments, replaced.

    rows: rows of the "envs", "db", "sample" and "tasks" files, and of the
    "scenarios" file (gen_scenario.jsonl, where scenarios are listed) if given.
    env_verifier_rows: verifier rows of augmented environments.
    Augmented environments of scenarios in keep_sources (normalized) stay; the
    others are dropped, and env_variants are added.
    """

    def kept(row: dict) -> bool:
        return (
            not is_augmented_env_row(row)
            or normalize_scenario_name(row["source_scenario"]) in keep_sources
        )

    new_rows = {kind: [r for r in rs if kept(r)] for kind, rs in rows.items()}
    new_verifiers = [r for r in env_verifier_rows if kept(r)]

    taken = {normalize_scenario_name(r["scenario"]) for r in new_rows["envs"]}
    for env in env_variants:
        name = normalize_scenario_name(env.scenario)
        if name in taken:
            raise ValueError(
                f"{env.scenario} has the database file of a scenario already in the dataset"
            )
        taken.add(name)

    def source_row(kind: str, scenario: str) -> dict:
        norm = normalize_scenario_name(scenario)
        return next(
            r
            for r in rows[kind]
            if not is_augmented_env_row(r)
            and normalize_scenario_name(row_name(kind, r)) == norm
        )

    for env in env_variants:
        marker = {
            "scenario": env.scenario,
            "source_scenario": env.source_scenario,
            "augmentation": env.augmentation,
        }
        code = {} if env.code is None else {"full_code": env.code}
        new_rows["envs"].append(
            {**source_row("envs", env.source_scenario), **marker, **code}
        )
        for kind in ("db", "sample"):
            new_rows[kind].append({**source_row(kind, env.source_scenario), **marker})
        if "scenarios" in rows:
            listed = {**source_row("scenarios", env.source_scenario), **marker}
            del listed["scenario"]
            listed["name"] = env.scenario
            new_rows["scenarios"].append(listed)
        new_rows["tasks"].append(
            {
                **marker,
                "tasks": [v.task for v in env.variants],
                "task_meta": [v.meta.model_dump() for v in env.variants],
            }
        )
        for idx, variant in enumerate(env.variants):
            new_verifiers.append(
                {
                    "scenario": env.scenario,
                    "task_idx": idx,
                    "task": variant.task,
                    "verification": {"code": variant.verifier_code},
                    "augmented": True,
                    "augmentations": variant.meta.augmentations,
                    "source_task_idx": variant.meta.source_task_idx,
                    "source_scenario": env.source_scenario,
                }
            )
    return new_rows, new_verifiers
