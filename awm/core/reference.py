"""Solve the original tasks once and keep the graded trajectories.

Task augmentation needs to know which tools and arguments solve a task (to
hide a tool or remove a parameter) and which tools each scenario offers. This
stage runs the agent on every original task, grades the run with the
code-mode verifier and stores the tool calls, plus the tool list per scenario.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass

from awm.core.agent import Config as AgentConfig
from awm.core.agent import run_agent
from awm.core.verify import Config as VerifyConfig
from awm.core.verify import VerificationMode, run_verify
from awm.tools import normalize_scenario_name, tools_jsonl_load, tools_jsonl_save

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # gen_tasks.jsonl; augmented tasks (task_meta set) are skipped
    input_task: str
    # code-mode verifier file (gen_verifier.pure_code.jsonl)
    input_verifier: str
    # graded runs, one row per task (gen_reference.jsonl)
    output: str
    # tool list per scenario as returned by list_tools (gen_tools.jsonl)
    output_tools: str
    envs_path: str = "./outputs/gen_envs.jsonl"
    db_path: str = "./outputs/gen_db.jsonl"
    sample_path: str = "./outputs/gen_sample.jsonl"
    # agent run directories; defaults to <output dir>/reference_runs
    runs_dir: str | None = None
    # agent model; defaults to AWM_SYN_OVERRIDE_MODEL
    model: str | None = None
    max_iterations: int = 30
    max_tokens: int = 8192
    temperature: float = 0.6
    # agent runs in parallel, each with its own MCP server
    concurrency: int = 16
    allowed_scenarios: list[str] | None = None
    # run at most this many tasks, for a sample run
    limit: int | None = None

    def pre_process(self):
        for path in [
            self.input_task,
            self.input_verifier,
            self.envs_path,
            self.db_path,
            self.sample_path,
        ]:
            assert os.path.exists(path), f"{path} not found"
        if self.runs_dir is None:
            self.runs_dir = os.path.join(
                os.path.dirname(self.output) or ".", "reference_runs"
            )
        if self.allowed_scenarios:
            self.allowed_scenarios = sorted(
                {normalize_scenario_name(name) for name in self.allowed_scenarios}
            )


def pending_tasks(
    task_items: list[dict],
    done: set[tuple[str, int]],
    allowed_scenarios: list[str] | None,
) -> list[tuple[str, int, str]]:
    """(scenario, task_idx, task) of original tasks without a reference yet."""
    pending = []
    for item in task_items:
        scenario = normalize_scenario_name(item["scenario"])
        if allowed_scenarios and scenario not in allowed_scenarios:
            continue
        metas = item.get("task_meta") or []
        for task_idx, task in enumerate(item["tasks"]):
            if task_idx < len(metas) and metas[task_idx] is not None:
                continue
            if (scenario, task_idx) not in done:
                pending.append((scenario, task_idx, task))
    return pending


async def solve_task(
    config: Config, scenario: str, task_idx: int, task: str
) -> tuple[dict, list[dict] | None]:
    """Run and grade one task; returns the reference row and the scenario's tools."""
    assert config.runs_dir is not None
    run_dir = os.path.join(config.runs_dir, scenario, f"task_{task_idx}")
    row = {"scenario": scenario, "task_idx": task_idx, "task": task}

    try:
        await run_agent(
            AgentConfig(
                scenario=scenario,
                task_id=task_idx,
                model=config.model,
                max_iterations=config.max_iterations,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                output_dir=run_dir,
                envs_path=config.envs_path,
                tasks_path=config.input_task,
                db_path=config.db_path,
                sample_path=config.sample_path,
                verbose=False,
            )
        )
        verdict = await run_verify(
            VerifyConfig(
                input=run_dir,
                mode=VerificationMode.code,
                verifier_code_path=config.input_verifier,
            )
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"{scenario} task {task_idx}: {e}")
        return {**row, "result": "agent_error", "error": str(e)}, None

    trajectory = load_json(os.path.join(run_dir, "trajectory.json"))
    final_answer = ""
    for msg in reversed(trajectory.get("messages", [])):
        if msg.get("role") == "assistant":
            final_answer = msg.get("content", "")
            break
    tools_path = os.path.join(run_dir, "tools.json")
    tools = load_json(tools_path) if os.path.exists(tools_path) else None

    row.update(
        {
            "result": verdict["reward_type"],
            "tool_calls": [
                {"tool_name": c["tool_name"], "arguments": c["arguments"]}
                for c in trajectory.get("tool_calls", [])
                if not c.get("rejected")
            ],
            "final_answer": final_answer,
            "run_dir": run_dir,
        }
    )
    return row, tools


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def run_all(config: Config) -> None:
    done = set()
    if os.path.exists(config.output):
        done = {
            (normalize_scenario_name(r["scenario"]), r["task_idx"])
            for r in tools_jsonl_load(config.output)
        }
    tools_done = set()
    if os.path.exists(config.output_tools):
        tools_done = {
            normalize_scenario_name(r["scenario"])
            for r in tools_jsonl_load(config.output_tools)
        }

    pending = pending_tasks(
        tools_jsonl_load(config.input_task), done, config.allowed_scenarios
    )
    if config.limit is not None:
        pending = pending[: config.limit]
    logger.info(f"{len(done)} tasks already have a reference, {len(pending)} to run")

    semaphore = asyncio.Semaphore(config.concurrency)

    async def worker(scenario: str, task_idx: int, task: str) -> None:
        async with semaphore:
            row, tools = await solve_task(config, scenario, task_idx, task)
        tools_jsonl_save([row], config.output, append=True)
        if tools is not None and scenario not in tools_done:
            tools_done.add(scenario)
            tools_jsonl_save(
                [{"scenario": scenario, "tools": tools}],
                config.output_tools,
                append=True,
            )
        logger.info(f"{scenario} task {task_idx}: {row['result']}")

    await asyncio.gather(*(worker(*p) for p in pending))

    rows = tools_jsonl_load(config.output) if os.path.exists(config.output) else []
    complete = sum(1 for r in rows if r.get("result") == "complete")
    logger.info(f"References: {complete}/{len(rows)} complete -> {config.output}")


def run(config: Config):
    asyncio.run(run_all(config))
