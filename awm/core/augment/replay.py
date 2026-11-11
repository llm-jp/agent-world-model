"""Make a task's reference calls on a fresh server and grade the database.

Parts that rewrite the server code (env_parts.py) check a rewrite this way: the
reference calls of solved tasks, renamed, have to behave as on the original
server and leave a database the task's verifier grades complete.
"""

import asyncio
import dataclasses
import os
import shutil
import tempfile
from dataclasses import dataclass, field

from awm.core.agent import MCPToolExecutor
from awm.core.augment.draft import Draft
from awm.core.server import RunningServer, serve_scenario
from awm.core.verify import run_verifier
from awm.tools import normalize_scenario_name, tools_jsonl_save

SERVER_START_TIMEOUT = 60.0
CALL_TIMEOUT = 60.0
# MCPToolExecutor.call_tool marks a failed call with this prefix
TOOL_ERROR_PREFIX = "Error: "


@dataclass
class ReplayContext:
    # gen_db.jsonl and gen_sample.jsonl, to create a fresh database
    db_path: str
    sample_path: str
    # directory for the servers and databases of the replays, removed after each
    work_dir: str
    # (normalized scenario, task index) -> whether each reference call failed on
    # the source server, None if the replay did not solve the task
    source_replays: dict[tuple[str, int], tuple[bool, ...] | None] = field(
        default_factory=dict
    )


@dataclass
class Replay:
    tools: list[dict]
    # error text of each call, None if it succeeded
    errors: list[str | None]
    reward: str


async def _make_calls(
    url: str, calls: list[dict]
) -> tuple[list[dict], list[str | None]]:
    """The listed tools, and the error text of each call (None if it succeeded)."""
    executor = MCPToolExecutor(url, timeout=CALL_TIMEOUT)
    tools = await executor.list_tools()
    errors: list[str | None] = []
    for call in calls:
        try:
            text = await executor.call_tool(
                call["tool_name"], call.get("arguments") or {}
            )
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
            continue
        errors.append(text if text.startswith(TOOL_ERROR_PREFIX) else None)
    return tools, errors


def replay(
    ctx: ReplayContext,
    code: str,
    db_scenario: str,
    calls: list[dict],
    verifier_code: str,
    final_answer: str,
) -> Replay:
    """Make the calls on a fresh server and database, then grade the database.

    The server runs the way `awm agent` runs it, on the database of db_scenario.
    Its code and databases are removed afterwards; a server that does not
    start raises ServerStartError with its output.
    """
    os.makedirs(ctx.work_dir, exist_ok=True)
    run_dir = tempfile.mkdtemp(dir=ctx.work_dir)
    try:
        envs_path = os.path.join(run_dir, "env.jsonl")
        tools_jsonl_save([{"scenario": db_scenario, "full_code": code}], envs_path)

        async def session() -> tuple[RunningServer, list[dict], list[str | None]]:
            async with serve_scenario(
                db_scenario,
                envs_path,
                run_dir,
                db_schema_path=ctx.db_path,
                sample_path=ctx.sample_path,
                timeout=SERVER_START_TIMEOUT,
            ) as server:
                tools, errors = await _make_calls(server.url, calls)
            return server, tools, errors

        server, tools, errors = asyncio.run(session())
        entry = {"verification": {"code": verifier_code}}
        reward, _ = run_verifier(
            entry, "code", server.initial_db_path, server.db_path, final_answer, [], []
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return Replay(tools=tools, errors=errors, reward=reward)


def tool_set_problems(listed: set[str], expected: set[str]) -> list[str]:
    """How the names of the tools a server lists differ from the expected ones."""
    if listed == expected:
        return []
    return [f"The server lists the tools {sorted(listed)}, expected {sorted(expected)}"]


def source_call_errors(
    ctx: ReplayContext, code: str, draft: Draft
) -> tuple[bool, ...] | None:
    """Whether each reference call of the draft fails on the source server.

    None if the replay does not solve the task. Each task is replayed once
    per run. Raises ServerStartError if the source server does not start.
    """
    key = (normalize_scenario_name(draft.scenario), draft.source_task_idx)
    if key not in ctx.source_replays:
        result = replay(
            ctx,
            code,
            draft.scenario,
            draft.reference_calls or [],
            draft.source_verifier_code,
            draft.reference_answer,
        )
        ctx.source_replays[key] = (
            tuple(e is not None for e in result.errors)
            if result.reward == "complete"
            else None
        )
    return ctx.source_replays[key]


def replayed_drafts(ctx: ReplayContext, code: str, drafts: list[Draft]) -> list[Draft]:
    """The drafts, in order, whose reference calls solve them on a replay.

    A run of the agent may have depended on something a replay does not
    reproduce (e.g. the time), so a draft is kept only if the server solves it
    again. Drafts of a rebuilt server were replayed when it was checked; the
    others are replayed on the source server. Each kept draft records which of
    its calls failed.
    Raises ServerStartError if the source server does not start.
    """
    kept = []
    for draft in drafts:
        errors = draft.call_errors
        if errors is None:
            assert draft.code is None, "a rebuilt server's drafts carry their replay"
            errors = source_call_errors(ctx, code, draft)
        if errors is not None:
            kept.append(dataclasses.replace(draft, call_errors=errors))
    return kept
