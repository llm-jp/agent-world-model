import asyncio
import json
import sqlite3
import time
import urllib.request

import pytest

from awm.core.env import test_run_specific_env as start_check
from awm.core.server import Config as ServerConfig
from awm.core.server import (
    ServerStartError,
    _generate_server_code,
    _prepare_database,
    serve_scenario,
    start_server_process,
    stop_server_process,
)
from awm.tools import get_random_available_port, wait_for_server

SERVER = """
from fastapi import FastAPI
from sqlalchemy import create_engine

engine = create_engine("sqlite:///placeholder.db", connect_args={"check_same_thread": False})
app = FastAPI(title="Shop")
__EXTRA__

@app.get("/products", summary="Search products.", operation_id="search_products")
def search_products(query: str):
    return {"ids": [42]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

BROKEN = 'raise RuntimeError("broken on purpose")'


def write_scenario(tmp_path, extra: str = "") -> dict[str, str]:
    """Environment, schema and sample files of a one-tool scenario "shop"."""
    paths = {
        "envs": tmp_path / "envs.jsonl",
        "db": tmp_path / "db.jsonl",
        "sample": tmp_path / "sample.jsonl",
    }
    code = SERVER.replace("__EXTRA__", extra)
    paths["envs"].write_text(json.dumps({"scenario": "shop", "full_code": code}))
    ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY)"
    schema = {"tables": [{"name": "t", "ddl": ddl}]}
    paths["db"].write_text(json.dumps({"scenario": "shop", "db_schema": schema}))
    sample = {
        "tables": [
            {"table_name": "t", "insert_statements": ["INSERT INTO t VALUES (7)"]}
        ]
    }
    paths["sample"].write_text(json.dumps({"scenario": "shop", "sample_data": sample}))
    return {k: str(v) for k, v in paths.items()}


def start(tmp_path, extra: str = ""):
    """A server started with the low-level helpers, and its port."""
    paths = write_scenario(tmp_path, extra)
    db_file = _prepare_database(
        ServerConfig(
            scenario="shop",
            envs_load_path=paths["envs"],
            db_schema_path=paths["db"],
            sample_path=paths["sample"],
        ),
        str(tmp_path),
    )
    port = get_random_available_port()
    proc = start_server_process("shop", paths["envs"], db_file, port, str(tmp_path))
    return proc, port


# ------------------------------------------------------- start and stop


def test_stopping_the_process_also_stops_the_server_it_runs(tmp_path):
    proc, port = start(tmp_path)
    try:
        assert wait_for_server(port, timeout=60, process=proc)
    finally:
        stop_server_process(proc)
    assert not wait_for_server(port, timeout=2)


def test_stopping_an_exited_process_is_harmless(tmp_path):
    proc, _ = start(tmp_path, extra=BROKEN)
    proc.wait(timeout=60)
    stop_server_process(proc)


def test_wait_gives_up_as_soon_as_the_server_exits(tmp_path):
    timeout = 30
    proc, port = start(tmp_path, extra=BROKEN)
    began = time.monotonic()
    try:
        assert not wait_for_server(port, timeout=timeout, process=proc)
    finally:
        stop_server_process(proc)
    assert time.monotonic() - began < timeout / 2


# ------------------------------------------------------- serve_scenario


def serve(tmp_path, paths: dict[str, str], **kwargs):
    return serve_scenario(
        "shop",
        paths["envs"],
        str(tmp_path / "run"),
        db_schema_path=paths["db"],
        sample_path=paths["sample"],
        **kwargs,
    )


def test_serve_scenario_runs_on_a_fresh_database_and_stops_after(tmp_path):
    paths = write_scenario(tmp_path)

    async def use():
        async with serve(tmp_path, paths) as server:
            assert await asyncio.to_thread(wait_for_server, server.port, 5)
            return server

    server = asyncio.run(use())
    assert not wait_for_server(server.port, timeout=2)
    for db in [server.db_path, server.initial_db_path]:
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT id FROM t").fetchall() == [(7,)]


def test_serve_scenario_raises_with_the_server_output(tmp_path):
    paths = write_scenario(tmp_path, extra=BROKEN)

    async def use():
        async with serve(tmp_path, paths, timeout=30):
            pytest.fail("the block must not run")

    began = time.monotonic()
    with pytest.raises(ServerStartError) as error:
        asyncio.run(use())
    assert "broken on purpose" in error.value.output
    assert time.monotonic() - began < 15


def test_env_start_check_reports_the_output_of_a_broken_server(tmp_path):
    for extra, works in [("", True), (BROKEN, False)]:
        paths = write_scenario(tmp_path, extra)
        db = _prepare_database(
            ServerConfig(
                scenario="shop",
                envs_load_path=paths["envs"],
                db_schema_path=paths["db"],
                sample_path=paths["sample"],
            ),
            str(tmp_path / f"db_{works}"),
        )
        with open(paths["envs"]) as f:
            env = {**json.loads(f.read()), "db_path": db}
        ok, output, _ = start_check(0, env)
        assert ok is works
        assert ("broken on purpose" in output) is not works


# ------------------------------------------------------- server code


DB_URL_ROUTE = """
@app.get("/db_url", summary="Database URL.", operation_id="db_url")
def db_url():
    return {"url": str(engine.url)}
"""


def test_server_code_goes_to_the_run_directory_of_each_server(tmp_path):
    paths = write_scenario(tmp_path)
    code_paths = []
    for run in ["run_a", "run_b"]:
        config = ServerConfig(scenario="shop", envs_load_path=paths["envs"])
        (tmp_path / run).mkdir()
        db = str(tmp_path / run / "env.db")
        code_paths.append(_generate_server_code(config, db, str(tmp_path / run)))
    assert code_paths == [
        str(tmp_path / "run_a" / "server_code.py"),
        str(tmp_path / "run_b" / "server_code.py"),
    ]
    for run, code_path in zip(["run_a", "run_b"], code_paths, strict=True):
        with open(code_path) as f:
            assert str(tmp_path / run / "env.db") in f.read()
    assert not list(tmp_path.glob("temp_server_*.py"))


def test_servers_of_one_scenario_at_once_use_their_own_databases(tmp_path):
    paths = write_scenario(tmp_path, extra=DB_URL_ROUTE)

    def db_url(port: int) -> str:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/db_url") as response:
            return json.load(response)["url"]

    async def use():
        async with (
            serve_scenario(
                "shop",
                paths["envs"],
                str(tmp_path / "run_a"),
                db_schema_path=paths["db"],
                sample_path=paths["sample"],
            ) as a,
            serve_scenario(
                "shop",
                paths["envs"],
                str(tmp_path / "run_b"),
                db_schema_path=paths["db"],
                sample_path=paths["sample"],
            ) as b,
        ):
            return [
                (server.db_path, await asyncio.to_thread(db_url, server.port))
                for server in (a, b)
            ]

    results = asyncio.run(use())
    assert results[0][0] != results[1][0]
    for db_path, url in results:
        assert url == f"sqlite:///{db_path}"
