"""Serve the operations of a FastAPI app as MCP tools over streamable HTTP.

This replaces `FastApiMCP(app).mount_http()` from fastapi-mcp 0.4.0, which only
supports mcp 1.x. Tool calls are forwarded to the app in-process, and the
results, including the error texts that the mcp 1.x low-level server produced,
are reproduced so that agent trajectories keep their format.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import jsonschema
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from mcp import types
from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import (
    StreamableHTTPASGIApp,
    StreamableHTTPSessionManager,
)
from starlette.routing import Route

from awm.core.mcp_bridge.openapi import convert_openapi_to_mcp_tools

API_BASE_URL = "http://apiserver"


class ToolCallError(Exception):
    pass


def build_tools(app: FastAPI) -> tuple[list[types.Tool], dict[str, dict[str, Any]]]:
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    tool_dicts, operation_map = convert_openapi_to_mcp_tools(openapi_schema)
    tools = [
        types.Tool(
            name=t["name"], description=t["description"], input_schema=t["inputSchema"]
        )
        for t in tool_dicts
    ]
    return tools, operation_map


async def execute_operation(
    client: httpx.AsyncClient,
    tool_name: str,
    arguments: dict[str, Any],
    operation_map: dict[str, dict[str, Any]],
) -> str:
    """Send the HTTP request of one operation and return the response text.

    Raises on unknown tools and on 4xx/5xx responses, with the messages of
    fastapi-mcp 0.4.0.
    """
    if tool_name not in operation_map:
        raise ToolCallError(f"Unknown tool: {tool_name}")

    operation = operation_map[tool_name]
    path: str = operation["path"]
    method: str = operation["method"].lower()
    parameters: list[dict[str, Any]] = operation.get("parameters", [])
    arguments = dict(arguments)

    for param in parameters:
        if param.get("in") == "path" and param.get("name") in arguments:
            name = param["name"]
            path = path.replace(f"{{{name}}}", str(arguments.pop(name)))

    query = {}
    headers = {}
    for location, target in (("query", query), ("header", headers)):
        for param in parameters:
            if param.get("in") == location and param.get("name") in arguments:
                target[param["name"]] = arguments.pop(param["name"])

    body = arguments or None

    if method in ("get", "delete"):
        response = await client.request(method, path, params=query, headers=headers)
    elif method in ("post", "put", "patch"):
        response = await client.request(
            method, path, params=query, headers=headers, json=body
        )
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    try:
        result_text = json.dumps(response.json(), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        result_text = response.text

    if 400 <= response.status_code < 600:
        raise ToolCallError(
            f"Error calling {tool_name}. Status code: {response.status_code}. Response: {response.text}"
        )
    return result_text


def _error_result(text: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], is_error=True
    )


def create_mcp_server(app: FastAPI) -> Server:
    tools, operation_map = build_tools(app)
    tools_by_name = {t.name: t for t in tools}
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url=API_BASE_URL,
        timeout=10.0,
    )

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    # Mirrors the call_tool handler of the mcp 1.x low-level server: arguments
    # are validated against the input schema, and every failure is returned
    # as an error result carrying the exception message.
    async def call_tool(
        ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        arguments = params.arguments or {}
        try:
            tool = tools_by_name.get(params.name)
            if tool is not None:
                try:
                    jsonschema.validate(instance=arguments, schema=tool.input_schema)
                except jsonschema.ValidationError as e:
                    return _error_result(f"Input validation error: {e.message}")
            text = await execute_operation(
                client, params.name, arguments, operation_map
            )
        except Exception as e:  # noqa: BLE001
            return _error_result(str(e))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], is_error=False
        )

    return Server(
        app.title or "FastAPI MCP",
        description=app.description,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def mount_mcp(app: FastAPI, path: str = "/mcp") -> None:
    """Expose the routes registered on `app` so far as MCP tools at `path`."""
    session_manager = StreamableHTTPSessionManager(
        app=create_mcp_server(app), json_response=True, stateless=False
    )
    app.router.routes.append(
        Route(path, endpoint=StreamableHTTPASGIApp(session_manager))
    )

    app_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(asgi_app: Any) -> AsyncIterator[Any]:
        async with session_manager.run(), app_lifespan(asgi_app) as state:
            yield state

    app.router.lifespan_context = lifespan
