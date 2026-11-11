"""Present a tool view from inside an environment's FastAPI server.

The source of this module is copied into the server code of an augmented
environment (bake_view in tool_view.py). Such a server also runs where awm is
not installed, e.g. under an RL trainer, so this module imports nothing from
awm, needs only FastAPI and Starlette, and keeps to Python 3.10 syntax.

The MCP layers that serve a FastAPI app as tools (awm.core.mcp_bridge and
fastapi-mcp) list the tools of the OpenAPI document that
fastapi.openapi.utils.get_openapi builds from the app's routes. install()
wraps that function so the document already presents the view, adds the
route that answers calls of distractor tools, and converts the values of
parameters presented in another format back before the routes see them.
"""

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode

# the MCP layers append the responses of an operation to its description from here on
RESPONSES_MARKER = "\n\n### Responses:\n"
# route that answers the calls of distractor tools, which the server does not have
DISTRACTOR_PATH = "/__distractor_tools__/{tool_name}"
DISTRACTOR_STATUS = 404
FORMAT_ERROR_STATUS = 422
SCHEMA_REF_PREFIX = "#/components/schemas/"


@dataclass(frozen=True)
class ValueFormat:
    """Another way to write the values of a parameter, convertible both ways."""

    # JSON schema type of the parameter on the server
    server_type: str
    # JSON schema keys of the presented parameter (its type and constraints)
    schema: dict
    # added to the parameter's description
    note: str
    # the presented value for a value the server takes; raises FORMAT_ERRORS if none
    to_agent: Callable[[Any], Any]
    # the value the server takes for a presented value; raises FORMAT_ERRORS if malformed
    to_server: Callable[[Any], Any]


FORMAT_ERRORS = (TypeError, ValueError)


def _match(pattern: str, value: Any) -> re.Match[str]:
    if not isinstance(value, str):
        raise TypeError(value)
    m = re.fullmatch(pattern, value)
    if m is None:
        raise ValueError(value)
    return m


def _from_bool(value: Any) -> str:
    if not isinstance(value, bool):
        raise TypeError(value)
    return "yes" if value else "no"


def _to_bool(value: Any) -> bool:
    if value not in ("yes", "no"):
        raise ValueError(value)
    return value == "yes"


def _from_int(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(value)
    return str(value)


ISO_DATE = r"(\d{4})-(\d{2})-(\d{2})"

VALUE_FORMATS: dict[str, ValueFormat] = {
    "date_compact": ValueFormat(
        server_type="string",
        schema={"type": "string", "pattern": r"^\d{8}$"},
        note="Write the date as YYYYMMDD, e.g. 20260520.",
        to_agent=lambda v: "".join(_match(ISO_DATE, v).groups()),
        to_server=lambda v: "-".join(_match(r"(\d{4})(\d{2})(\d{2})", v).groups()),
    ),
    "date_slash": ValueFormat(
        server_type="string",
        schema={"type": "string", "pattern": r"^\d{4}/\d{2}/\d{2}$"},
        note="Write the date as YYYY/MM/DD, e.g. 2026/05/20.",
        to_agent=lambda v: "/".join(_match(ISO_DATE, v).groups()),
        to_server=lambda v: "-".join(_match(r"(\d{4})/(\d{2})/(\d{2})", v).groups()),
    ),
    "bool_yes_no": ValueFormat(
        server_type="boolean",
        schema={"type": "string", "enum": ["yes", "no"]},
        note='Write "yes" or "no".',
        to_agent=_from_bool,
        to_server=_to_bool,
    ),
    "int_string": ValueFormat(
        server_type="integer",
        schema={"type": "string", "pattern": r"^-?\d+$"},
        note='Write the number as a string of digits, e.g. "42".',
        to_agent=_from_int,
        to_server=lambda v: int(_match(r"-?\d+", v).group()),
    ),
}

# schema keys that describe the server's form of a value, replaced by a format
VALUE_KEYS = (
    "type",
    "format",
    "pattern",
    "enum",
    "anyOf",
    "oneOf",
    "default",
    "example",
    "examples",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
)


def formatted_schema(schema: dict, fmt: str) -> dict:
    """The parameter's schema with its values written in another format."""
    value_format = VALUE_FORMATS[fmt]
    out = {k: v for k, v in schema.items() if k not in VALUE_KEYS}
    out.update(copy.deepcopy(value_format.schema))
    if "example" in schema:
        try:
            out["example"] = value_format.to_agent(schema["example"])
        except FORMAT_ERRORS:
            pass
    out["description"] = f"{schema.get('description', '')} {value_format.note}".strip()
    return out


def format_rejection(tool: str, param: str, value: Any, fmt: str) -> str:
    """Error text for a value that is not written in the presented format."""
    note = VALUE_FORMATS[fmt].note
    return f"The parameter '{param}' of '{tool}' does not accept {value!r}. {note}"


# ------------------------------------------------------------ the document


def _inlined(schema: dict, components: dict) -> dict:
    """The schema with a reference at its top level replaced by what it names."""
    ref = schema.get("$ref", "")
    if not ref.startswith(SCHEMA_REF_PREFIX):
        return schema
    named = copy.deepcopy(components.get(ref[len(SCHEMA_REF_PREFIX) :], {}))
    named.update({k: v for k, v in schema.items() if k != "$ref"})
    return named


def _body_schema(operation: dict, components: dict) -> dict:
    """The operation's request body schema, inlined so that edits stay in it."""
    content = (operation.get("requestBody") or {}).get("content") or {}
    if not content:
        return {}
    media = next(iter(content.values()))
    schema = _inlined(media.get("schema") or {}, components)
    media["schema"] = schema
    return schema


def _parameter(operation: dict, name: str) -> dict | None:
    for param in operation.get("parameters", []):
        if param.get("name") == name and param.get("in") in ("path", "query", "header"):
            return param
    return None


def _property(operation: dict, components: dict, name: str) -> dict | None:
    """The parameter's schema with its description, as the MCP layers list it."""
    param = _parameter(operation, name)
    if param is not None:
        prop = dict(param.get("schema") or {})
        if param.get("description"):
            prop["description"] = param["description"]
        return prop
    found = _body_schema(operation, components).get("properties", {}).get(name)
    return None if found is None else _inlined(found, components)


def _set_property(operation: dict, components: dict, name: str, prop: dict) -> None:
    param = _parameter(operation, name)
    if param is None:
        _body_schema(operation, components)["properties"][name] = prop
        return
    prop = dict(prop)
    param["description"] = prop.pop("description", "")
    param["schema"] = prop


def _present_operation(
    operation: dict, name: str, view: dict, components: dict
) -> None:
    description = view.get("tool_descriptions", {}).get(name)
    if description is not None:
        # the MCP layers add the responses again after the summary
        operation["summary"] = description.split(RESPONSES_MARKER)[0]
        operation.pop("description", None)

    edits: list[tuple[str, Callable[[dict], dict]]] = []
    for param, text in view.get("param_descriptions", {}).get(name, {}).items():
        edits.append((param, lambda p, text=text: {**p, "description": text}))
    for param, values in view.get("param_enums", {}).get(name, {}).items():
        edits.append((param, lambda p, values=values: {**p, "enum": list(values)}))
    for param, fmt in view.get("param_formats", {}).get(name, {}).items():
        edits.append((param, lambda p, fmt=fmt: formatted_schema(p, fmt)))
    for param, edit in edits:
        prop = _property(operation, components, param)
        if prop is not None:
            _set_property(operation, components, param, edit(prop))

    for param, schema in view.get("extra_params", {}).get(name, {}).items():
        if _property(operation, components, param) is not None:
            continue
        # a query parameter the route does not declare, so the server ignores it
        operation.setdefault("parameters", []).append(
            {
                "name": param,
                "in": "query",
                "required": False,
                "schema": {k: v for k, v in schema.items() if k != "description"},
                "description": schema.get("description", ""),
            }
        )


def present_openapi(doc: dict, view: dict) -> dict:
    """The OpenAPI document with its operations presented as the view says.

    view: a ToolView as a dict.
    """
    doc = copy.deepcopy(doc)
    components = doc.get("components", {}).get("schemas", {})
    paths = doc.setdefault("paths", {})
    for path_item in paths.values():
        for operation in path_item.values():
            if isinstance(operation, dict) and operation.get("operationId"):
                _present_operation(
                    operation, operation["operationId"], view, components
                )
    for tool in view.get("distractor_tools", []):
        schema = copy.deepcopy(tool.get("inputSchema") or {})
        paths[DISTRACTOR_PATH.format(tool_name=tool["name"])] = {
            "post": {
                "operationId": tool["name"],
                # without responses, the MCP layers list the summary as it is
                "summary": tool.get("description", ""),
                "requestBody": {"content": {"application/json": {"schema": schema}}},
            }
        }
    return doc


# -------------------------------------------------------------- the server


@dataclass(frozen=True)
class FormattedRoute:
    """A route whose parameters are presented in another format."""

    tool: str
    methods: frozenset[str]
    path_regex: re.Pattern[str]
    # parameter -> (where it is sent: "path", "query" or "body", format name)
    params: dict[str, tuple[str, str]]


class FormatRejected(Exception):
    pass


PATH_PARAM = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*)(?::[A-Za-z_]+)?}")


def presented_path_regex(path: str, formatted: set[str]) -> re.Pattern[str]:
    """A pattern of the route's path as an agent's calls write it.

    A presented format may write a value with "/" (YYYY/MM/DD), which the
    route's own pattern does not take in a path parameter.
    """
    pattern = "^"
    end = 0
    for m in PATH_PARAM.finditer(path):
        name = m.group(1)
        pattern += re.escape(path[end : m.start()])
        pattern += f"(?P<{name}>.+?)" if name in formatted else f"(?P<{name}>[^/]+)"
        end = m.end()
    return re.compile(pattern + re.escape(path[end:]) + "$")


def formatted_routes(app: Any, param_formats: dict) -> list[FormattedRoute]:
    """The routes of the tools with parameters presented in another format."""
    found = []
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        tool = route.operation_id or route.unique_id
        formats = param_formats.get(tool)
        if not formats:
            continue
        where = {f.alias: "path" for f in dependant.path_params}
        where.update({f.alias: "query" for f in dependant.query_params})
        found.append(
            FormattedRoute(
                tool=tool,
                methods=frozenset(route.methods or ()),
                path_regex=presented_path_regex(
                    route.path, {p for p in formats if where.get(p) == "path"}
                ),
                params={p: (where.get(p, "body"), fmt) for p, fmt in formats.items()},
            )
        )
    return found


def _server_value(route: FormattedRoute, param: str, fmt: str, value: Any) -> Any:
    try:
        return VALUE_FORMATS[fmt].to_server(value)
    except FORMAT_ERRORS:
        raise FormatRejected(format_rejection(route.tool, param, value, fmt)) from None


def _url_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class ValueFormatMiddleware:
    """Converts parameters presented in another format back to the server's form."""

    def __init__(self, app: Any, routes: list[FormattedRoute]) -> None:
        self.app = app
        self.routes = routes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for route in self.routes:
            if scope["method"] not in route.methods:
                continue
            match = route.path_regex.match(scope["path"])
            if match is not None:
                break
        else:
            await self.app(scope, receive, send)
            return
        try:
            scope = self._converted_url(route, match, scope)
            receive = await self._converted_body(route, scope, receive)
        except FormatRejected as e:
            await _send_error(send, str(e))
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _converted_url(
        route: FormattedRoute, match: re.Match[str], scope: dict
    ) -> dict:
        path = scope["path"]
        spans = sorted(
            (
                (match.span(param), param, fmt)
                for param, (where, fmt) in route.params.items()
                if where == "path" and param in match.groupdict()
            ),
            reverse=True,
        )
        for (start, end), param, fmt in spans:
            value = _url_text(_server_value(route, param, fmt, path[start:end]))
            path = path[:start] + value + path[end:]

        query = parse_qsl(
            scope.get("query_string", b"").decode(), keep_blank_values=True
        )
        converted = []
        for key, value in query:
            where, fmt = route.params.get(key, ("", ""))
            if where == "query":
                value = _url_text(_server_value(route, key, fmt, value))
            converted.append((key, value))
        return {
            **scope,
            "path": path,
            "raw_path": path.encode(),
            "query_string": urlencode(converted).encode(),
        }

    @staticmethod
    async def _converted_body(route: FormattedRoute, scope: dict, receive: Any) -> Any:
        body_params = {
            p: fmt for p, (where, fmt) in route.params.items() if where == "body"
        }
        if not body_params:
            return receive
        chunks = []
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                return _replaying(message, receive)
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        try:
            data = json.loads(body) if body else None
        except ValueError:
            data = None
        if isinstance(data, dict):
            for param, fmt in body_params.items():
                if data.get(param) is not None:
                    data[param] = _server_value(route, param, fmt, data[param])
            body = json.dumps(data).encode()
            headers = [
                (k, v) for k, v in scope["headers"] if k.lower() != b"content-length"
            ]
            headers.append((b"content-length", str(len(body)).encode()))
            scope["headers"] = headers
        return _replaying(
            {"type": "http.request", "body": body, "more_body": False}, receive
        )


def _replaying(first: dict, receive: Any) -> Any:
    """A receive that returns first once, then what receive returns."""
    pending = [first]

    async def replay() -> dict:
        if pending:
            return pending.pop()
        return await receive()

    return replay


async def _send_error(send: Any, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": FORMAT_ERROR_STATUS,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def distractor_call(tool_name: str) -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=DISTRACTOR_STATUS,
        content={"detail": f"{tool_name} is not available in this service."},
    )


def install(app: Any, view: dict) -> None:
    """Make the MCP layers serve the app's tools as the view presents them.

    Call it after every route is added and before an MCP layer is imported.
    """
    from fastapi.openapi import utils

    unpresented = utils.get_openapi

    def get_openapi(*args: Any, **kwargs: Any) -> dict:
        return present_openapi(unpresented(*args, **kwargs), view)

    utils.get_openapi = get_openapi  # ty: ignore[invalid-assignment]
    if view.get("distractor_tools"):
        app.add_api_route(
            DISTRACTOR_PATH, distractor_call, methods=["POST"], include_in_schema=False
        )
    routes = formatted_routes(app, view.get("param_formats", {}))
    if routes:
        app.add_middleware(ValueFormatMiddleware, routes=routes)
