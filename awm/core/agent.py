"""
Simple MCP Agent
Please refer to https://github.com/Snowflake-Labs/agent-world-model for more details.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent

from mcp.types import TextContent
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
)

from awm.core.server import serve_scenario
from awm.core.tool_view import PresentedTools, TaskMeta, ToolCallRejected
from awm.tools import (
    find_scenario_entry,
    mcp_call_tool,
    mcp_list_tools,
    resolve_llm_config,
    tools_json_save,
    tools_jsonl_load,
    tools_robust_json_loads,
)

logger = logging.getLogger(__name__)

for _name in ["mcp", "httpx", "httpcore", "anyio"]:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
    logging.getLogger(_name).propagate = False
    logging.getLogger(_name).handlers = []


@dataclass
class Config:
    # Task description for the agent (optional if --scenario and --task_id are provided)
    task: str | None = None
    # Scenario name (e.g. e_commerce_33). Used with --task_id to look up tasks.
    scenario: str | None = None
    # Task index within the scenario (0-based). Used with --scenario.
    task_id: int | None = None
    # MCP server URL (streamable HTTP transport). Auto-started if --scenario is provided.
    mcp_url: str | None = None
    # OpenAI-compatible API URL (also supports Azure OpenAI via env vars)
    api_url: str | None = None
    # Model name
    model: str | None = None
    # Max agent loop iterations
    max_iterations: int = 30
    # Generation temperature
    temperature: float = 1.0
    # Max completion tokens; none leaves it to the server (vLLM: the rest of the context)
    max_tokens: int | None = 2048
    # Output directory for saving trajectory. Defaults to outputs/agents/<timestamp>
    output_dir: str | None = None
    # Path to generated environments file
    envs_path: str = "./outputs/gen_envs.jsonl"
    # Path to generated tasks file
    tasks_path: str = "./outputs/gen_tasks.jsonl"
    # Path to generated db schemas file
    db_path: str = "./outputs/gen_db.jsonl"
    # Path to generated sample data file
    sample_path: str = "./outputs/gen_sample.jsonl"
    # Verbose logging
    verbose: bool = True
    # Pass list_tools and call_tool as tools of the chat API and read the calls the
    # server parsed (vLLM --tool-call-parser), instead of <tool_call> tags in the reply
    native_tool_calls: bool = False
    # reasoning_effort of the chat API (low, medium, high), for models whose chat
    # template reads it (e.g. llm-jp-4.1); none leaves it to the template
    reasoning_effort: str | None = None

    def pre_process(self):
        if self.task is None:
            print(self)
            assert self.scenario is not None and self.task_id is not None, (
                "task is None, you must specify the scenario and task_id to lookup the task"
            )


# texts of the two functions the agent calls, shared by the system prompt with
# <tool_call> tags and by the tools of the chat API (native_tool_calls)
LIST_TOOLS_DOC = "List all available MCP tools for the current environment to help you finish the user task."
LIST_TOOLS_OUTPUT = "A list of MCP environment-specific tools and their descriptions"
CALL_TOOL_DOC = "Call a MCP environment-specific tool"
CALL_TOOL_OUTPUT = "the result of the <tool_name> tool call"
TOOL_NAME_DOC = "the tool name in the list_tools output"
CALL_TOOL_ARGUMENTS_DOC = (
    "the arguments for calling <tool_name>. You must pass a valid JSON string without any "
    "markdown fences or additional commentary. This JSON str will be parsed by the tool and "
    "executed. You can pass an empty JSON str if no arguments are required by <tool_name>."
)

PROMPT_INTRO = "You are at a MCP environment. You need to call MCP tools to assist with the user query. At each step, you can only call one function. You have already logged in, and your user id is 1 if required for the MCP tool."
PROMPT_RULES = "You should always call list_tools function first to get the available tools, and should only call it once. You should always directly output the answer or summary at the final step instead of calling any function."

NATIVE_TOOLS: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": LIST_TOOLS_DOC,
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": f"{CALL_TOOL_DOC}. Returns {CALL_TOOL_OUTPUT}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "description": TOOL_NAME_DOC},
                    "arguments": {
                        "type": "string",
                        "description": CALL_TOOL_ARGUMENTS_DOC,
                    },
                },
                "required": ["tool_name", "arguments"],
            },
        },
    },
]


def get_native_system_prompt() -> str:
    return (
        "# MCP Tools\n\n"
        f"{PROMPT_INTRO}\n\n"
        "You are provided with two functions: list_tools, which lists the MCP environment-specific tools, and call_tool, which calls one of them.\n\n"
        f"{PROMPT_RULES}"
    )


def get_system_prompt() -> str:
    tools_str = "\n".join(
        [
            "1. list_tools",
            f"    - Description: {LIST_TOOLS_DOC}",
            "    - Arguments: None",
            f"    - Output: {LIST_TOOLS_OUTPUT}",
            "",
            "2. call_tool",
            f"    - Description: {CALL_TOOL_DOC}",
            "    - Arguments:",
            f"        - tool_name: str, required, {TOOL_NAME_DOC}",
            f"        - arguments: str, required, {CALL_TOOL_ARGUMENTS_DOC}",
            f"    - Output: {CALL_TOOL_OUTPUT.capitalize()}",
        ]
    )

    return dedent(f"""\
        # MCP Toolss

        {PROMPT_INTRO}

        You are provided with TWO functions within <tools></tools> XML tags:
        <tools>
        {tools_str}
        </tools>

        {PROMPT_RULES}

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {{"name": <function-name>, "arguments": <args-json-object>}}
        </tool_call>

        Example Function Call #1:
        <tool_call>
        {{"name": "list_tools", "arguments": null}}
        </tool_call>

        Example Function Call #2:
        <tool_call>
        {{"name": "call_tool", "arguments": {{"tool_name": "get_weather", "arguments": "{{"city": "Beijing"}}"}}}}
        </tool_call>""")


def parse_tool_calls(content: str) -> list[dict]:
    tool_calls = []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        data = tools_robust_json_loads(match.strip())
        if not data:
            logger.warning(f"Failed to parse tool call JSON: {match[:100]}")
            continue

        # Handle list wrapping: [{"name": ..., "arguments": ...}]
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                continue

        if not isinstance(data, dict):
            continue

        name = data.get("name", "")
        arguments = data.get("arguments", {})

        if name.startswith("mcp_tool_"):
            arguments = {
                "tool_name": name,
                "arguments": arguments if arguments else {},
            }
            name = "call_tool"

        tool_calls.append(
            {
                "id": f"call_{int(time.time() * 1000)}_{i}",
                "name": name,
                "arguments": arguments,
            }
        )

    return tool_calls


def parse_call_tool_arguments(arguments: dict | str | None) -> tuple[str, dict]:
    if isinstance(arguments, str):
        arguments = tools_robust_json_loads(arguments)
    if not isinstance(arguments, dict):
        return "", {}

    tool_name = arguments.get("tool_name", "")
    inner_args = arguments.get("arguments", {})

    tool_name = tool_name.removeprefix("mcp_tool_")

    if isinstance(inner_args, str):
        parsed = tools_robust_json_loads(inner_args) if inner_args.strip() else {}
        if isinstance(parsed, dict):
            inner_args = parsed
        else:
            inner_args = {}

    if not isinstance(inner_args, dict):
        inner_args = {}

    return tool_name, inner_args


def format_tools_for_response(tools: list[dict]) -> str:

    def format_input_schema(
        schema: dict, indent_level: int = 6, parent_required: list | None = None
    ) -> str:
        if not schema:
            return ""

        result = ""
        indent_str = " " * indent_level
        properties = schema.get("properties", {})
        required_fields = (
            parent_required
            if parent_required is not None
            else schema.get("required", [])
        )

        for prop_name, prop_info in properties.items():
            is_required = prop_name in required_fields
            required_str = " (required)" if is_required else " (optional)"
            prop_type = prop_info.get("type", "unknown")
            description = prop_info.get("description", "")
            default = prop_info.get("default")
            enum_values = prop_info.get("enum")
            nested_properties = prop_info.get("properties", {})
            nested_required = prop_info.get("required", [])

            result += f"{indent_str}- {prop_name}: {prop_type}{required_str}\n"
            if description:
                result += f"{indent_str}  Description: {description}\n"
            if default is not None:
                result += f"{indent_str}  Default: {default}\n"
            if enum_values:
                result += f"{indent_str}  Allowed values: {enum_values}\n"

            if prop_type == "object" and nested_properties:
                result += f"{indent_str}  Properties:\n"
                nested_schema = {
                    "properties": nested_properties,
                    "required": nested_required,
                }
                result += format_input_schema(
                    nested_schema, indent_level + 4, nested_required
                )

        return result

    # filter out list_tools meta-tool
    actual_tools = [t for t in tools if t.get("name") != "list_tools"]

    docs_text = f"Available MCP Tools ({len(actual_tools)} tools):\n"
    docs_text += "=" * 80 + "\n\n"

    for i, tool in enumerate(actual_tools, 1):
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", tool.get("input_schema", {}))

        # add mcp_tool_ prefix
        mcp_name = f"mcp_tool_{name}" if not name.startswith("mcp_tool_") else name

        # parse description for multi-line
        desc_lines = description.split("\n")
        first_line = desc_lines[0].strip() if desc_lines else "No description"
        more_desc = (
            "\n".join(line.strip() for line in desc_lines[1:]).strip()
            if len(desc_lines) > 1
            else ""
        )

        docs_text += f"{i}. {mcp_name}\n"
        docs_text += f"   Description: {first_line}\n"
        if more_desc:
            for line in more_desc.split("\n"):
                if line.strip():
                    docs_text += f"   {line}\n"

        if input_schema and input_schema.get("properties"):
            docs_text += "   Parameters:\n"
            docs_text += format_input_schema(input_schema)
        else:
            docs_text += "   Parameters: None\n"

        docs_text += "\n"

    return docs_text.strip()


class MCPToolExecutor:
    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: list[dict] = []

    async def list_tools(self) -> list[dict]:
        tools = await mcp_list_tools(self.mcp_url, self.timeout)
        self._tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.input_schema or {},
            }
            for t in tools
        ]
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = await mcp_call_tool(self.mcp_url, tool_name, arguments, self.timeout)
        text = "\n".join(
            c.text if isinstance(c, TextContent) else str(c) for c in result.content
        )
        if result.is_error:
            return f"Error: {text}"
        return text


def to_api_messages(
    messages: list[dict], native_tool_calls: bool, use_vllm_extras: bool
) -> list[ChatCompletionMessageParam]:
    """The agent's messages in the chat API format."""
    api_messages: list[ChatCompletionMessageParam] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            api_messages.append({"role": "system", "content": content})
        elif role == "user":
            api_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if native_tool_calls and msg.get("tool_calls"):
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(
                                        tc["arguments"] or {}, ensure_ascii=False
                                    ),
                                },
                            }
                            for tc in msg["tool_calls"]
                        ],
                    }
                )
            else:
                api_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            if native_tool_calls or use_vllm_extras:
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", "call_unknown"),
                        "content": content,
                    }
                )
            else:
                # For standard OpenAI API, tool role requires tool_calls in the assistant message.
                # Since we use XML-based tool calls, send tool responses as user messages.
                api_messages.append(
                    {"role": "user", "content": f"Tool response:\n{content}"}
                )
    return api_messages


def parse_native_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None,
) -> list[dict]:
    """Tool calls the server parsed, in the form parse_tool_calls returns."""
    parsed = []
    for tc in tool_calls or []:
        # NATIVE_TOOLS are all function tools, so other kinds of call cannot match them
        if not isinstance(tc, ChatCompletionMessageFunctionToolCall):
            continue
        arguments = tc.function.arguments
        data = tools_robust_json_loads(arguments) if arguments.strip() else {}
        if not isinstance(data, dict):
            logger.warning(f"Failed to parse tool call arguments: {arguments[:100]}")
            data = {}
        parsed.append({"id": tc.id, "name": tc.function.name, "arguments": data})
    return parsed


async def generate_response(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    config: Config,
    use_vllm_extras: bool = False,
) -> tuple[str, list[dict]]:
    api_messages = to_api_messages(messages, config.native_tool_calls, use_vllm_extras)

    template_kwargs: dict[str, bool | str] = {"enable_thinking": True}
    optional_args: dict = {}
    if config.native_tool_calls:
        optional_args["tools"] = NATIVE_TOOLS
        optional_args["tool_choice"] = "auto"
    if config.reasoning_effort is not None:
        optional_args["reasoning_effort"] = config.reasoning_effort
        # the template variable too, in case the server does not forward the field to it
        template_kwargs["reasoning_effort"] = config.reasoning_effort
    extra_body = (
        {
            "add_generation_prompt": True,
            "min_tokens": 16,
            "chat_template_kwargs": template_kwargs,
        }
        if use_vllm_extras
        else None
    )

    response = await client.chat.completions.create(
        model=model_name,
        messages=api_messages,
        max_completion_tokens=config.max_tokens,
        temperature=config.temperature,
        extra_body=extra_body,
        **optional_args,
    )

    message = response.choices[0].message
    content = message.content or ""
    if config.native_tool_calls:
        tool_calls = parse_native_tool_calls(message.tool_calls)
    else:
        tool_calls = parse_tool_calls(content)

    return content, tool_calls


async def run_agent(config: Config):
    # Resolve scenario/task
    scenario = config.scenario
    task_id = config.task_id
    # settings of an augmented task (hidden tools, scripted follow-up turns)
    task_meta: TaskMeta | None = None

    if config.task:
        task = config.task

    elif scenario and task_id is not None:
        tasks_data = tools_jsonl_load(config.tasks_path)
        entry = find_scenario_entry(tasks_data, scenario)
        if not entry:
            raise ValueError(f"Scenario '{scenario}' not found in {config.tasks_path}")
        tasks = entry["tasks"]
        if task_id < 0 or task_id >= len(tasks):
            raise ValueError(
                f"task_id {task_id} out of range (0-{len(tasks) - 1}) for scenario {scenario}"
            )
        task = tasks[task_id]
        metas = entry.get("task_meta") or []
        if task_id < len(metas) and metas[task_id] is not None:
            task_meta = TaskMeta.model_validate(metas[task_id])
        logger.info(
            f"Resolved task from scenario={scenario}, task_id={task_id}: {task}"
        )

    else:
        raise ValueError(
            "Either --task or --scenario (with optional --task_id) must be provided"
        )

    # Resolve LLM config
    api_url, api_key, model = resolve_llm_config(
        api_url_override=config.api_url,
        model_override=config.model,
    )
    provider = os.environ.get("AWM_SYN_LLM_PROVIDER", "vllm").lower()
    use_vllm_extras = provider == "vllm" or (
        "localhost" in api_url and "openai.azure.com" not in api_url and "v1" in api_url
    )
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_dir:
        output_dir = config.output_dir
    else:
        run_name = f"{timestamp}"
        if scenario:
            run_name = f"{timestamp}_{scenario}_task_{task_id}"
        output_dir = os.path.join("outputs", "agents", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup database and MCP server if scenario is provided
    db_file_path = None
    servers = contextlib.AsyncExitStack()

    if scenario and not config.mcp_url:
        server = await servers.enter_async_context(
            serve_scenario(
                scenario,
                config.envs_path,
                output_dir,
                db_schema_path=config.db_path,
                sample_path=config.sample_path,
            )
        )
        mcp_url = server.url
        db_file_path = server.db_path
    else:
        mcp_url = config.mcp_url
        if not mcp_url:
            raise ValueError("--mcp_url is required when --scenario is not provided")

    try:
        logger.info("=" * 80)
        logger.info("MCP Agent")
        logger.info("=" * 80)
        logger.info(f"Task: {task}")
        logger.info(f"MCP Server: {mcp_url}")
        logger.info(f"LLM API: {api_url}")
        logger.info(f"Model: {model}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)

        mcp = MCPToolExecutor(mcp_url)
        llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_url,
        )

        logger.info("Connecting to MCP server and fetching tools...")
        tools = await mcp.list_tools()
        logger.info(f"Loaded {len(tools)} tools from MCP server")
        tools_json_save(tools, os.path.join(output_dir, "tools.json"))
        presented = PresentedTools(tools, task_meta.hidden_tools if task_meta else ())
        tools_response_text = format_tools_for_response(presented.tools)

        pending_turns = list(task_meta.turns) if task_meta else []
        turn = 0
        turn_dbs: list[str] = []
        tool_call_log: list[dict] = []

        messages: list[dict] = [
            {
                "role": "system",
                "content": get_native_system_prompt()
                if config.native_tool_calls
                else get_system_prompt(),
            },
            {"role": "user", "content": task},
        ]

        # Trajectory recording
        trajectory: list[dict] = []

        iteration = 0
        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{config.max_iterations} ---")

            content, tool_calls = await generate_response(
                llm_client,
                model,
                messages,
                config,
                use_vllm_extras=use_vllm_extras,
            )

            if config.verbose:
                preview = content[:500] + "..." if len(content) > 500 else content
                logger.info(f"Assistant ({len(content)} chars): {preview}")
                logger.info(f"Tool calls: {len(tool_calls)}")

            assistant_message: dict = {"role": "assistant", "content": content}
            if config.native_tool_calls and tool_calls:
                # only the first call runs, and the chat API needs a reply to every call it sees
                assistant_message["tool_calls"] = tool_calls[:1]
            messages.append(assistant_message)

            # no tool calls -> the turn ends; scripted follow-ups continue the dialogue
            if not tool_calls and pending_turns:
                trajectory.append(
                    {
                        "iteration": iteration,
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [],
                        "turn": turn,
                    }
                )
                if db_file_path and os.path.exists(db_file_path):
                    turn_db = f"turn_{turn}.db"
                    shutil.copy2(db_file_path, os.path.join(output_dir, turn_db))
                    turn_dbs.append(turn_db)
                next_turn = pending_turns.pop(0)
                turn += 1
                if next_turn.shown_tools:
                    presented.show(next_turn.shown_tools)
                    tools_response_text = format_tools_for_response(presented.tools)
                logger.info(f"Turn {turn} user message: {next_turn.user}")
                messages.append({"role": "user", "content": next_turn.user})
                trajectory.append(
                    {"role": "user", "content": next_turn.user, "turn": turn}
                )
                continue

            # no tool calls -> task complete
            if not tool_calls:
                logger.info("No tool calls detected - task complete.")
                logger.info(f"\n{'=' * 40} Final Answer {'=' * 40}")
                logger.info(content)
                logger.info("=" * 80)
                trajectory.append(
                    {
                        "iteration": iteration,
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [],
                        "is_final": True,
                    }
                )
                break

            # process first tool call only (one per turn)
            tc = tool_calls[0]
            name = tc["name"]
            arguments = tc["arguments"]
            tool_call_id = tc["id"]

            if len(tool_calls) > 1:
                skipped = [t["name"] for t in tool_calls[1:]]
                logger.warning(
                    f"Multiple tool calls detected. Only executing first: {name}. Skipped: {skipped}"
                )

            # execute tool call
            if name == "list_tools":
                logger.info("Executing: list_tools")
                response_text = tools_response_text

            elif name == "call_tool":
                tool_name, tool_args = parse_call_tool_arguments(arguments)
                logger.info(
                    f"Executing: call_tool({tool_name}, {json.dumps(tool_args, ensure_ascii=False)[:200]})"
                )
                try:
                    presented.check(tool_name)
                    tool_call_log.append(
                        {"turn": turn, "tool_name": tool_name, "arguments": tool_args}
                    )
                    response_text = await mcp.call_tool(tool_name, tool_args)
                except ToolCallRejected as e:
                    tool_call_log.append(
                        {
                            "turn": turn,
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "rejected": True,
                        }
                    )
                    response_text = f"Error: {e}"
                except TimeoutError:
                    response_text = f"Error: Tool call timed out after {mcp.timeout}s"
                    logger.error(response_text)
                except Exception as e:  # noqa: BLE001
                    response_text = f"Error: {e}"
                    logger.error(f"Tool call failed: {e}")
            else:
                response_text = f"Error: Unknown tool '{name}'. Only 'list_tools' and 'call_tool' are available."

            if config.verbose:
                preview = (
                    response_text[:500] + "..."
                    if len(response_text) > 500
                    else response_text
                )
                logger.info(f"Tool response ({len(response_text)} chars): {preview}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": response_text,
                }
            )

            # Record trajectory entry
            trajectory.append(
                {
                    "iteration": iteration,
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [tc],
                    "tool_response": {
                        "tool_call_id": tool_call_id,
                        "content": response_text,
                    },
                }
            )
        else:
            logger.warning("Max iterations reached without completion.")

        logger.info("=" * 80)
        logger.info(f"Agent execution complete. Total iterations: {iteration}")
        logger.info("=" * 80)

        # Save trajectory and outputs
        trajectory_data = {
            "scenario": scenario,
            "task_id": task_id,
            "task": task,
            "model": model,
            "api_url": api_url,
            "max_iterations": config.max_iterations,
            "temperature": config.temperature,
            "total_iterations": iteration,
            "timestamp": timestamp,
            "trajectory": trajectory,
            "messages": messages,
            "task_meta": task_meta.model_dump() if task_meta else None,
            "tool_calls": tool_call_log,
            "turn_dbs": turn_dbs,
        }
        trajectory_path = os.path.join(output_dir, "trajectory.json")
        tools_json_save(trajectory_data, trajectory_path)
        logger.info(f"Saved trajectory to {trajectory_path}")

        # Save final database
        # a database built for the run is already output_dir/final.db
        final_db_path = os.path.join(output_dir, "final.db")
        if (
            db_file_path
            and os.path.exists(db_file_path)
            and not (
                os.path.exists(final_db_path)
                and os.path.samefile(db_file_path, final_db_path)
            )
        ):
            shutil.copy2(db_file_path, final_db_path)
            logger.info(f"Saved final database to {final_db_path}")

        logger.info(f"Run outputs saved to: {output_dir}")

    finally:
        # Cleanup MCP server
        await servers.aclose()


def run(config: Config):
    asyncio.run(run_agent(config))


if __name__ == "__main__":
    from simpleArgParser import parse_args

    config: Config = parse_args(Config)
    run(config)
