import asyncio
import json
from types import SimpleNamespace

from openai.types.chat import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
)

from awm.core import agent

CALL = {
    "id": "c1",
    "name": "call_tool",
    "arguments": {"tool_name": "list_products", "arguments": "{}"},
}


def function_call(call_id: str, name: str, arguments: str):
    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function={"name": name, "arguments": arguments},
    )


def test_native_messages_carry_the_tool_calls_and_their_replies():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    api = agent.to_api_messages(messages, native_tool_calls=True, use_vllm_extras=False)
    assert api[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "call_tool",
                    "arguments": json.dumps(CALL["arguments"]),
                },
            }
        ],
    }
    assert api[3] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_tag_messages_send_tool_replies_as_user_messages_outside_vllm():
    messages = [
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    api = agent.to_api_messages(
        messages, native_tool_calls=False, use_vllm_extras=False
    )
    assert api == [
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "user", "content": "Tool response:\nok"},
    ]


def test_parse_native_tool_calls():
    calls = [
        function_call("c1", "call_tool", json.dumps(CALL["arguments"])),
        function_call("c2", "list_tools", ""),
    ]
    assert agent.parse_native_tool_calls(calls) == [
        CALL,
        {"id": "c2", "name": "list_tools", "arguments": {}},
    ]


def test_parse_native_tool_calls_keeps_a_call_with_broken_arguments():
    calls = [
        function_call("c1", "call_tool", "[1, 2]"),
        ChatCompletionMessageCustomToolCall(
            id="c2", type="custom", custom={"name": "x", "input": "y"}
        ),
    ]
    assert agent.parse_native_tool_calls(calls) == [
        {"id": "c1", "name": "call_tool", "arguments": {}}
    ]
    assert agent.parse_native_tool_calls(None) == []


class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


def generate(config: agent.Config, message) -> tuple[FakeCompletions, tuple]:
    completions = FakeCompletions(message)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = asyncio.run(
        agent.generate_response(
            client,  # type: ignore[arg-type]
            "m",
            [{"role": "user", "content": "u"}],
            config,
            use_vllm_extras=True,
        )
    )
    return completions, result


def test_generate_response_reads_the_calls_the_server_parsed():
    message = SimpleNamespace(
        content="",
        tool_calls=[function_call("c1", "list_tools", "{}")],
    )
    config = agent.Config(native_tool_calls=True, reasoning_effort="medium")
    completions, result = generate(config, message)
    assert completions.kwargs["tools"] == agent.NATIVE_TOOLS
    assert completions.kwargs["reasoning_effort"] == "medium"
    template_kwargs = completions.kwargs["extra_body"]["chat_template_kwargs"]
    assert template_kwargs["reasoning_effort"] == "medium"
    assert result == ("", [{"id": "c1", "name": "list_tools", "arguments": {}}])


def test_generate_response_without_native_calls_reads_the_tags():
    reply = '<tool_call>{"name": "list_tools", "arguments": null}</tool_call>'
    message = SimpleNamespace(content=reply, tool_calls=None)
    completions, (content, tool_calls) = generate(agent.Config(), message)
    assert "tools" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs
    assert content == reply
    assert [tc["name"] for tc in tool_calls] == ["list_tools"]


def test_native_run_keeps_only_the_executed_call_in_the_history(tmp_path, monkeypatch):
    replies = iter(
        [
            ("", [CALL, {**CALL, "id": "c2"}]),
            ("done", []),
        ]
    )

    async def fake_generate(*args, **kwargs):
        return next(replies)

    class FakeMCP:
        timeout = 60.0

        def __init__(self, mcp_url: str, timeout: float = 60.0):
            pass

        async def list_tools(self) -> list[dict]:
            return [{"name": "list_products", "description": "", "inputSchema": {}}]

        async def call_tool(self, tool_name: str, arguments: dict) -> str:
            return "ok"

    monkeypatch.setattr(agent, "MCPToolExecutor", FakeMCP)
    monkeypatch.setattr(agent, "generate_response", fake_generate)
    config = agent.Config(
        task="list products",
        mcp_url="http://unused/mcp",
        model="m",
        api_url="http://unused/v1",
        output_dir=str(tmp_path / "run"),
        verbose=False,
        native_tool_calls=True,
    )
    asyncio.run(agent.run_agent(config))
    data = json.loads((tmp_path / "run" / "trajectory.json").read_text())
    assert data["messages"][0]["content"] == agent.get_native_system_prompt()
    assert data["messages"][2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [CALL],
    }
    assert data["messages"][3]["tool_call_id"] == "c1"
    assert data["tool_calls"] == [
        {"turn": 0, "tool_name": "list_products", "arguments": {}}
    ]


def test_both_tool_call_formats_describe_the_functions_alike():
    prompt = agent.get_system_prompt()
    for tool in agent.NATIVE_TOOLS:
        function = tool["function"]
        assert function["name"] in prompt
        properties = function.get("parameters", {}).get("properties")
        assert isinstance(properties, dict)
        for param in properties.values():
            assert param["description"] in prompt
    assert agent.PROMPT_INTRO in agent.get_native_system_prompt()
    assert agent.PROMPT_RULES in agent.get_native_system_prompt()
