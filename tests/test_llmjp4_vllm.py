import importlib
import sys

import pytest

MAKE_ID = "def make_tool_call_id(id_type='random', func_name=None, idx=None):\n    return 'fake-id'\n"
# what llm_jp_vllm.llmjp4 does on import that breaks with vLLM 0.32
PLUGIN = "from vllm.entrypoints.chat_utils import make_tool_call_id\n"


@pytest.fixture
def fake_packages(tmp_path, monkeypatch):
    """Stand-ins for vllm and llm_jp_vllm, written as files under tmp_path."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("vllm", "llm_jp_vllm") or name == "awm.llmjp4_vllm":
            monkeypatch.delitem(sys.modules, name)
    files = {
        "vllm/__init__.py": "",
        "vllm/entrypoints/__init__.py": "",
        "vllm/renderers/__init__.py": "",
        "vllm/renderers/chat_utils.py": MAKE_ID,
        "llm_jp_vllm/__init__.py": "",
        "llm_jp_vllm/llmjp4/__init__.py": PLUGIN,
    }
    for path, text in files.items():
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_plugin_loads_with_make_tool_call_id_from_its_new_module(fake_packages):
    importlib.import_module("awm.llmjp4_vllm")
    plugin = sys.modules["llm_jp_vllm.llmjp4"]
    assert plugin.make_tool_call_id() == "fake-id"


def test_an_existing_legacy_module_is_left_alone(fake_packages):
    (fake_packages / "vllm/entrypoints/chat_utils.py").write_text(
        MAKE_ID.replace("fake-id", "legacy-id")
    )
    importlib.import_module("awm.llmjp4_vllm")
    assert sys.modules["llm_jp_vllm.llmjp4"].make_tool_call_id() == "legacy-id"
