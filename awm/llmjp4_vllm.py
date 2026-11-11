"""Register the llm-jp-4 parsers of llm-jp-vllm with the installed vLLM.

vLLM imports this module through --tool-parser-plugin and
--reasoning-parser-plugin (awm.llmjp4_vllm); importing llm_jp_vllm.llmjp4
registers the "llmjp4" tool and reasoning parsers.
"""

import importlib
import sys
import types

LEGACY_MODULE = "vllm.entrypoints.chat_utils"
CURRENT_MODULE = "vllm.renderers.chat_utils"


def install_chat_utils_alias() -> None:
    """Make make_tool_call_id importable from vllm.entrypoints.chat_utils."""
    # LIMITATION: llm-jp-vllm 0.1.1 imports make_tool_call_id from
    # vllm.entrypoints.chat_utils, which vLLM 0.32 no longer has; drop this
    # alias once llm-jp-vllm imports it from vllm.renderers.chat_utils.
    try:
        importlib.import_module(LEGACY_MODULE)
        return
    except ModuleNotFoundError:
        pass
    current = importlib.import_module(CURRENT_MODULE)
    alias = types.ModuleType(LEGACY_MODULE)
    alias.__dict__["make_tool_call_id"] = current.make_tool_call_id
    sys.modules[LEGACY_MODULE] = alias


install_chat_utils_alias()
importlib.import_module("llm_jp_vllm.llmjp4")
