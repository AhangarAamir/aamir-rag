"""Print each agent tool call on the process console."""

from __future__ import annotations

from typing import Any

from agno.utils.log import log_info


def _log_tool_call(call: Any) -> None:
    agent = getattr(getattr(call, "function", None), "_agent", None)
    agent_name = getattr(agent, "name", None) or "Agent"
    try:
        rendered = call.get_call_str()
    except Exception:
        rendered = getattr(getattr(call, "function", None), "name", "tool")
    log_info(f"{agent_name} tool call: {rendered}")


def install_tool_call_logging() -> None:
    """Log tool calls from both the sync and async agent run paths."""
    from agno.tools.function import FunctionCall

    if getattr(FunctionCall, "_legal_tool_log", False):
        return

    original_execute = FunctionCall.execute
    original_aexecute = FunctionCall.aexecute

    def execute(self, *args, **kwargs):
        _log_tool_call(self)
        return original_execute(self, *args, **kwargs)

    async def aexecute(self, *args, **kwargs):
        _log_tool_call(self)
        return await original_aexecute(self, *args, **kwargs)

    FunctionCall.execute = execute  # type: ignore[method-assign]
    FunctionCall.aexecute = aexecute  # type: ignore[method-assign]
    FunctionCall._legal_tool_log = True  # type: ignore[attr-defined]
