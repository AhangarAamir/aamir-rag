"""OpenAI Responses model that keeps reasoning items with tool calls.

gpt-6-luna emits a reasoning item for every function call. The Responses
API rejects the next request when that function call is sent again without
the reasoning item. Agno only does this pairing for o3, o4-mini, and gpt-5.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from agno.models.message import Message
from agno.models.openai.responses import OpenAIResponses

try:
    from openai.types.responses import ResponseReasoningItem
except ImportError:  # pragma: no cover
    ResponseReasoningItem = None  # type: ignore[assignment]


def _is_reasoning_model(model_id: str) -> bool:
    name = (model_id or "").lower()
    return (
        name.startswith("o3")
        or name.startswith("o4-mini")
        or name.startswith("gpt-5")
        or name.startswith("gpt-6")
        or "luna" in name
    )


class LunaResponses(OpenAIResponses):
    """Responses model for gpt-5 and gpt-6 Luna, including tool-call continuations."""

    def _using_reasoning_model(self) -> bool:
        return _is_reasoning_model(self.id)

    def _parse_provider_response(self, response, **kwargs):
        model_response = super()._parse_provider_response(response, **kwargs)
        self._keep_reasoning_item(model_response, getattr(response, "output", None))
        return model_response

    def _parse_provider_response_delta(self, stream_event, assistant_message, tool_use):
        model_response, tool_use = super()._parse_provider_response_delta(
            stream_event, assistant_message, tool_use
        )
        if getattr(stream_event, "type", None) == "response.completed":
            output = getattr(getattr(stream_event, "response", None), "output", None)
            self._keep_reasoning_item(model_response, output)
        return model_response, tool_use

    @staticmethod
    def _keep_reasoning_item(model_response, output) -> None:
        if not output:
            return
        for item in output:
            if getattr(item, "type", None) != "reasoning":
                continue
            if model_response.provider_data is None:
                model_response.provider_data = {}
            model_response.provider_data["reasoning_output"] = item.model_dump(exclude_none=True)
            return

    def _format_messages(
        self,
        messages: List[Message],
        compress_tool_results: bool = False,
        tools: Optional[List[Union[Any, Dict[str, Any]]]] = None,
    ):
        formatted = super()._format_messages(
            messages, compress_tool_results=compress_tool_results, tools=tools
        )
        if not self._using_reasoning_model():
            return formatted
        if not any(isinstance(item, dict) and item.get("type") == "function_call" for item in formatted):
            return formatted
        reasoning_items = _reasoning_items(messages)
        if not reasoning_items:
            return formatted
        present = {
            getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
            for item in formatted
        }
        missing = [item for item in reasoning_items if getattr(item, "id", None) not in present]
        if not missing:
            return formatted
        paired: list = []
        inserted = False
        for item in formatted:
            if not inserted and isinstance(item, dict) and item.get("type") == "function_call":
                paired.extend(missing)
                inserted = True
            paired.append(item)
        return paired


def _reasoning_items(messages: List[Message]) -> list:
    if ResponseReasoningItem is None:
        return []
    items = []
    for message in messages:
        raw = (getattr(message, "provider_data", None) or {}).get("reasoning_output")
        if not raw:
            continue
        items.append(ResponseReasoningItem.model_validate(raw))
    return items


def build_model() -> LunaResponses:
    from legal_rag.config import OPENAI_MODEL

    return LunaResponses(id=OPENAI_MODEL)
