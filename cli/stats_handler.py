import threading
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.last_activity_time: Optional[float] = None
        self.current_llm_model: Optional[str] = None
        self.current_tool_name: Optional[str] = None

    def _touch(self) -> None:
        """Update the last-activity timestamp. Must be called under lock."""
        self.last_activity_time = time.time()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1
            self.current_llm_model = (
                serialized.get("name")
                or serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model", "unknown")
            )
            self.current_tool_name = None
            self._touch()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1
            self.current_llm_model = (
                serialized.get("name")
                or serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model", "unknown")
            )
            self.current_tool_name = None
            self._touch()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        with self._lock:
            if usage_metadata:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)
            self.current_llm_model = None
            self._touch()

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1
            self.current_tool_name = serialized.get("name", "unknown")
            self._touch()

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Clear current tool name when tool ends."""
        with self._lock:
            self.current_tool_name = None
            self._touch()

    def get_activity(self) -> Dict[str, Any]:
        """Return current activity info: model, tool, last-activity seconds ago."""
        with self._lock:
            seconds_ago = None
            if self.last_activity_time is not None:
                seconds_ago = time.time() - self.last_activity_time
            return {
                "current_llm": self.current_llm_model,
                "current_tool": self.current_tool_name,
                "seconds_since_activity": seconds_ago,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }
