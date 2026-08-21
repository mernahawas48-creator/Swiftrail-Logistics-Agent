from typing import Any

from .base import ContextStrategy


class SlidingWindow(ContextStrategy):
    """Keep only the most recent messages."""

    def __init__(self, max_messages: int = 10):
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        self.max_messages = max_messages

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages[-self.max_messages:]
