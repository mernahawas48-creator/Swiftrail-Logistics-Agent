from typing import Any

from .base import ContextStrategy


class ZoneBasedPruning(ContextStrategy):
    """Keep important zones and prune older context."""

    def __init__(self, keep_recent_messages: int = 3):
        if keep_recent_messages < 1:
            raise ValueError("keep_recent_messages must be at least 1.")
        self.keep_recent_messages = keep_recent_messages

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system_messages = [
            message for message in messages
            if message.get("role") == "system"
        ]

        non_system_messages = [
            message for message in messages
            if message.get("role") != "system"
        ]

        recent_messages = non_system_messages[-self.keep_recent_messages:]

        return system_messages + recent_messages
