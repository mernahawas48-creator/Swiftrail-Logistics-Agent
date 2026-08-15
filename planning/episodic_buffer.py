from __future__ import annotations

from collections import deque


class EpisodicReflectionBuffer:
    """Bounded verbal memory used only within one Reflexion run."""

    def __init__(self, max_size: int = 3) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self._items: deque[str] = deque(maxlen=max_size)

    def add(self, reflection: str) -> None:
        text = reflection.strip()
        if text:
            self._items.append(text)

    def items(self) -> list[str]:
        return list(self._items)

    def prompt_text(self) -> str:
        return "\n".join(f"- {item}" for item in self._items) or "- No prior trials."
