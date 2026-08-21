from abc import ABC, abstractmethod
from typing import Any


class ContextStrategy(ABC):
    """Common interface for all context management strategies."""

    @abstractmethod
    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prune or transform messages before sending them to the model."""
