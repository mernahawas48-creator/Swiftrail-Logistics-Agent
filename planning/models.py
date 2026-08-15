from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EnvironmentFeedback:
    success: bool
    score: float
    details: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
