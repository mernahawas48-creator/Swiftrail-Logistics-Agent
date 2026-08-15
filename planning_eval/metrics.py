from __future__ import annotations

from dataclasses import dataclass, asdict
import time


@dataclass(slots=True)
class RunMetrics:
    method: str
    success: bool
    llm_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000
