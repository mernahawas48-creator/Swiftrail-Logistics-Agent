from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RateExceptionState:
    """Durable state for a rate-exception approval workflow."""

    run_id: str
    shipment_id: int
    session_id: str
    employee_id: int = 3
    mcp_url: str = "http://127.0.0.1:8000/mcp"
    current_node: str = "START"
    shipment: dict[str, Any] | None = None
    rate_exception: dict[str, Any] | None = None
    policy_evidence: list[dict[str, Any]] = field(default_factory=list)
    discount_pct: float | None = None
    requires_human: bool = False
    admin_decision: str | None = None
    admin_note: str | None = None
    final_status: str | None = None
    error: str | None = None
    ticket_id: str | None = None
    ticket_status: str | None = None
    failed_node: str | None = None
    hitl_task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RateExceptionState":
        return cls(**data)
