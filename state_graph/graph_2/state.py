from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RateExceptionRequest:
    """Validated input stored inside the shared graph-state envelope."""

    shipment_id: int
    session_id: str
    employee_id: int = 3

    @classmethod
    def from_input(cls, value: dict[str, Any]) -> RateExceptionRequest:
        shipment_id = int(value.get("shipment_id", 0))
        employee_id = int(value.get("employee_id", 0))
        session_id = str(value.get("session_id", "")).strip()
        if shipment_id < 1:
            raise ValueError("shipment_id must be positive.")
        if employee_id < 1:
            raise ValueError("employee_id must be positive.")
        if len(session_id) < 8:
            raise ValueError("session_id must contain at least 8 characters.")
        return cls(shipment_id, session_id, employee_id)


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
    def from_dict(cls, data: dict[str, Any]) -> RateExceptionState:
        return cls(**data)
