from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CreditHoldRequest:
    customer_id: int
    session_id: str
    employee_id: int
    customer_claim: str | None = None

    @classmethod
    def from_input(cls, value: dict[str, Any]) -> CreditHoldRequest:
        try:
            request = cls(
                customer_id=int(value["customer_id"]),
                session_id=str(value["session_id"]).strip(),
                employee_id=int(value["employee_id"]),
                customer_claim=(str(value["customer_claim"]).strip() or None)
                if value.get("customer_claim") is not None
                else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Credit-hold graph input is incomplete or invalid.") from exc
        if request.customer_id < 1 or request.employee_id < 1:
            raise ValueError("Customer and employee IDs must be positive.")
        if len(request.session_id) < 8:
            raise ValueError("session_id must contain at least 8 characters.")
        return request
