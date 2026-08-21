from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeliveryRecoveryRequest:
    shipment_id: int
    session_id: str
    employee_id: int
    failure_reason: str

    @classmethod
    def from_input(cls, value: dict[str, Any]) -> DeliveryRecoveryRequest:
        try:
            request = cls(
                shipment_id=int(value["shipment_id"]),
                session_id=str(value["session_id"]).strip(),
                employee_id=int(value["employee_id"]),
                failure_reason=str(value["failure_reason"]).strip(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Delivery recovery input is incomplete or invalid.") from exc
        if request.shipment_id < 1 or request.employee_id < 1:
            raise ValueError("Shipment and employee IDs must be positive.")
        if len(request.session_id) < 8:
            raise ValueError("session_id must contain at least 8 characters.")
        if len(request.failure_reason) < 10:
            raise ValueError("failure_reason must contain at least 10 characters.")
        return request
