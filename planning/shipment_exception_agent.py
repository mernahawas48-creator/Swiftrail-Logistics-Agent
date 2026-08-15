from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .environment import Environment
from .reflexion import ReflexionResult, reflexion
from .self_refine import ReflectionResult, reflect_and_refine


_ID_RE = re.compile(r"\b(?:shipment|employee)\s*#?\s*(\d+)\b", re.I)


@dataclass(slots=True)
class PlanningRun:
    method: str
    shipment_id: int
    employee_id: int
    result: ReflectionResult | ReflexionResult


class ShipmentExceptionResolutionAgent:
    """Planning-layer adapter for the existing Swiftrail agent.

    This is intentionally separate from the legacy AgentLoop because the
    submitted repository's agent_loop.py is incomplete. It provides a stable
    integration point for the planning concern without duplicating MCP or DB
    implementations.
    """

    def __init__(self, llm: Any, *, critic_llm: Any | None = None):
        self.llm = llm
        self.critic_llm = critic_llm or llm

    @staticmethod
    def extract_ids(request: str) -> tuple[int, int]:
        matches = list(_ID_RE.finditer(request))
        if len(matches) < 2:
            raise ValueError(
                "Request must contain shipment and employee IDs, e.g. "
                "'Resolve shipment 3 for employee 1'."
            )
        shipment_id = int(matches[0].group(1))
        employee_id = int(matches[1].group(1))
        if "employee" not in matches[0].group(0).lower() and "shipment" not in matches[0].group(0).lower():
            raise ValueError("Could not identify shipment/employee IDs safely")
        return shipment_id, employee_id

    def run_self_refine(self, request: str, draft: str) -> PlanningRun:
        shipment_id, employee_id = self.extract_ids(request)
        environment = Environment(shipment_id=shipment_id, employee_id=employee_id)
        result = reflect_and_refine(
            request,
            draft,
            self.llm,
            critic_llm=self.critic_llm,
            environment=environment,
        )
        return PlanningRun("self_refine", shipment_id, employee_id, result)

    def run_reflexion(
        self,
        request: str,
        *,
        max_trials: int = 3,
        memory_size: int = 3,
    ) -> PlanningRun:
        shipment_id, employee_id = self.extract_ids(request)
        environment = Environment(shipment_id=shipment_id, employee_id=employee_id)
        result = reflexion(
            request,
            self.llm,
            environment,
            max_trials=max_trials,
            memory_size=memory_size,
            critic_llm=self.critic_llm,
        )
        return PlanningRun("reflexion", shipment_id, employee_id, result)
