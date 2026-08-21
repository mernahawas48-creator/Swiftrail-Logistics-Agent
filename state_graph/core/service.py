from __future__ import annotations

from typing import Any

from state_graph.core.engine import GraphEngine
from state_graph.core.state import SharedGraphState
from state_graph.core.types import HITLStatus, TicketStatus


class GraphService:
    """Stable backend facade consumed by the agent and platform teams."""

    def __init__(self, engine: GraphEngine) -> None:
        self.engine = engine

    def start_run(
        self, graph_name: str, input_data: dict[str, Any]
    ) -> SharedGraphState:
        return self.engine.start(graph_name, input_data)

    def get_run(self, run_id: str) -> SharedGraphState:
        return self.engine._require_run(run_id)

    def submit_external_input(
        self, run_id: str, payload: dict[str, Any]
    ) -> SharedGraphState:
        return self.engine.resume_external(run_id, payload)

    def pending_hitl_tasks(self) -> list[dict[str, Any]]:
        return self.engine.store.list_hitl_tasks(HITLStatus.PENDING)

    def resolve_hitl(
        self,
        task_id: str,
        *,
        approved: bool,
        note: str,
        admin_employee_id: int,
    ) -> SharedGraphState:
        return self.engine.resolve_hitl(
            task_id,
            approved=approved,
            note=note,
            admin_employee_id=admin_employee_id,
        )

    def tickets(
        self, status: TicketStatus | None = None
    ) -> list[dict[str, Any]]:
        return self.engine.store.list_tickets(status)

    def investigate_ticket(self, ticket_id: str) -> dict[str, Any]:
        return self.engine.investigate_ticket(ticket_id)

    def resolve_ticket(
        self, ticket_id: str, *, resolution_note: str
    ) -> SharedGraphState:
        return self.engine.resolve_ticket(
            ticket_id, resolution_note=resolution_note
        )
