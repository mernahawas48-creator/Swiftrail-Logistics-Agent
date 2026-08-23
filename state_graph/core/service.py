from __future__ import annotations

from typing import Any

from state_graph.core.engine import GraphEngine
from state_graph.core.exceptions import RunNotFoundError
from state_graph.core.state import SharedGraphState
from state_graph.core.types import HITLStatus, TicketStatus


class GraphService:
    """Stable backend facade consumed by the agent and platform teams."""

    def __init__(self, engine: GraphEngine) -> None:
        self.engine = engine
        graph_names = engine.registry.list()
        if len(graph_names) != 1:
            raise ValueError("GraphService requires exactly one registered graph.")
        self.graph_name = graph_names[0]

    def start_run(
        self,
        graph_name: str,
        input_data: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> SharedGraphState:
        if graph_name != self.graph_name:
            raise ValueError(f"GraphService is configured for {self.graph_name}.")
        return self.engine.start(graph_name, input_data, run_id=run_id)

    def get_run(self, run_id: str) -> SharedGraphState:
        state = self.engine._require_run(run_id)
        if state.graph_name != self.graph_name:
            raise RunNotFoundError(f"Run was not found for {self.graph_name}: {run_id}")
        return state

    def list_runs(self) -> list[SharedGraphState]:
        return self.engine.store.list_runs(self.graph_name)

    def submit_external_input(
        self, run_id: str, payload: dict[str, Any]
    ) -> SharedGraphState:
        self.get_run(run_id)
        return self.engine.resume_external(run_id, payload)

    def pending_hitl_tasks(self) -> list[dict[str, Any]]:
        return [
            task
            for task in self.engine.store.list_hitl_tasks(HITLStatus.PENDING)
            if task.get("state", {}).get("graph_name") == self.graph_name
        ]

    def resolve_hitl(
        self,
        task_id: str,
        *,
        approved: bool,
        note: str,
        admin_employee_id: int,
    ) -> SharedGraphState:
        task = self.engine.store.get_hitl_task(task_id)
        if not self._owns(task):
            raise ValueError("HITL task was not found for this graph.")
        return self.engine.resolve_hitl(
            task_id,
            approved=approved,
            note=note,
            admin_employee_id=admin_employee_id,
        )

    def tickets(
        self, status: TicketStatus | None = None
    ) -> list[dict[str, Any]]:
        return [
            ticket
            for ticket in self.engine.store.list_tickets(status)
            if ticket.get("state", {}).get("graph_name") == self.graph_name
        ]

    def investigate_ticket(self, ticket_id: str) -> dict[str, Any]:
        ticket = self.engine.store.get_ticket(ticket_id)
        if not self._owns(ticket):
            raise ValueError("Failure ticket was not found for this graph.")
        return self.engine.investigate_ticket(ticket_id)

    def resolve_ticket(
        self, ticket_id: str, *, resolution_note: str
    ) -> SharedGraphState:
        ticket = self.engine.store.get_ticket(ticket_id)
        if not self._owns(ticket):
            raise ValueError("Failure ticket was not found for this graph.")
        return self.engine.resolve_ticket(
            ticket_id, resolution_note=resolution_note
        )

    def _owns(self, record: dict[str, Any] | None) -> bool:
        return bool(
            record
            and record.get("state", {}).get("graph_name") == self.graph_name
        )
