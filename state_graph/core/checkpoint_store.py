from __future__ import annotations

from typing import Any, Protocol

from state_graph.core.state import SharedGraphState
from state_graph.core.types import HITLStatus, TicketStatus


class CheckpointStore(Protocol):
    def create_run(self, state: SharedGraphState) -> None: ...

    def load_run(self, run_id: str) -> SharedGraphState | None: ...

    def list_runs(self, graph_name: str | None = None) -> list[SharedGraphState]: ...

    def save_checkpoint(
        self,
        state: SharedGraphState,
        *,
        node: str,
        event: str,
    ) -> int: ...

    def checkpoint_history(self, run_id: str) -> list[dict[str, Any]]: ...

    def save_node_result(
        self,
        execution_key: str,
        *,
        run_id: str,
        node: str,
        result: dict[str, Any],
    ) -> None: ...

    def load_node_result(self, execution_key: str) -> dict[str, Any] | None: ...

    def create_hitl_task(self, task: dict[str, Any]) -> None: ...

    def get_hitl_task(self, task_id: str) -> dict[str, Any] | None: ...

    def list_hitl_tasks(
        self, status: HITLStatus = HITLStatus.PENDING
    ) -> list[dict[str, Any]]: ...

    def update_hitl_task(
        self,
        task_id: str,
        *,
        status: HITLStatus,
        decision: dict[str, Any],
    ) -> None: ...

    def create_ticket(self, ticket: dict[str, Any]) -> None: ...

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None: ...

    def list_tickets(
        self, status: TicketStatus | None = None
    ) -> list[dict[str, Any]]: ...

    def update_ticket(
        self,
        ticket_id: str,
        *,
        status: TicketStatus,
        resolution_note: str | None = None,
    ) -> None: ...
