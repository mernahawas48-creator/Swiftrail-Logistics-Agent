from __future__ import annotations

import uuid
from typing import Any

from state_graph.core.checkpoint_store import CheckpointStore
from state_graph.core.exceptions import InvalidRunStatusError, RunNotFoundError
from state_graph.core.nodes import NodeContext, NodeResult
from state_graph.core.registry import GraphRegistry
from state_graph.core.state import SharedGraphState, utc_now
from state_graph.core.types import HITLStatus, NodeDirective, RunStatus, TicketStatus


class GraphEngine:
    """Execute registered graphs with durable pause and recovery semantics."""

    def __init__(
        self,
        registry: GraphRegistry,
        store: CheckpointStore,
        *,
        services: dict[str, Any] | None = None,
        max_steps_per_run: int = 100,
    ) -> None:
        self.registry = registry
        self.store = store
        self.context = NodeContext(services or {})
        self.max_steps_per_run = max_steps_per_run

    def start(
        self,
        graph_name: str,
        input_data: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> SharedGraphState:
        graph = self.registry.get(graph_name)
        state = SharedGraphState(
            run_id=run_id or uuid.uuid4().hex,
            graph_name=graph_name,
            current_node=graph.start_node,
            input_data=dict(input_data),
        )
        self.store.create_run(state)
        return self.run(state.run_id)

    def run(self, run_id: str) -> SharedGraphState:
        state = self._require_run(run_id)
        if state.status is not RunStatus.RUNNING:
            return state

        graph = self.registry.get(state.graph_name)
        steps = 0
        while state.status is RunStatus.RUNNING:
            if state.current_node == graph.end_node:
                state.status = RunStatus.COMPLETED
                return state
            steps += 1
            if steps > self.max_steps_per_run:
                self._record_failure(
                    state,
                    RuntimeError("Graph exceeded its per-run transition limit."),
                )
                return state

            source = state.current_node
            execution_key = f"{state.run_id}:{state.revision}:{source}"
            try:
                cached = self.store.load_node_result(execution_key)
                if cached is None:
                    node = graph.nodes[source]
                    result = node.run(state, self.context)
                    self.store.save_node_result(
                        execution_key,
                        run_id=state.run_id,
                        node=source,
                        result=result.to_dict(),
                    )
                else:
                    result = NodeResult.from_dict(cached)
                self._apply_result(state, result)
            except Exception as exc:
                self._record_failure(state, exc)
                return state

        return state

    def _apply_result(self, state: SharedGraphState, result: NodeResult) -> None:
        graph = self.registry.get(state.graph_name)
        source = state.current_node
        graph.require_transition(source, result.next_node)
        state.apply_updates(result.updates)

        if result.directive is NodeDirective.WAIT_EXTERNAL:
            state.status = RunStatus.WAITING_EXTERNAL
            state.resume_node = result.next_node
            state.revision += 1
            state.updated_at = utc_now()
            state.transition_history.append(
                {
                    "source": source,
                    "target": source,
                    "event": "external_wait_started",
                    "at": state.updated_at,
                }
            )
            state.data["external_request"] = result.request
            self.store.save_checkpoint(
                state, node=source, event="external_wait_started"
            )
            return

        if result.directive is NodeDirective.WAIT_HITL:
            task_id = f"HITL-{uuid.uuid4().hex[:12]}"
            state.status = RunStatus.WAITING_HITL
            state.resume_node = result.next_node
            state.hitl_task_id = task_id
            state.revision += 1
            state.updated_at = utc_now()
            state.transition_history.append(
                {
                    "source": source,
                    "target": source,
                    "event": "hitl_requested",
                    "at": state.updated_at,
                }
            )
            self.store.save_checkpoint(state, node=source, event="hitl_requested")
            self.store.create_hitl_task(
                {
                    "task_id": task_id,
                    "run_id": state.run_id,
                    "node": source,
                    "status": HITLStatus.PENDING.value,
                    "reason": result.reason or "Admin decision required.",
                    "request": result.request,
                    "state": state.to_dict(),
                }
            )
            return

        event = (
            "run_completed"
            if result.directive is NodeDirective.COMPLETE
            else "node_completed"
        )
        state.completed_nodes.append(source)
        state.record_transition(source, result.next_node, event)
        if result.directive is NodeDirective.COMPLETE:
            state.status = RunStatus.COMPLETED
        self.store.save_checkpoint(state, node=result.next_node, event=event)

    def resume_external(
        self, run_id: str, payload: dict[str, Any]
    ) -> SharedGraphState:
        state = self._require_run(run_id)
        if state.status is not RunStatus.WAITING_EXTERNAL or not state.resume_node:
            raise InvalidRunStatusError("Run is not waiting for external input.")
        source = state.current_node
        target = state.resume_node
        self.registry.get(state.graph_name).require_transition(source, target)
        state.data["external_input"] = dict(payload)
        state.data.pop("external_request", None)
        state.status = RunStatus.RUNNING
        state.resume_node = None
        state.record_transition(source, target, "external_input_received")
        self.store.save_checkpoint(
            state, node=target, event="external_input_received"
        )
        return self.run(run_id)

    def resolve_hitl(
        self,
        task_id: str,
        *,
        approved: bool,
        note: str,
        admin_employee_id: int,
    ) -> SharedGraphState:
        if len(note.strip()) < 10:
            raise ValueError("Admin note must contain at least 10 characters.")
        task = self.store.get_hitl_task(task_id)
        if task is None:
            raise ValueError("HITL task was not found.")
        state = self._require_run(task["run_id"])
        if (
            task["status"] in {HITLStatus.APPROVED.value, HITLStatus.REJECTED.value}
            and state.status is RunStatus.RUNNING
            and state.hitl_task_id == task_id
        ):
            return self.run(state.run_id)
        if (
            state.status is not RunStatus.WAITING_HITL
            or state.hitl_task_id != task_id
            or not state.resume_node
        ):
            raise InvalidRunStatusError("Run is not waiting for this HITL task.")

        decision = {
            "approved": approved,
            "note": note.strip(),
            "admin_employee_id": admin_employee_id,
        }
        requested_status = (
            HITLStatus.APPROVED if approved else HITLStatus.REJECTED
        )
        if task["status"] == HITLStatus.PENDING.value:
            self.store.update_hitl_task(
                task_id,
                status=requested_status,
                decision=decision,
            )
        elif task["status"] == requested_status.value and isinstance(
            task.get("decision"), dict
        ):
            # The decision was committed before a process crash. Reuse the
            # persisted decision and finish the run transition idempotently.
            decision = task["decision"]
        else:
            raise ValueError("HITL task was already resolved differently.")
        source = state.current_node
        target = state.resume_node
        self.registry.get(state.graph_name).require_transition(source, target)
        state.data["admin_decision"] = decision
        state.status = RunStatus.RUNNING
        state.resume_node = None
        state.record_transition(source, target, "hitl_resolved")
        self.store.save_checkpoint(state, node=target, event="hitl_resolved")
        return self.run(state.run_id)

    def investigate_ticket(self, ticket_id: str) -> dict[str, Any]:
        self.store.update_ticket(
            ticket_id,
            status=TicketStatus.INVESTIGATING,
        )
        ticket = self.store.get_ticket(ticket_id)
        assert ticket is not None
        return ticket

    def resolve_ticket(
        self, ticket_id: str, *, resolution_note: str
    ) -> SharedGraphState:
        if len(resolution_note.strip()) < 10:
            raise ValueError("Resolution note must contain at least 10 characters.")
        ticket = self.store.get_ticket(ticket_id)
        if ticket is None:
            raise ValueError("Failure ticket was not found.")
        state = self._require_run(ticket["run_id"])
        if (
            ticket["status"] == TicketStatus.RESOLVED.value
            and state.status is RunStatus.RUNNING
            and state.ticket_id == ticket_id
        ):
            return self.run(state.run_id)
        if (
            state.status is not RunStatus.WAITING_TICKET
            or state.ticket_id != ticket_id
            or not state.failed_node
        ):
            raise InvalidRunStatusError("Run is not waiting for this ticket.")
        if ticket["status"] == TicketStatus.INVESTIGATING.value:
            self.store.update_ticket(
                ticket_id,
                status=TicketStatus.RESOLVED,
                resolution_note=resolution_note.strip(),
            )
        elif ticket["status"] != TicketStatus.RESOLVED.value:
            raise ValueError("Ticket must be investigating before resolution.")
        state.status = RunStatus.RUNNING
        state.current_node = state.failed_node
        state.resume_node = None
        state.error = None
        state.revision += 1
        state.updated_at = utc_now()
        state.transition_history.append(
            {
                "source": "failure_ticket",
                "target": state.failed_node,
                "event": "ticket_resolved",
                "at": state.updated_at,
            }
        )
        self.store.save_checkpoint(
            state, node=state.current_node, event="ticket_resolved"
        )
        return self.run(state.run_id)

    def _record_failure(
        self, state: SharedGraphState, error: Exception
    ) -> None:
        ticket_id = f"FT-{uuid.uuid4().hex[:12]}"
        failed_node = state.current_node
        state.status = RunStatus.WAITING_TICKET
        state.failed_node = failed_node
        state.resume_node = failed_node
        state.ticket_id = ticket_id
        state.error = {
            "type": type(error).__name__,
            "message": str(error),
        }
        state.revision += 1
        state.updated_at = utc_now()
        state.transition_history.append(
            {
                "source": failed_node,
                "target": "failure_ticket",
                "event": "failure_detected",
                "at": state.updated_at,
            }
        )
        self.store.save_checkpoint(
            state, node=failed_node, event="failure_detected"
        )
        self.store.create_ticket(
            {
                "ticket_id": ticket_id,
                "run_id": state.run_id,
                "failed_node": failed_node,
                "status": TicketStatus.OPEN.value,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "state": state.to_dict(),
            }
        )

    def _require_run(self, run_id: str) -> SharedGraphState:
        state = self.store.load_run(run_id)
        if state is None:
            raise RunNotFoundError(f"Run was not found: {run_id}")
        return state
