from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from state_graph.core.exceptions import RunNotFoundError
from state_graph.core.types import RunStatus
from state_graph.graph_1.graph import GRAPH_NAME as GRAPH_1_NAME
from state_graph.graph_2.definition import GRAPH_NAME as GRAPH_2_NAME
from state_graph.graph_3.definition import GRAPH_NAME as GRAPH_3_NAME

GRAPH_1_AGENT_ID = "graph1_delivery_exception"
GRAPH_2_AGENT_ID = "graph2_rate_exception"
GRAPH_3_AGENT_ID = "graph3_credit_hold_remediation"

_GRAPH_1_START = re.compile(
    r"^start\s+shipment\s+(?P<shipment>\d+)"
    r"(?:\s*,?\s*employee\s+(?P<employee>\d+))?"
    r"(?:\s*,?\s*reason:\s*(?P<reason>.+))?$",
    re.IGNORECASE,
)
_GRAPH_1_CHOICE = re.compile(
    r"^(?P<action>reroute|redeliver)\s+to\s+(?P<destination>.+?)"
    r"\s*,\s*cost\s+(?P<cost>\d+(?:\.\d+)?)"
    r"\s*,\s*verified\s+(?P<verified>yes|no|true|false)$",
    re.IGNORECASE,
)
_GRAPH_2_START = re.compile(
    r"^start\s+shipment\s+(?P<shipment>\d+)"
    r"(?:\s*,?\s*employee\s+(?P<employee>\d+))?$",
    re.IGNORECASE,
)
_GRAPH_3_START = re.compile(
    r"^start\s+customer\s+(?P<customer>\d+)"
    r"(?:\s*,?\s*employee\s+(?P<employee>\d+))?"
    r"(?:\s*,?\s*claim:\s*(?P<claim>.+))?$",
    re.IGNORECASE,
)


def _default_graph_1_service():
    from state_graph.graph_1.live import build_live_service

    return build_live_service()


def _default_graph_2_service():
    from state_graph.graph_2.live import build_live_service

    return build_live_service()


def _default_graph_3_service():
    from state_graph.graph_3.live import build_live_service

    return build_live_service()


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        current = value if value.tzinfo else value.replace(tzinfo=UTC)
        return current.timestamp()
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            except ValueError:
                return datetime.now(UTC).timestamp()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    return datetime.now(UTC).timestamp()


def _graph_1_status(status: RunStatus) -> str:
    return {
        RunStatus.RUNNING: "running",
        RunStatus.WAITING_EXTERNAL: "paused_wait",
        RunStatus.WAITING_HITL: "paused_hitl",
        RunStatus.WAITING_TICKET: "ticketed",
        RunStatus.COMPLETED: "completed",
    }[status]


class PlatformGraphIntegration:
    """Connect all shared-core state graphs without duplicating graph logic."""

    def __init__(
        self,
        graph_1_factory: Callable[[], Any] = _default_graph_1_service,
        graph_2_factory: Callable[[], Any] = _default_graph_2_service,
        graph_3_factory: Callable[[], Any] = _default_graph_3_service,
    ) -> None:
        self._graph_1_factory = graph_1_factory
        self._graph_2_factory = graph_2_factory
        self._graph_3_factory = graph_3_factory
        self._graph_1: Any | None = None
        self._graph_2: Any | None = None
        self._graph_3: Any | None = None

    @property
    def graph_1(self):
        if self._graph_1 is None:
            self._graph_1 = self._graph_1_factory()
        return self._graph_1

    @property
    def graph_2(self):
        if self._graph_2 is None:
            self._graph_2 = self._graph_2_factory()
        return self._graph_2

    @property
    def graph_3(self):
        if self._graph_3 is None:
            self._graph_3 = self._graph_3_factory()
        return self._graph_3

    def chat(self, agent_id: str, message: str, run_id: str | None) -> dict[str, Any]:
        if agent_id == GRAPH_1_AGENT_ID:
            return self._chat_graph_1(message, run_id)
        if agent_id == GRAPH_2_AGENT_ID:
            return self._chat_graph_2(message, run_id)
        if agent_id == GRAPH_3_AGENT_ID:
            return self._chat_graph_3(message, run_id)
        raise KeyError(agent_id)

    def _chat_graph_1(self, message: str, run_id: str | None) -> dict[str, Any]:
        if run_id is None:
            match = _GRAPH_1_START.fullmatch(message.strip())
            if match is None or not match.group("reason"):
                return {
                    "reply": (
                        "Start Graph 1 with: start shipment 6, employee 1, "
                        "reason: customer unavailable at the delivery destination"
                    )
                }
            state = self.graph_1.start_run(
                GRAPH_1_NAME,
                {
                    "shipment_id": int(match.group("shipment")),
                    "employee_id": int(match.group("employee") or 1),
                    "session_id": f"platform-g1-{uuid.uuid4().hex[:12]}",
                    "failure_reason": match.group("reason").strip(),
                },
            )
            return self._graph_1_chat_response(state)

        state = self.graph_1.get_run(run_id)
        if state.status is RunStatus.WAITING_EXTERNAL:
            lowered = message.strip().lower()
            if lowered in {"new options", "request new options"}:
                payload = {"action": "request_new_options"}
            else:
                choice = _GRAPH_1_CHOICE.fullmatch(message.strip())
                if choice is None:
                    return self._graph_1_chat_response(
                        state,
                        prefix=(
                            "Choose: reroute to Giza Warehouse, cost 650, verified no; "
                            "or type new options. "
                        ),
                    )
                payload = {
                    "action": choice.group("action").lower(),
                    "new_destination": choice.group("destination").strip(),
                    "estimated_cost": float(choice.group("cost")),
                    "destination_verified": choice.group("verified").lower()
                    in {"yes", "true"},
                }
            state = self.graph_1.submit_external_input(run_id, payload)
        return self._graph_1_chat_response(state)

    @staticmethod
    def _graph_1_chat_response(state: Any, *, prefix: str = "") -> dict[str, Any]:
        status = _graph_1_status(state.status)
        if status == "paused_wait":
            options = state.data.get("recovery_options", [])
            labels = "; ".join(option.get("label", option["action"]) for option in options)
            reply = f"{prefix}Recovery options: {labels}."
        elif status == "paused_hitl":
            reply = "The selected recovery option is waiting for an admin decision."
        elif status == "ticketed":
            reply = "Graph 1 failed safely and opened a ticket for an admin."
        elif status == "completed":
            shipment = state.data.get("shipment", {})
            reply = f"Delivery recovery completed. Destination: {shipment.get('destination', 'updated')}."
        else:
            reply = f"Graph 1 is running at {state.current_node}."
        return {
            "run_id": state.run_id,
            "reply": reply,
            "status": status,
            "current_node": state.current_node,
        }

    def _chat_graph_2(self, message: str, run_id: str | None) -> dict[str, Any]:
        if run_id is None:
            match = _GRAPH_2_START.fullmatch(message.strip())
            if match is None:
                return {"reply": "Start Graph 2 with: start shipment 5, employee 3"}
            state = self.graph_2.start_run(
                GRAPH_2_NAME,
                {
                    "shipment_id": int(match.group("shipment")),
                    "session_id": f"platform-g2-{uuid.uuid4().hex[:12]}",
                    "employee_id": int(match.group("employee") or 3),
                },
            )
        else:
            state = self.graph_2.get_run(run_id)
        return self._graph_2_chat_response(state)

    @staticmethod
    def _graph_2_chat_response(state: Any) -> dict[str, Any]:
        status = _graph_1_status(state.status)
        if status == "paused_hitl":
            reply = (
                f"The {state.data.get('discount_pct')}% rate exception is waiting "
                "for finance approval."
            )
        elif status == "ticketed":
            reply = "Graph 2 failed safely and opened a ticket for an admin."
        elif status == "completed":
            reply = (
                "Rate-exception workflow completed with status: "
                f"{state.data.get('final_status')}."
            )
        else:
            reply = f"Graph 2 is running at {state.current_node}."
        return {
            "run_id": state.run_id,
            "reply": reply,
            "status": status,
            "current_node": state.current_node,
        }

    def _chat_graph_3(self, message: str, run_id: str | None) -> dict[str, Any]:
        if run_id is None:
            match = _GRAPH_3_START.fullmatch(message.strip())
            if match is None:
                return {
                    "reply": (
                        "Start Graph 3 with: start customer 3, employee 3, "
                        "claim: invoice amount is disputed"
                    )
                }
            state = self.graph_3.start_run(
                GRAPH_3_NAME,
                {
                    "customer_id": int(match.group("customer")),
                    "employee_id": int(match.group("employee") or 3),
                    "session_id": f"platform-g3-{uuid.uuid4().hex[:12]}",
                    "customer_claim": match.group("claim"),
                },
            )
        else:
            state = self.graph_3.get_run(run_id)
            if state.status is RunStatus.WAITING_EXTERNAL:
                if state.data.get("waiting_on") == "dispute_evidence":
                    payload = {"evidence": message.strip()}
                else:
                    amount_match = re.search(r"\d+(?:\.\d+)?", message.replace(",", ""))
                    if amount_match is None:
                        return self._graph_3_chat_response(state)
                    payload = {"amount": float(amount_match.group())}
                state = self.graph_3.submit_external_input(run_id, payload)
        return self._graph_3_chat_response(state)

    @staticmethod
    def _graph_3_chat_response(state: Any) -> dict[str, Any]:
        status = _graph_1_status(state.status)
        if status == "paused_wait":
            reply = f"Waiting for customer {state.data.get('waiting_on')}."
        elif status == "paused_hitl":
            reply = "The severe credit hold is waiting for finance approval."
        elif status == "ticketed":
            reply = "Graph 3 failed safely and opened a ticket for an admin."
        elif status == "completed":
            reply = f"Credit-hold workflow completed: {state.data.get('final_status')}."
        else:
            reply = f"Graph 3 is running at {state.current_node}."
        return {
            "run_id": state.run_id,
            "reply": reply,
            "status": status,
            "current_node": state.current_node,
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        for service in (self.graph_1, self.graph_2, self.graph_3):
            try:
                state = service.get_run(run_id)
            except RunNotFoundError:
                continue
            history = [
                {"sequence": index, "node_name": item["target"]}
                for index, item in enumerate(state.transition_history, start=1)
            ]
            return {
                "run": {
                    "run_id": state.run_id,
                    "graph_name": state.graph_name,
                    "status": _graph_1_status(state.status),
                    "current_node": state.current_node,
                },
                "history": history,
                "state": state.to_dict(),
            }
        return None

    def list_runs(self, graph_name: str | None = None) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for service in (self.graph_1, self.graph_2, self.graph_3):
            if graph_name is not None and service.graph_name != graph_name:
                continue
            runs.extend(
                {
                    "run_id": state.run_id,
                    "graph_name": state.graph_name,
                    "status": _graph_1_status(state.status),
                    "current_node": state.current_node,
                    "updated_at": state.updated_at,
                }
                for state in service.list_runs()
            )
        return sorted(runs, key=lambda item: item["updated_at"], reverse=True)

    def hitl_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for task in self.graph_1.pending_hitl_tasks():
            tasks.append(
                {
                    **task,
                    "agent_id": GRAPH_1_AGENT_ID,
                    "graph_status": "paused_hitl",
                    "created_at": _epoch(task.get("created_at")),
                }
            )
        for task in self.graph_2.pending_hitl_tasks():
            tasks.append(
                {
                    **task,
                    "created_at": _epoch(task.get("created_at")),
                    "agent_id": GRAPH_2_AGENT_ID,
                    "graph_status": "paused_hitl",
                }
            )
        for task in self.graph_3.pending_hitl_tasks():
            tasks.append(
                {
                    **task,
                    "created_at": _epoch(task.get("created_at")),
                    "agent_id": GRAPH_3_AGENT_ID,
                    "graph_status": "paused_hitl",
                }
            )
        return tasks

    def decide_hitl(
        self,
        task_id: str,
        *,
        decision: str,
        decided_by: str,
        admin_employee_id: int,
    ) -> dict[str, Any] | None:
        graph_1_tasks = {task["task_id"]: task for task in self.graph_1.pending_hitl_tasks()}
        if task_id in graph_1_tasks:
            state = self.graph_1.resolve_hitl(
                task_id,
                approved=decision == "approve",
                note=f"Platform decision by {decided_by}.",
                admin_employee_id=admin_employee_id,
            )
            return self._graph_1_chat_response(state)

        graph_2_tasks = {
            task["task_id"]: task for task in self.graph_2.pending_hitl_tasks()
        }
        if task_id in graph_2_tasks:
            state = self.graph_2.resolve_hitl(
                task_id,
                approved=decision == "approve",
                note=f"Platform decision by {decided_by}.",
                admin_employee_id=admin_employee_id,
            )
            return self._graph_2_chat_response(state)

        graph_3_tasks = {
            task["task_id"]: task for task in self.graph_3.pending_hitl_tasks()
        }
        if task_id not in graph_3_tasks:
            return None
        state = self.graph_3.resolve_hitl(
            task_id,
            approved=decision == "approve",
            note=f"Platform decision by {decided_by}.",
            admin_employee_id=admin_employee_id,
        )
        return self._graph_3_chat_response(state)

    def tickets(self) -> list[dict[str, Any]]:
        tickets: list[dict[str, Any]] = []
        for ticket in self.graph_1.tickets():
            tickets.append(
                {
                    **ticket,
                    "agent_id": GRAPH_1_AGENT_ID,
                    "graph_status": "ticketed" if ticket["status"] != "resolved" else None,
                    "created_at": _epoch(ticket.get("created_at")),
                }
            )
        for ticket in self.graph_2.tickets():
            tickets.append(
                {
                    **ticket,
                    "node_name": ticket["failed_node"],
                    "created_at": _epoch(ticket.get("created_at")),
                    "agent_id": GRAPH_2_AGENT_ID,
                    "graph_status": (
                        "ticketed" if ticket["status"] != "resolved" else None
                    ),
                }
            )
        for ticket in self.graph_3.tickets():
            tickets.append(
                {
                    **ticket,
                    "node_name": ticket["failed_node"],
                    "created_at": _epoch(ticket.get("created_at")),
                    "agent_id": GRAPH_3_AGENT_ID,
                    "graph_status": (
                        "ticketed" if ticket["status"] != "resolved" else None
                    ),
                }
            )
        return tickets

    def set_ticket_status(self, ticket_id: str, status: str) -> dict[str, Any] | None:
        graph_1_tickets = {ticket["ticket_id"]: ticket for ticket in self.graph_1.tickets()}
        ticket = graph_1_tickets.get(ticket_id)
        if ticket is not None:
            if status == "investigating":
                updated = self.graph_1.investigate_ticket(ticket_id)
                return {"ticket": updated, "run": self.get_run(ticket["run_id"])["run"]}
            if status == "resolved":
                state = self.graph_1.resolve_ticket(
                    ticket_id,
                    resolution_note="Resolved through the platform admin dashboard.",
                )
                return {
                    "ticket": {**ticket, "status": "resolved"},
                    "run": self.get_run(state.run_id)["run"],
                }

        graph_2_tickets = {
            ticket["ticket_id"]: ticket for ticket in self.graph_2.tickets()
        }
        ticket = graph_2_tickets.get(ticket_id)
        if ticket is not None:
            if status == "investigating":
                updated = self.graph_2.investigate_ticket(ticket_id)
                return {
                    "ticket": updated,
                    "run": self.get_run(ticket["run_id"])["run"],
                }
            if status == "resolved":
                state = self.graph_2.resolve_ticket(
                    ticket_id,
                    resolution_note="Resolved through the platform admin dashboard.",
                )
                return {
                    "ticket": {"ticket_id": ticket_id, "status": "resolved"},
                    "run": self._graph_2_chat_response(state),
                }

        graph_3_tickets = {
            ticket["ticket_id"]: ticket for ticket in self.graph_3.tickets()
        }
        ticket = graph_3_tickets.get(ticket_id)
        if ticket is None:
            return None
        if status == "investigating":
            updated = self.graph_3.investigate_ticket(ticket_id)
            return {
                "ticket": updated,
                "run": self.get_run(ticket["run_id"])["run"],
            }
        if status == "resolved":
            state = self.graph_3.resolve_ticket(
                ticket_id,
                resolution_note="Resolved through the platform admin dashboard.",
            )
            return {
                "ticket": {"ticket_id": ticket_id, "status": "resolved"},
                "run": self._graph_3_chat_response(state),
            }
        raise ValueError("Ticket status must be investigating or resolved.")
