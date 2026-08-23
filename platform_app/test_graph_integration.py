from __future__ import annotations

from platform_app.graph_integration import (
    GRAPH_1_AGENT_ID,
    GRAPH_2_AGENT_ID,
    PlatformGraphIntegration,
)
from state_graph.core.exceptions import RunNotFoundError
from state_graph.core.state import SharedGraphState
from state_graph.core.types import RunStatus


class FakeGraphService:
    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        self.state: SharedGraphState | None = None
        self.hitl = []
        self.failure_tickets = []

    def start_run(self, graph_name, input_data):
        assert graph_name == self.graph_name
        if graph_name == "delivery_exception_recovery":
            data = {"recovery_options": [{"action": "reroute", "label": "Reroute"}]}
            current_node = "wait_for_customer"
            status = RunStatus.WAITING_EXTERNAL
            run_id = "g1-run"
        else:
            data = {"discount_pct": 25.0, "final_status": None}
            current_node = "wait_for_admin"
            status = RunStatus.WAITING_HITL
            run_id = "g2-run"
        self.state = SharedGraphState(
            run_id=run_id,
            graph_name=graph_name,
            current_node=current_node,
            status=status,
            input_data=input_data,
            data=data,
            transition_history=[
                {"source": "load_shipment", "target": current_node, "event": "node_completed"}
            ],
        )
        if graph_name == "rate_exception_approval":
            self.hitl = [
                {
                    "task_id": "g2-hitl",
                    "run_id": run_id,
                    "status": "pending",
                    "reason": "Finance review required.",
                    "created_at": "2026-08-23T10:00:00+00:00",
                    "state": self.state.to_dict(),
                }
            ]
        return self.state

    def get_run(self, run_id):
        if self.state is None or self.state.run_id != run_id:
            raise RunNotFoundError(run_id)
        return self.state

    def submit_external_input(self, run_id, payload):
        state = self.get_run(run_id)
        state.data["customer_choice"] = payload
        state.status = RunStatus.WAITING_HITL
        state.current_node = "wait_for_admin"
        self.hitl = [
            {
                "task_id": "g1-hitl",
                "run_id": run_id,
                "status": "pending",
                "reason": "Destination is not verified.",
                "created_at": "2026-08-23T10:00:00+00:00",
                "state": state.to_dict(),
            }
        ]
        return state

    def pending_hitl_tasks(self):
        return self.hitl

    def resolve_hitl(self, task_id, *, approved, note, admin_employee_id):
        del task_id, approved, note, admin_employee_id
        assert self.state is not None
        self.state.status = RunStatus.COMPLETED
        self.state.current_node = "END"
        if self.graph_name == "delivery_exception_recovery":
            self.state.data["shipment"] = {"destination": "Giza Warehouse"}
        else:
            self.state.data["final_status"] = "approved"
        self.hitl = []
        return self.state

    def tickets(self):
        return self.failure_tickets

    def investigate_ticket(self, ticket_id):
        ticket = next(item for item in self.failure_tickets if item["ticket_id"] == ticket_id)
        ticket["status"] = "investigating"
        return ticket

    def resolve_ticket(self, ticket_id, *, resolution_note):
        del resolution_note
        ticket = next(item for item in self.failure_tickets if item["ticket_id"] == ticket_id)
        ticket["status"] = "resolved"
        assert self.state is not None
        self.state.status = RunStatus.COMPLETED
        self.state.current_node = "END"
        self.state.data["final_status"] = "recovered"
        return self.state


def integration():
    graph_1 = FakeGraphService("delivery_exception_recovery")
    graph_2 = FakeGraphService("rate_exception_approval")
    return (
        PlatformGraphIntegration(lambda: graph_1, lambda: graph_2),
        graph_1,
        graph_2,
    )


def test_graph_1_chat_starts_and_submits_customer_choice():
    platform, graph_1, _ = integration()
    started = platform.chat(
        GRAPH_1_AGENT_ID,
        "start shipment 6, employee 1, reason: customer unavailable at destination",
        None,
    )
    assert started["status"] == "paused_wait"

    waiting = platform.chat(
        GRAPH_1_AGENT_ID,
        "reroute to Giza Warehouse, cost 650, verified no",
        "g1-run",
    )
    assert waiting["status"] == "paused_hitl"
    assert graph_1.state.data["customer_choice"]["destination_verified"] is False


def test_graph_2_chat_uses_shared_state_service_and_history():
    platform, _, _ = integration()
    started = platform.chat(
        GRAPH_2_AGENT_ID,
        "start shipment 5, employee 3",
        None,
    )

    assert started["status"] == "paused_hitl"
    assert "25.0%" in started["reply"]
    run = platform.get_run("g2-run")
    assert run["run"]["graph_name"] == "rate_exception_approval"
    assert run["history"][-1]["node_name"] == "wait_for_admin"


def test_admin_hitl_routes_graph_2_decision_through_shared_service():
    platform, _, _ = integration()
    platform.chat(GRAPH_2_AGENT_ID, "start shipment 5, employee 3", None)

    task = platform.hitl_tasks()[0]
    assert task["agent_id"] == GRAPH_2_AGENT_ID
    resolved = platform.decide_hitl(
        "g2-hitl",
        decision="approve",
        decided_by="admin",
        admin_employee_id=3,
    )
    assert resolved["status"] == "completed"


def test_graph_2_ticket_lifecycle_uses_shared_service():
    platform, _, graph_2 = integration()
    graph_2.start_run(
        "rate_exception_approval",
        {"shipment_id": 5, "session_id": "platform-session", "employee_id": 3},
    )
    graph_2.state.status = RunStatus.WAITING_TICKET
    graph_2.state.current_node = "retrieve_policy"
    graph_2.failure_tickets = [
        {
            "ticket_id": "g2-ticket",
            "run_id": "g2-run",
            "failed_node": "retrieve_policy",
            "error_type": "RuntimeError",
            "error_message": "Qdrant unavailable",
            "status": "open",
            "created_at": "2026-08-23T10:00:00+00:00",
            "state": graph_2.state.to_dict(),
        }
    ]

    investigating = platform.set_ticket_status("g2-ticket", "investigating")
    assert investigating["ticket"]["status"] == "investigating"
    resolved = platform.set_ticket_status("g2-ticket", "resolved")
    assert resolved["run"]["status"] == "completed"
