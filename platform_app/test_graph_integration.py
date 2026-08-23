from __future__ import annotations

from platform_app.graph_integration import (
    GRAPH_1_AGENT_ID,
    GRAPH_2_AGENT_ID,
    PlatformGraphIntegration,
)
from state_graph.core.state import SharedGraphState
from state_graph.core.types import RunStatus
from state_graph.graph_2.state import RateExceptionState


class FakeGraph1Service:
    def __init__(self) -> None:
        self.state: SharedGraphState | None = None
        self.decision: bool | None = None

    def start_run(self, graph_name, input_data):
        self.state = SharedGraphState(
            run_id="g1-run",
            graph_name=graph_name,
            current_node="wait_for_customer",
            status=RunStatus.WAITING_EXTERNAL,
            input_data=input_data,
            data={
                "recovery_options": [
                    {"action": "reroute", "label": "Reroute shipment"}
                ]
            },
            transition_history=[
                {"source": "START", "target": "load_shipment", "event": "continue"},
                {
                    "source": "generate_recovery_options",
                    "target": "wait_for_customer",
                    "event": "wait_external",
                },
            ],
        )
        return self.state

    def get_run(self, run_id):
        if self.state is None or self.state.run_id != run_id:
            raise LookupError(run_id)
        return self.state

    def submit_external_input(self, run_id, payload):
        state = self.get_run(run_id)
        state.data["customer_choice"] = payload
        state.status = RunStatus.WAITING_HITL
        state.current_node = "wait_for_admin"
        return state

    def pending_hitl_tasks(self):
        if self.state is None or self.state.status is not RunStatus.WAITING_HITL:
            return []
        return [
            {
                "task_id": "g1-hitl",
                "run_id": self.state.run_id,
                "status": "pending",
                "reason": "Destination is not verified.",
                "created_at": "2026-08-23T10:00:00+00:00",
            }
        ]

    def resolve_hitl(self, task_id, *, approved, note, admin_employee_id):
        del task_id, note, admin_employee_id
        self.decision = approved
        assert self.state is not None
        self.state.status = RunStatus.COMPLETED
        self.state.current_node = "END"
        self.state.data["shipment"] = {"destination": "Giza Warehouse"}
        return self.state

    def tickets(self):
        return []


class FakeGraph2Store:
    def __init__(self) -> None:
        self.state: dict | None = None
        self.task_status = "open"

    def latest(self, run_id):
        if self.state is None or self.state["run_id"] != run_id:
            return None
        return dict(self.state)

    def history(self, run_id):
        if self.latest(run_id) is None:
            return []
        return [
            {"sequence": 1, "node": "load_shipment", "state": self.state},
            {"sequence": 2, "node": self.state["current_node"], "state": self.state},
        ]

    def list_tasks(self, task_type=None, status="open"):
        if self.state is None or self.task_status != status:
            return []
        return [
            {
                "task_id": "g2-hitl" if task_type == "hitl" else "g2-ticket",
                "run_id": self.state["run_id"],
                "task_type": task_type,
                "status": status,
                "state": dict(self.state),
                "created_at": "2026-08-23 10:00:00",
                "resolved_at": None,
            }
        ]

    def update_task_status(self, task_id, status):
        del task_id
        self.task_status = status


class FakeGraph2:
    def __init__(self) -> None:
        self.checkpoints = FakeGraph2Store()

    def start(self, shipment_id, session_id, *, employee_id):
        state = RateExceptionState(
            run_id="g2-run",
            shipment_id=shipment_id,
            session_id=session_id,
            employee_id=employee_id,
            current_node="wait_for_admin",
            discount_pct=25.0,
            hitl_task_id="g2-hitl",
        )
        self.checkpoints.state = state.to_dict()
        return state

    def pending_hitl_tasks(self):
        return self.checkpoints.list_tasks("hitl", "open")

    def resume(self, run_id, *, admin_decision, admin_note):
        del admin_decision, admin_note
        state = RateExceptionState.from_dict(self.checkpoints.latest(run_id))
        state.current_node = "END"
        state.final_status = "approved"
        self.checkpoints.state = state.to_dict()
        self.checkpoints.task_status = "resolved"
        return state

    def resolve_failure(self, run_id):
        state = RateExceptionState.from_dict(self.checkpoints.latest(run_id))
        state.current_node = "END"
        state.final_status = "recovered"
        self.checkpoints.state = state.to_dict()
        self.checkpoints.task_status = "resolved"
        return state


def integration():
    graph_1 = FakeGraph1Service()
    graph_2 = FakeGraph2()
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
    assert started["run_id"] == "g1-run"

    waiting = platform.chat(
        GRAPH_1_AGENT_ID,
        "reroute to Giza Warehouse, cost 650, verified no",
        "g1-run",
    )
    assert waiting["status"] == "paused_hitl"
    assert graph_1.state.data["customer_choice"]["destination_verified"] is False


def test_graph_2_chat_starts_live_workflow_and_exposes_run_history():
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


def test_admin_hitl_queue_routes_decisions_to_the_correct_graph():
    platform, graph_1, _ = integration()
    platform.chat(
        GRAPH_1_AGENT_ID,
        "start shipment 6, employee 1, reason: customer unavailable at destination",
        None,
    )
    platform.chat(
        GRAPH_1_AGENT_ID,
        "reroute to Giza Warehouse, cost 650, verified no",
        "g1-run",
    )

    tasks = platform.hitl_tasks()
    assert tasks[0]["agent_id"] == GRAPH_1_AGENT_ID
    resolved = platform.decide_hitl(
        "g1-hitl",
        decision="approve",
        decided_by="admin",
        admin_employee_id=3,
    )
    assert resolved["status"] == "completed"
    assert graph_1.decision is True


def test_graph_2_ticket_lifecycle_supports_investigating_and_resolved():
    platform, _, graph_2 = integration()
    state = RateExceptionState(
        run_id="g2-run",
        shipment_id=5,
        session_id="platform-session",
        current_node="failure_ticket",
        failed_node="retrieve_policy",
        error="Qdrant unavailable",
        ticket_id="g2-ticket",
        ticket_status="open",
    )
    graph_2.checkpoints.state = state.to_dict()

    ticket = platform.tickets()[0]
    assert ticket["status"] == "open"
    investigating = platform.set_ticket_status("g2-ticket", "investigating")
    assert investigating["ticket"]["status"] == "investigating"
    resolved = platform.set_ticket_status("g2-ticket", "resolved")
    assert resolved["run"]["status"] == "completed"
    resolved_tickets = platform.tickets()
    assert resolved_tickets[0]["status"] == "resolved"
    assert resolved_tickets[0]["graph_status"] is None
