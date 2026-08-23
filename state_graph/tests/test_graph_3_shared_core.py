from pathlib import Path

from state_graph.core.engine import GraphEngine
from state_graph.core.registry import GraphRegistry
from state_graph.core.service import GraphService
from state_graph.core.sqlite_store import SQLiteCheckpointStore
from state_graph.core.state import SharedGraphState
from state_graph.core.types import RunStatus, TicketStatus
from state_graph.graph_3.definition import GRAPH_NAME, build_credit_hold_graph


class FakeCreditTools:
    def __init__(self, *, severity="severe", fail_release=False, active=True):
        self.severity = severity
        self.fail_release = fail_release
        self.active = active
        self.releases = []

    def load_account(self, **kwargs):
        del kwargs
        invoices = [
            {"id": 5, "amount": "12000.00", "paid_status": "overdue"}
        ]
        holds = (
            [{"id": 3, "severity": self.severity, "status": "active"}]
            if self.active
            else []
        )
        return invoices, holds

    def release_hold(self, **kwargs):
        if self.fail_release:
            raise RuntimeError("MCP credit-hold release unavailable")
        self.releases.append(kwargs)
        return {"hold_id": kwargs["hold_id"], "status": "released"}


def service(tmp_path: Path, **tool_options):
    registry = GraphRegistry()
    registry.register(build_credit_hold_graph())
    store = SQLiteCheckpointStore(tmp_path / "shared.db")
    tools = FakeCreditTools(**tool_options)
    engine = GraphEngine(registry, store, services={"credit_tools": tools})
    return GraphService(engine), tools, store


def start(graph_service: GraphService, *, claim=None, run_id="graph-3-run"):
    return graph_service.start_run(
        GRAPH_NAME,
        {
            "customer_id": 3,
            "employee_id": 3,
            "session_id": "graph-3-session",
            "customer_claim": claim,
        },
        run_id=run_id,
    )


def test_graph_3_cycles_for_evidence_then_persists_hitl_and_resumes(tmp_path):
    graph_service, tools, store = service(tmp_path)
    waiting = start(graph_service, claim="invoice is incorrect")

    assert isinstance(waiting, SharedGraphState)
    assert waiting.status is RunStatus.WAITING_EXTERNAL
    assert waiting.data["waiting_on"] == "dispute_evidence"

    waiting_again = graph_service.submit_external_input(
        waiting.run_id, {"evidence": "too short"}
    )
    assert waiting_again.status is RunStatus.WAITING_EXTERNAL
    assert waiting_again.data["evidence_attempts"] == 1

    paused = graph_service.submit_external_input(
        waiting.run_id,
        {"evidence": "Invoice 5 duplicates shipment 2 and includes a receipt."},
    )
    assert paused.status is RunStatus.WAITING_HITL
    task = graph_service.pending_hitl_tasks()[0]
    assert store.load_run(paused.run_id).hitl_task_id == task["task_id"]

    completed = graph_service.resolve_hitl(
        task["task_id"],
        approved=True,
        note="Finance manager verified the supplied invoice evidence.",
        admin_employee_id=3,
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["final_status"] == "released"
    assert tools.releases[0]["approved"] is True


def test_graph_3_minor_hold_releases_after_full_payment(tmp_path):
    graph_service, tools, _ = service(tmp_path, severity="minor")
    waiting = start(graph_service)
    completed = graph_service.submit_external_input(
        waiting.run_id, {"amount": 12000}
    )

    assert completed.status is RunStatus.COMPLETED
    assert completed.data["final_status"] == "released"
    assert tools.releases[0]["approved"] is None
    assert graph_service.pending_hitl_tasks() == []


def test_graph_3_partial_payment_keeps_hold(tmp_path):
    graph_service, tools, _ = service(tmp_path, severity="minor")
    waiting = start(graph_service)
    completed = graph_service.submit_external_input(waiting.run_id, {"amount": 5000})

    assert completed.status is RunStatus.COMPLETED
    assert completed.data["final_status"] == "partial_payment_hold"
    assert tools.releases == []


def test_graph_3_failure_ticket_resumes_exact_failed_node(tmp_path):
    graph_service, tools, _ = service(
        tmp_path, severity="minor", fail_release=True
    )
    waiting = start(graph_service)
    failed = graph_service.submit_external_input(waiting.run_id, {"amount": 12000})
    ticket = graph_service.tickets(TicketStatus.OPEN)[0]

    assert failed.status is RunStatus.WAITING_TICKET
    assert failed.failed_node == "execute_remediation_action"
    graph_service.investigate_ticket(ticket["ticket_id"])
    tools.fail_release = False
    recovered = graph_service.resolve_ticket(
        ticket["ticket_id"],
        resolution_note="The live MCP connection was restored successfully.",
    )

    assert recovered.status is RunStatus.COMPLETED
    assert any(
        item["event"] == "ticket_resolved"
        and item["target"] == "execute_remediation_action"
        for item in recovered.transition_history
    )


def test_graph_3_completes_when_customer_has_no_active_hold(tmp_path):
    graph_service, tools, _ = service(tmp_path, active=False)
    completed = start(graph_service)

    assert completed.status is RunStatus.COMPLETED
    assert completed.data["final_status"] == "no_active_hold"
    assert tools.releases == []
