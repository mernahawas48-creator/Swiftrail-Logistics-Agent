from pathlib import Path
from types import SimpleNamespace

from state_graph.core.engine import GraphEngine
from state_graph.core.registry import GraphRegistry
from state_graph.core.service import GraphService
from state_graph.core.sqlite_store import SQLiteCheckpointStore
from state_graph.core.state import SharedGraphState
from state_graph.core.types import HITLStatus, RunStatus, TicketStatus
from state_graph.graph_2.definition import GRAPH_NAME, build_rate_exception_graph
from state_graph.graph_2.react import ReActDecision


class FakeRateTools:
    def __init__(self, discount=25.0):
        self.discount = discount
        self.applied = []

    def load_shipment(self, **kwargs):
        return {"id": kwargs["shipment_id"], "status": "pending"}

    def load_rate_exception(self, **kwargs):
        return {"id": 2, "discount_pct": self.discount, "status": "pending"}

    def apply_decision(self, **kwargs):
        self.applied.append(kwargs)
        approve = kwargs["approve"]
        return {
            "status": "auto_approved" if approve is None else "approved" if approve else "rejected"
        }


class FakeSearch:
    def __init__(self, *, fail=False):
        self.fail = fail

    def search(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("Qdrant unavailable")
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                metadata={"doc_id": "rate_exception_policy", "section_id": "RE-2"},
                text="Discounts above 15 percent require finance review.",
            )
        ]


class FakePlanner:
    def decide(self, *, exception, **kwargs):
        del kwargs
        human = float(exception["discount_pct"]) > 15
        return ReActDecision(
            "human_review" if human else "auto_approve",
            "Applied the delegated authority threshold.",
            "approve_rate_exception",
        )


def service(tmp_path: Path, *, discount=25.0, search=None):
    registry = GraphRegistry()
    registry.register(build_rate_exception_graph())
    store = SQLiteCheckpointStore(tmp_path / "shared.db")
    tools = FakeRateTools(discount)
    engine = GraphEngine(
        registry,
        store,
        services={
            "rate_tools": tools,
            "policy_search": search or FakeSearch(),
            "decision_planner": FakePlanner(),
        },
    )
    return GraphService(engine), tools, store


def start(graph_service: GraphService, run_id="graph-2-run"):
    return graph_service.start_run(
        GRAPH_NAME,
        {
            "shipment_id": 5,
            "session_id": "graph-2-session",
            "employee_id": 3,
        },
        run_id=run_id,
    )


def test_graph_2_uses_shared_state_and_persists_hitl(tmp_path):
    graph_service, _, store = service(tmp_path)
    state = start(graph_service)

    assert isinstance(state, SharedGraphState)
    assert state.status is RunStatus.WAITING_HITL
    assert state.current_node == "wait_for_admin"
    assert store.load_run(state.run_id).data["discount_pct"] == 25.0
    assert graph_service.pending_hitl_tasks()[0]["run_id"] == state.run_id


def test_graph_2_resumes_hitl_through_shared_engine(tmp_path):
    graph_service, tools, _ = service(tmp_path)
    start(graph_service)
    task = graph_service.pending_hitl_tasks()[0]

    completed = graph_service.resolve_hitl(
        task["task_id"],
        approved=True,
        note="Finance manager approved this exception.",
        admin_employee_id=3,
    )

    assert completed.status is RunStatus.COMPLETED
    assert completed.data["final_status"] == "approved"
    assert tools.applied[0]["approve"] is True
    assert store_status(graph_service, task["task_id"]) == HITLStatus.APPROVED.value


def store_status(graph_service, task_id):
    return graph_service.engine.store.get_hitl_task(task_id)["status"]


def test_graph_2_auto_approval_completes_without_hitl(tmp_path):
    graph_service, tools, _ = service(tmp_path, discount=10.0)
    state = start(graph_service)

    assert state.status is RunStatus.COMPLETED
    assert state.data["final_status"] == "auto_approved"
    assert graph_service.pending_hitl_tasks() == []
    assert tools.applied[0]["approve"] is None


def test_graph_2_failure_ticket_resumes_exact_failed_node(tmp_path):
    search = FakeSearch(fail=True)
    graph_service, _, _ = service(tmp_path, discount=10.0, search=search)
    failed = start(graph_service)
    ticket = graph_service.tickets(TicketStatus.OPEN)[0]

    assert failed.status is RunStatus.WAITING_TICKET
    assert failed.failed_node == "retrieve_policy"
    graph_service.investigate_ticket(ticket["ticket_id"])
    search.fail = False
    recovered = graph_service.resolve_ticket(
        ticket["ticket_id"],
        resolution_note="Qdrant connectivity was restored successfully.",
    )

    assert recovered.status is RunStatus.COMPLETED
    assert any(
        item["event"] == "ticket_resolved"
        and item["target"] == "retrieve_policy"
        for item in recovered.transition_history
    )


def test_graph_service_filters_shared_tasks_by_graph_name(tmp_path):
    graph_service, _, store = service(tmp_path)
    unrelated = SharedGraphState(
        run_id="other-run",
        graph_name="another_graph",
        current_node="wait_for_admin",
        status=RunStatus.WAITING_HITL,
    )
    store.create_run(unrelated)
    store.create_hitl_task(
        {
            "task_id": "other-hitl",
            "run_id": unrelated.run_id,
            "node": "wait_for_admin",
            "status": HITLStatus.PENDING.value,
            "reason": "Other graph decision.",
            "request": {},
            "state": unrelated.to_dict(),
        }
    )
    start(graph_service)

    assert [task["run_id"] for task in graph_service.pending_hitl_tasks()] == [
        "graph-2-run"
    ]
