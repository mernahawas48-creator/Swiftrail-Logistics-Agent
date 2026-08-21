from dataclasses import dataclass, field
from pathlib import Path

from state_graph.core.engine import GraphEngine
from state_graph.core.registry import GraphRegistry
from state_graph.core.sqlite_store import SQLiteCheckpointStore
from state_graph.core.types import RunStatus, TicketStatus
from state_graph.graph_1.graph import GRAPH_NAME, build_delivery_recovery_graph
from state_graph.graph_1.llm import MistralRecoveryDecomposer


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        assert "task-decomposition" in prompt
        return """{
          "steps": [
            "Inspect shipment",
            "Verify delivery policy",
            "Collect customer input",
            "Apply safe action"
          ],
          "customer_question": "Which delivery recovery option do you prefer?",
          "policy_query": "What delivery rerouting rules and approvals apply?"
        }"""


@dataclass
class FakeVerification:
    passed: bool = True
    reason: str = "Evidence is grounded."


@dataclass
class FakeSource:
    doc_id: str = "delivery_exception_policy"
    section_id: str = "DR-3"
    number: int = 1


@dataclass
class FakeRAGResponse:
    answer: str = "Unverified destinations require admin approval [1]."
    sources: tuple = (FakeSource(),)
    verification: FakeVerification = field(default_factory=FakeVerification)


class FakePolicyRAG:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, query, **kwargs):
        self.calls += 1
        assert "rerouting" in query
        assert kwargs["doc_ids"] == ("delivery_exception_policy",)
        return FakeRAGResponse()


class FakeDeliveryTools:
    def __init__(self) -> None:
        self.shipment = {
            "id": 6,
            "customer_id": 1,
            "destination": "Cairo Yard",
            "status": "delivery_exception",
        }
        self.apply_calls = 0
        self.available = True

    def load_shipment(self, **kwargs):
        assert kwargs["shipment_id"] == 6
        return dict(self.shipment)

    def create_case(self, **kwargs):
        assert kwargs["shipment_id"] == 6
        return {"id": 10, "case_status": "waiting_customer"}

    def apply_reroute(self, *, request, **kwargs):
        del kwargs
        if not self.available:
            raise RuntimeError("MCP reroute tool unavailable")
        self.apply_calls += 1
        self.shipment["destination"] = request["new_destination"]
        self.shipment["status"] = "pending"
        return {"case_id": request["case_id"], "destination": request["new_destination"]}


def _engine(tmp_path: Path, tools: FakeDeliveryTools | None = None):
    registry = GraphRegistry()
    graph = build_delivery_recovery_graph()
    registry.register(graph)
    generator = FakeGenerator()
    rag = FakePolicyRAG()
    delivery_tools = tools or FakeDeliveryTools()
    engine = GraphEngine(
        registry,
        SQLiteCheckpointStore(tmp_path / "graph1.db"),
        services={
            "delivery_tools": delivery_tools,
            "task_decomposer": MistralRecoveryDecomposer(generator),
            "policy_rag": rag,
        },
    )
    return engine, generator, rag, delivery_tools


def _input():
    return {
        "shipment_id": 6,
        "session_id": "graph1-session",
        "employee_id": 1,
        "failure_reason": "Customer was unavailable at the delivery destination.",
    }


def test_graph_1_uses_two_llm_additions_and_waits_for_customer(tmp_path: Path):
    engine, generator, rag, _ = _engine(tmp_path)
    state = engine.start(GRAPH_NAME, _input(), run_id="graph1-llm")
    assert state.status is RunStatus.WAITING_EXTERNAL
    assert state.current_node == "wait_for_customer"
    assert generator.calls == 1
    assert rag.calls == 1
    assert state.data["policy_sources"][0]["section_id"] == "DR-3"


def test_customer_and_admin_rejection_cycles_are_real(tmp_path: Path):
    engine, _, _, _ = _engine(tmp_path)
    waiting = engine.start(GRAPH_NAME, _input(), run_id="graph1-cycle")
    second_wait = engine.resume_external(
        waiting.run_id, {"action": "request_new_options"}
    )
    assert second_wait.status is RunStatus.WAITING_EXTERNAL
    assert second_wait.data["option_round"] == 2

    hitl = engine.resume_external(
        second_wait.run_id,
        {
            "action": "reroute",
            "new_destination": "Giza Warehouse",
            "destination_verified": False,
            "estimated_cost": 700,
        },
    )
    assert hitl.status is RunStatus.WAITING_HITL
    task = engine.store.list_hitl_tasks()[0]
    revised = engine.resolve_hitl(
        task["task_id"],
        approved=False,
        note="Destination evidence is not sufficient for approval.",
        admin_employee_id=3,
    )
    assert revised.status is RunStatus.WAITING_EXTERNAL
    assert revised.data["option_round"] == 3


def test_approved_hitl_applies_reroute_and_completes(tmp_path: Path):
    engine, _, _, tools = _engine(tmp_path)
    waiting = engine.start(GRAPH_NAME, _input(), run_id="graph1-approved")
    engine.resume_external(
        waiting.run_id,
        {
            "action": "reroute",
            "new_destination": "Giza Warehouse",
            "destination_verified": False,
            "estimated_cost": 700,
        },
    )
    task = engine.store.list_hitl_tasks()[0]
    completed = engine.resolve_hitl(
        task["task_id"],
        approved=True,
        note="Finance manager verified and approved the destination.",
        admin_employee_id=3,
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["final_status"] == "resolved"
    assert tools.apply_calls == 1


def test_mcp_failure_opens_ticket_and_resumes_failed_node(tmp_path: Path):
    tools = FakeDeliveryTools()
    tools.available = False
    engine, _, _, _ = _engine(tmp_path, tools)
    waiting = engine.start(GRAPH_NAME, _input(), run_id="graph1-failure")
    failed = engine.resume_external(
        waiting.run_id,
        {
            "action": "redeliver",
            "new_destination": "Cairo Yard",
            "destination_verified": True,
            "estimated_cost": 0,
        },
    )
    assert failed.status is RunStatus.WAITING_TICKET
    ticket = engine.store.list_tickets(TicketStatus.OPEN)[0]
    assert ticket["failed_node"] == "apply_reroute"
    engine.investigate_ticket(ticket["ticket_id"])
    tools.available = True
    completed = engine.resolve_ticket(
        ticket["ticket_id"],
        resolution_note="MCP reroute tool restored and checked.",
    )
    assert completed.status is RunStatus.COMPLETED
    assert tools.apply_calls == 1
