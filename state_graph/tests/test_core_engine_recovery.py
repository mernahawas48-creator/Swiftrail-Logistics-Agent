from pathlib import Path

import pytest

from state_graph.core.engine import GraphEngine
from state_graph.core.nodes import (
    ExternalWaitNode,
    FunctionNode,
    HITLNode,
    NodeResult,
)
from state_graph.core.registry import GraphRegistry
from state_graph.core.sqlite_store import SQLiteCheckpointStore
from state_graph.core.transitions import GraphDefinition
from state_graph.core.types import (
    HITLStatus,
    NodeDirective,
    RunStatus,
    TicketStatus,
)


def _engine(tmp_path: Path, graph: GraphDefinition, **services) -> GraphEngine:
    registry = GraphRegistry()
    registry.register(graph)
    return GraphEngine(
        registry,
        SQLiteCheckpointStore(tmp_path / "graph.db"),
        services=services,
    )


def test_external_wait_survives_restart_without_reexecuting_completed_node(
    tmp_path: Path,
):
    calls = {"load": 0}

    def load(state, context):
        del state, context
        calls["load"] += 1
        return NodeResult("wait", {"shipment": {"id": 5}})

    def finish(state, context):
        del context
        return NodeResult(
            "END",
            {"choice": state.data["external_input"]["choice"]},
            NodeDirective.COMPLETE,
        )

    graph = GraphDefinition(
        name="external-demo",
        start_node="load",
        nodes={
            "load": FunctionNode("load", load),
            "wait": ExternalWaitNode(
                "wait",
                "finish",
                "Customer response required.",
                lambda state: {"shipment_id": state.data["shipment"]["id"]},
            ),
            "finish": FunctionNode("finish", finish),
        },
        transitions={
            "load": frozenset({"wait"}),
            "wait": frozenset({"finish"}),
            "finish": frozenset({"END"}),
        },
    )
    engine = _engine(tmp_path, graph)
    waiting = engine.start("external-demo", {}, run_id="run-external")
    assert waiting.status is RunStatus.WAITING_EXTERNAL
    assert calls["load"] == 1

    restarted = _engine(tmp_path, graph)
    completed = restarted.resume_external(
        "run-external", {"choice": "redeliver"}
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["choice"] == "redeliver"
    assert calls["load"] == 1


def test_hitl_is_persisted_and_resumes_only_after_admin_decision(tmp_path: Path):
    def evaluate(state, context):
        del state, context
        return NodeResult("admin_review", {"risk": "high"})

    def apply_decision(state, context):
        del context
        return NodeResult(
            "END",
            {"applied": state.data["admin_decision"]["approved"]},
            NodeDirective.COMPLETE,
        )

    graph = GraphDefinition(
        name="hitl-demo",
        start_node="evaluate",
        nodes={
            "evaluate": FunctionNode("evaluate", evaluate),
            "admin_review": HITLNode(
                "admin_review",
                "apply_decision",
                lambda state: f"Review {state.data['risk']} risk reroute.",
                lambda state: {"risk": state.data["risk"]},
            ),
            "apply_decision": FunctionNode("apply_decision", apply_decision),
        },
        transitions={
            "evaluate": frozenset({"admin_review"}),
            "admin_review": frozenset({"apply_decision"}),
            "apply_decision": frozenset({"END"}),
        },
    )
    engine = _engine(tmp_path, graph)
    waiting = engine.start("hitl-demo", {}, run_id="run-hitl")
    assert waiting.status is RunStatus.WAITING_HITL
    tasks = engine.store.list_hitl_tasks()
    assert len(tasks) == 1

    completed = engine.resolve_hitl(
        tasks[0]["task_id"],
        approved=True,
        note="Verified destination and approved reroute.",
        admin_employee_id=3,
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["applied"] is True
    assert engine.store.list_tickets() == []


def test_failure_ticket_requires_open_investigating_resolved_lifecycle(
    tmp_path: Path,
):
    service = {"available": False}

    def unstable(state, context):
        del state, context
        if not service["available"]:
            raise RuntimeError("MCP is unavailable")
        return NodeResult("END", {"recovered": True}, NodeDirective.COMPLETE)

    graph = GraphDefinition(
        name="failure-demo",
        start_node="unstable",
        nodes={"unstable": FunctionNode("unstable", unstable)},
        transitions={"unstable": frozenset({"END"})},
    )
    engine = _engine(tmp_path, graph)
    failed = engine.start("failure-demo", {}, run_id="run-failure")
    assert failed.status is RunStatus.WAITING_TICKET
    ticket = engine.store.list_tickets(TicketStatus.OPEN)[0]
    assert ticket["failed_node"] == "unstable"

    with pytest.raises(ValueError, match="cannot move"):
        engine.store.update_ticket(
            ticket["ticket_id"], status=TicketStatus.RESOLVED
        )
    investigating = engine.investigate_ticket(ticket["ticket_id"])
    assert investigating["status"] == "investigating"

    service["available"] = True
    completed = engine.resolve_ticket(
        ticket["ticket_id"],
        resolution_note="MCP service restored and validated.",
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["recovered"] is True
    assert engine.store.get_ticket(ticket["ticket_id"])["status"] == "resolved"


def test_cached_node_result_is_replayed_after_crash_without_node_call(
    tmp_path: Path,
):
    calls = {"write": 0}

    def write(state, context):
        del state, context
        calls["write"] += 1
        return NodeResult("END", {"write_id": 42}, NodeDirective.COMPLETE)

    graph = GraphDefinition(
        name="receipt-demo",
        start_node="write",
        nodes={"write": FunctionNode("write", write)},
        transitions={"write": frozenset({"END"})},
    )
    engine = _engine(tmp_path, graph)
    from state_graph.core.state import SharedGraphState

    state = SharedGraphState("run-receipt", "receipt-demo", "write")
    engine.store.create_run(state)
    engine.store.save_node_result(
        "run-receipt:0:write",
        run_id=state.run_id,
        node="write",
        result=NodeResult(
            "END", {"write_id": 42}, NodeDirective.COMPLETE
        ).to_dict(),
    )

    completed = engine.run("run-receipt")
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["write_id"] == 42
    assert calls["write"] == 0


def test_hitl_resume_recovers_if_process_crashes_after_decision_commit(
    tmp_path: Path,
):
    graph = GraphDefinition(
        name="hitl-crash-demo",
        start_node="review",
        nodes={
            "review": HITLNode(
                "review",
                "finish",
                lambda state: "Admin approval is required.",
                lambda state: {"run_id": state.run_id},
            ),
            "finish": FunctionNode(
                "finish",
                lambda state, context: NodeResult(
                    "END",
                    {"approved": state.data["admin_decision"]["approved"]},
                    NodeDirective.COMPLETE,
                ),
            ),
        },
        transitions={
            "review": frozenset({"finish"}),
            "finish": frozenset({"END"}),
        },
    )
    engine = _engine(tmp_path, graph)
    engine.start("hitl-crash-demo", {}, run_id="run-hitl-crash")
    task = engine.store.list_hitl_tasks()[0]
    decision = {
        "approved": True,
        "note": "Admin checked the request and approved it.",
        "admin_employee_id": 3,
    }
    engine.store.update_hitl_task(
        task["task_id"], status=HITLStatus.APPROVED, decision=decision
    )

    restarted = _engine(tmp_path, graph)
    completed = restarted.resolve_hitl(
        task["task_id"],
        approved=True,
        note=decision["note"],
        admin_employee_id=3,
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["approved"] is True


def test_ticket_resume_recovers_if_process_crashes_after_resolution_commit(
    tmp_path: Path,
):
    service = {"available": False}

    def unstable(state, context):
        del state, context
        if not service["available"]:
            raise RuntimeError("MCP is unavailable")
        return NodeResult("END", {"recovered": True}, NodeDirective.COMPLETE)

    graph = GraphDefinition(
        name="ticket-crash-demo",
        start_node="unstable",
        nodes={"unstable": FunctionNode("unstable", unstable)},
        transitions={"unstable": frozenset({"END"})},
    )
    engine = _engine(tmp_path, graph)
    engine.start("ticket-crash-demo", {}, run_id="run-ticket-crash")
    ticket = engine.store.list_tickets(TicketStatus.OPEN)[0]
    engine.investigate_ticket(ticket["ticket_id"])
    engine.store.update_ticket(
        ticket["ticket_id"],
        status=TicketStatus.RESOLVED,
        resolution_note="MCP service was restored and verified.",
    )

    service["available"] = True
    restarted = _engine(tmp_path, graph)
    completed = restarted.resolve_ticket(
        ticket["ticket_id"],
        resolution_note="MCP service was restored and verified.",
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.data["recovered"] is True
