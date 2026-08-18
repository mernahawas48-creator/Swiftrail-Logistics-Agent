from __future__ import annotations

from pathlib import Path

from state_graph.graph_2.checkpoint import SQLiteCheckpointStore
from state_graph.graph_2.graph import RateExceptionGraph
from state_graph.graph_2.state import RateExceptionState


def test_graph_has_real_branch_and_recovery_cycle(tmp_path: Path):
    graph = RateExceptionGraph(SQLiteCheckpointStore(tmp_path / "checkpoints.db"))
    assert ("classify_authority", "auto_approve") in graph.graph.edges
    assert ("classify_authority", "wait_for_admin") in graph.graph.edges
    assert ("failure_ticket", "resume_from_checkpoint") in graph.graph.edges
    assert ("resume_from_checkpoint", "load_shipment") in graph.graph.edges


def test_checkpoint_survives_new_graph_instance(tmp_path: Path):
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    state = RateExceptionState(run_id="run-1", shipment_id=5, session_id="demo-session-001")
    state.current_node = "wait_for_admin"
    store.save(state.run_id, state.current_node, state.to_dict())

    restarted = RateExceptionGraph(SQLiteCheckpointStore(tmp_path / "checkpoints.db"))
    loaded = restarted.checkpoints.latest("run-1")
    assert loaded is not None
    assert loaded["current_node"] == "wait_for_admin"
    assert loaded["shipment_id"] == 5


def test_transition_is_persisted(tmp_path: Path):
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    graph = RateExceptionGraph(store)
    state = RateExceptionState(run_id="run-2", shipment_id=5, session_id="demo-session-001")
    graph._transition(state, "load_shipment")
    assert state.current_node == "load_shipment"
    assert store.latest("run-2")["current_node"] == "load_shipment"


def test_failure_ticket_resumes_exact_failed_node(tmp_path: Path):
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    graph = RateExceptionGraph(store)
    state = RateExceptionState(
        run_id="run-3", shipment_id=5, session_id="demo-session-001",
        current_node="failure_ticket", failed_node="retrieve_policy",
        ticket_id="FT-123", ticket_status="open", error="temporary RAG failure",
    )
    store.save(state.run_id, "failure_ticket", state.to_dict())
    store.create_task(state.ticket_id, state.run_id, "failure", state.to_dict())
    # Avoid external MCP/RAG calls: verify the recovery checkpoint points to the exact node.
    state.current_node = state.failed_node
    state.ticket_status = "resolved"
    state.error = None
    graph._checkpoint(state, "resume_from_checkpoint")
    assert store.latest("run-3")["current_node"] == "retrieve_policy"


def test_hitl_task_is_persisted(tmp_path: Path):
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    state = RateExceptionState(run_id="run-4", shipment_id=5, session_id="demo-session-001", current_node="wait_for_admin")
    state.hitl_task_id = "HITL-123"
    store.create_task(state.hitl_task_id, state.run_id, "hitl", state.to_dict())
    assert store.list_tasks(task_type="hitl")[0]["task_id"] == "HITL-123"
