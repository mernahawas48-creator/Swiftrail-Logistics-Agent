from pathlib import Path

import pytest

from state_graph.core.registry import GraphRegistry
from state_graph.core.sqlite_store import SQLiteCheckpointStore
from state_graph.core.state import SharedGraphState
from state_graph.core.transitions import GraphDefinition
from state_graph.core.types import HITLStatus, RunStatus, TicketStatus


def _store(tmp_path: Path) -> SQLiteCheckpointStore:
    return SQLiteCheckpointStore(tmp_path / "state.db")


def test_definition_validates_transitions_and_detects_cycle():
    graph = GraphDefinition(
        name="demo",
        start_node="first",
        nodes={"first": object(), "wait": object(), "revise": object()},
        transitions={
            "first": frozenset({"wait"}),
            "wait": frozenset({"revise", "END"}),
            "revise": frozenset({"wait"}),
        },
    )
    assert graph.has_cycle() is True
    assert graph.allows("wait", "END") is True
    with pytest.raises(ValueError, match="does not allow"):
        graph.require_transition("first", "END")


def test_registry_rejects_duplicate_graph_names():
    graph = GraphDefinition(
        name="demo",
        start_node="first",
        nodes={"first": object()},
        transitions={"first": frozenset({"END"})},
    )
    registry = GraphRegistry()
    registry.register(graph)
    assert registry.get("demo") is graph
    with pytest.raises(ValueError, match="already registered"):
        registry.register(graph)


def test_checkpoint_survives_new_store_instance(tmp_path: Path):
    store = _store(tmp_path)
    state = SharedGraphState("run-1", "demo", "first")
    store.create_run(state)
    state.record_transition("first", "second", "node_completed")
    store.save_checkpoint(state, node="second", event="node_completed")

    restarted = _store(tmp_path).load_run("run-1")
    assert restarted is not None
    assert restarted.current_node == "second"
    assert restarted.revision == 1
    assert len(_store(tmp_path).checkpoint_history("run-1")) == 2
    assert _store(tmp_path).list_runs("demo")[0].run_id == "run-1"
    assert _store(tmp_path).list_runs("another_graph") == []


def test_node_execution_result_is_durable_and_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    state = SharedGraphState("run-2", "demo", "write_node")
    store.create_run(state)
    result = {"next_node": "verify", "updates": {"write_id": 10}}
    store.save_node_result(
        "run-2:0:write_node",
        run_id=state.run_id,
        node="write_node",
        result=result,
    )
    store.save_node_result(
        "run-2:0:write_node",
        run_id=state.run_id,
        node="write_node",
        result={"should_not": "replace"},
    )
    assert store.load_node_result("run-2:0:write_node") == result


def test_hitl_and_ticket_records_use_separate_tables(tmp_path: Path):
    store = _store(tmp_path)
    state = SharedGraphState(
        "run-3", "demo", "review", status=RunStatus.WAITING_HITL
    )
    store.create_run(state)
    store.create_hitl_task(
        {
            "task_id": "HITL-1",
            "run_id": state.run_id,
            "node": "review",
            "status": HITLStatus.PENDING.value,
            "reason": "Admin authority required.",
            "request": {"choice": "reroute"},
            "state": state.to_dict(),
        }
    )
    store.create_ticket(
        {
            "ticket_id": "FT-1",
            "run_id": state.run_id,
            "failed_node": "write",
            "status": TicketStatus.OPEN.value,
            "error_type": "RuntimeError",
            "error_message": "MCP unavailable",
            "state": state.to_dict(),
        }
    )
    assert store.get_hitl_task("HITL-1")["status"] == "pending"
    assert store.get_ticket("FT-1")["status"] == "open"
