"""Real pytest coverage for AgentLoop.

The RAG pipeline (HybridRAG) is mocked here since it needs a live
Qdrant instance + MISTRAL_API_KEY -- that's covered separately by
rag/tests/test_hybrid_rag_integration.py, not duplicated here. This
file is about the agent loop's own wiring: routing, memory overflow,
verification fallback, and scratchpad state.
"""

from unittest.mock import MagicMock

import pytest

import agent.agent_loop as agent_loop_module
from agent.agent_loop import AgentLoop


@pytest.fixture
def agent(monkeypatch):
    mock_rag = MagicMock()
    monkeypatch.setattr(agent_loop_module, "HybridRAG", lambda: mock_rag)
    a = AgentLoop()
    a._mock_rag = mock_rag  # exposed for tests that need to configure it
    return a


def test_start_creates_a_session(agent):
    session_id = agent.start(customer_id="12")
    assert agent.session_manager.get_session(session_id) is not None


def test_process_unknown_session_raises(agent):
    with pytest.raises(ValueError):
        agent.process("not-a-real-session", [{"role": "employee", "content": "hi"}])


def test_tool_routing_matches_query_router_labels(agent):
    """Regression test: QueryRouter returns '*_tool' labels
    (shipment_tool, invoice_tool, ...) that must resolve to the
    matching call_mcp_tool category, not fall through to 'unknown'."""
    session_id = agent.start(customer_id="12")
    result = agent.process(
        session_id,
        [{"role": "employee", "content": "what is the credit hold status here"}],
    )
    assert result["category"] == "credit_tool"
    assert result["evidence"]["source"] == "finance_database"
    assert result["verified"] is True


def test_context_strategy_is_applied(agent):
    session_id = agent.start(customer_id="12")
    messages = [{"role": "employee", "content": "checking shipment 1"}] * 15
    result = agent.process(session_id, messages)
    # SlidingWindow(max_messages=10) must actually prune the transcript
    assert len(result["context"]) <= 10


def test_short_term_overflow_triggers_promote_or_drop(agent):
    session_id = agent.start(customer_id="12")
    result = None
    for i in range(14):
        result = agent.process(
            session_id, [{"role": "employee", "content": f"checking shipment {i}"}]
        )
    routed_notes = [
        note for note in result["scratchpad"]
        if isinstance(note, str) and note.startswith("memory routed ->")
    ]
    assert len(routed_notes) >= 1


def test_unverified_evidence_returns_safe_fallback(agent):
    session_id = agent.start(customer_id="12")
    # A query that matches none of QueryRouter's keyword buckets falls
    # through to "context", which call_mcp_tool doesn't recognize --
    # evidence["data"] stays None, so the loop must not fabricate an answer.
    result = agent.process(session_id, [{"role": "employee", "content": "hello there"}])
    assert result["verified"] is False
    assert "cannot provide a reliable answer" in result["answer"]


