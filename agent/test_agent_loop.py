"""Unit coverage for AgentLoop without live DB, Qdrant, or Mistral calls."""

from typing import Any

import pytest

from agent.agent_loop import AgentLoop


class FakeMCPGateway:
    def __init__(self, *, success: bool = True):
        self.success = success
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        destination: str,
        query: str,
        *,
        session_id: str,
        customer_id: int | None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "destination": destination,
                "query": query,
                "session_id": session_id,
                "customer_id": customer_id,
            }
        )
        return {
            "source": "list_customer_credit_holds",
            "data": {"active_holds": []} if self.success else None,
            "code": "CREDIT_HOLDS_RETRIEVED" if self.success else "FAILED",
            "message": "Retrieved credit holds." if self.success else "failed",
            "success": self.success,
        }


@pytest.fixture
def gateway() -> FakeMCPGateway:
    return FakeMCPGateway()


@pytest.fixture
def agent(gateway: FakeMCPGateway, tmp_path) -> AgentLoop:
    from memory.episodic_store import EpisodicMemory
    from memory.semantic_store import SemanticMemory

    episodic = EpisodicMemory(str(tmp_path / "episodic.db"))
    semantic = SemanticMemory(str(tmp_path / "semantic.db"))
    return AgentLoop(
        mcp_gateway=gateway,
        episodic_memory=episodic,
        semantic_memory=semantic,
    )


def test_start_creates_a_session(agent: AgentLoop):
    session_id = agent.start(customer_id="12")
    assert agent.session_manager.get_session(session_id) is not None


def test_process_unknown_session_raises(agent: AgentLoop):
    with pytest.raises(ValueError, match="Session not found"):
        agent.process("not-a-real-session", [{"role": "employee", "content": "hi"}])


def test_tool_routing_calls_real_gateway_contract(
    agent: AgentLoop,
    gateway: FakeMCPGateway,
):
    session_id = agent.start(customer_id="12")
    result = agent.process(
        session_id,
        [{"role": "employee", "content": "what is the credit hold status?"}],
    )

    assert result["category"] == "credit"
    assert result["evidence"][0]["source"] == "list_customer_credit_holds"
    assert result["verified"] is True
    assert gateway.calls[0]["customer_id"] == 12


def test_context_strategy_is_applied(agent: AgentLoop):
    session_id = agent.start(customer_id="12")
    messages = [{"role": "employee", "content": "checking shipment 1"}] * 15
    result = agent.process(session_id, messages)
    assert len(result["context"]) <= 10


def test_short_term_overflow_is_sent_to_promote_drop_router(agent: AgentLoop):
    session_id = agent.start(customer_id="12")
    for i in range(14):
        agent.process(
            session_id,
            [{"role": "employee", "content": f"checking shipment {i}"}],
        )

    assert agent._promote_router is not None
    assert len(agent._promote_router.decision_log) >= 1
    assert all(
        decision.action in {"forget", "episodic"}
        for decision in agent._promote_router.decision_log
    )


def test_unverified_mcp_evidence_returns_safe_fallback(tmp_path):
    failed_gateway = FakeMCPGateway(success=False)
    agent = AgentLoop(mcp_gateway=failed_gateway)
    session_id = agent.start(customer_id="12")

    result = agent.process(
        session_id,
        [{"role": "employee", "content": "show the credit hold"}],
    )

    assert result["verified"] is False
    assert "cannot provide a reliable answer" in result["answer"]
    assert result["evidence"] == []


def test_operational_route_without_gateway_never_fabricates_data():
    agent = AgentLoop()
    session_id = agent.start(customer_id="12")
    result = agent.process(
        session_id,
        [{"role": "employee", "content": "show customer account"}],
    )

    assert result["verified"] is False
    assert result["evidence"] == []
    assert "No MCP gateway" in result["verification"]
