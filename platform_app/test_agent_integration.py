from __future__ import annotations

from platform_app.agent_integration import (
    MEMORY_RAG_AGENT_ID,
    PLANNING_AGENT_ID,
    PlatformAgentIntegration,
)


class FakeMemoryAgent:
    def __init__(self) -> None:
        self.customer_id: int | None = None
        self.role: str | None = None

    def start(self, customer_id):
        self.customer_id = customer_id
        return "memory-run"

    def process(self, session_id, messages, *, role):
        assert session_id == "memory-run"
        self.role = role
        return {
            "session_id": session_id,
            "category": "rag",
            "verified": True,
            "answer": f"Verified answer for: {messages[-1]['content']}",
            "context": messages,
            "evidence": [{"section_id": "CH-2"}],
            "verification": "The answer is supported.",
        }


def successful_plan(**kwargs):
    return {
        "method": kwargs["method"],
        "result": "Release only after the verified blockers are cleared.",
        "artifact_path": "artifacts/planning-run.json",
        "action_results": [
            {
                "action": "release_credit_hold",
                "success": True,
                "message": "Verified.",
            }
        ],
    }


def test_memory_rag_chat_keeps_a_customer_scoped_session():
    memory = FakeMemoryAgent()
    platform = PlatformAgentIntegration(lambda: memory, successful_plan)

    started = platform.chat(
        MEMORY_RAG_AGENT_ID,
        "start customer 3, role finance_manager",
        None,
    )
    assert started["run_id"] == "memory-run"
    assert started["status"] == "active"
    assert memory.customer_id == 3

    answered = platform.chat(
        MEMORY_RAG_AGENT_ID,
        "What does policy CH-2 require?",
        "memory-run",
    )
    assert answered["current_node"] == "rag"
    assert answered["reply"].startswith("Verified answer")
    assert memory.role == "finance_manager"

    run = platform.get_run("memory-run")
    assert run["state"]["last_result"]["verified"] is True
    assert run["history"][-1]["node_name"] == "rag"


def test_memory_rag_requires_an_explicit_customer_session():
    platform = PlatformAgentIntegration(FakeMemoryAgent, successful_plan)

    response = platform.chat(MEMORY_RAG_AGENT_ID, "What is policy CH-2?", None)

    assert "run_id" not in response
    assert "start customer 3" in response["reply"].lower()


def test_memory_rag_failure_is_visible_in_run_status():
    class FailingMemoryAgent(FakeMemoryAgent):
        def process(self, session_id, messages, *, role):
            del session_id, messages, role
            raise RuntimeError("Qdrant unavailable")

    platform = PlatformAgentIntegration(FailingMemoryAgent, successful_plan)
    started = platform.chat(MEMORY_RAG_AGENT_ID, "start customer 3", None)
    response = platform.chat(
        MEMORY_RAG_AGENT_ID,
        "What is the credit-hold policy?",
        started["run_id"],
    )

    assert response["status"] == "failed"
    assert response["current_node"] == "agent_failed"
    run = platform.get_run(started["run_id"])
    assert run["state"]["error"] == "Qdrant unavailable"


def test_planning_chat_runs_the_existing_orchestrator_adapter():
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return successful_plan(**kwargs)

    platform = PlatformAgentIntegration(FakeMemoryAgent, runner)
    response = platform.chat(
        PLANNING_AGENT_ID,
        "plan shipment 3, customer 3, employee 1, method dynamic",
        None,
    )

    assert response["status"] == "completed"
    assert response["current_node"] == "complete"
    assert "release_credit_hold: success" in response["reply"]
    assert calls[0]["shipment_id"] == 3
    assert calls[0]["customer_id"] == 3
    assert calls[0]["employee_id"] == 1
    assert calls[0]["method"] == "dynamic"

    run = platform.get_run(response["run_id"])
    assert run["history"][-1]["node_name"] == "complete"
    assert run["state"]["outcome"]["artifact_path"].endswith("planning-run.json")


def test_planning_failure_is_visible_in_run_status():
    def failing_runner(**kwargs):
        del kwargs
        raise RuntimeError("MCP server unavailable")

    platform = PlatformAgentIntegration(FakeMemoryAgent, failing_runner)
    response = platform.chat(
        PLANNING_AGENT_ID,
        "plan shipment 3, customer 3, employee 1",
        None,
    )

    assert response["status"] == "failed"
    assert response["current_node"] == "planning_failed"
    run = platform.get_run(response["run_id"])
    assert run["state"]["error"] == "MCP server unavailable"
