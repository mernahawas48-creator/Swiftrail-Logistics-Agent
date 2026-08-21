from __future__ import annotations

import os

import pytest

from rag.agentic_rag.controller import AgenticRAG

pytestmark = pytest.mark.skipif(
    os.getenv(
        "RUN_AGENTIC_RAG_INTEGRATION"
    ) != "1",
    reason=(
        "Set RUN_AGENTIC_RAG_INTEGRATION=1 "
        "after populating Qdrant."
    ),
)


class EvidenceEchoGenerator:
    model_name = "evidence-echo"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt):
        self.prompt = prompt
        return (
            "Only an authenticated finance manager may "
            "release an active severe credit hold [1]."
        )


def test_real_agentic_retrieval_returns_trace_and_evidence():
    generator = EvidenceEchoGenerator()

    response = AgenticRAG(
        generator=generator,
        max_attempts=2,
    ).answer(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    assert response.answer.endswith("[1].")
    assert response.sources
    assert response.sources[0].doc_id == (
        "credit_hold_policy"
    )
    assert response.sources[0].section_id == "CH-3"

    actions = [
        step.action
        for step in response.trace
    ]

    assert "plan" in actions
    assert "retrieve" in actions
    assert "grade_evidence" in actions
    assert "generate_answer" in actions

    assert "CH-3" in generator.prompt
    assert "AGENT TRACE" in generator.prompt


def test_real_agentic_exact_section_routing():
    response = AgenticRAG(
        generator=EvidenceEchoGenerator(),
    ).answer(
        "RE-2",
        role="finance_manager",
        top_k=3,
    )

    assert response.sources
    assert response.sources[0].section_id == "RE-2"
    assert response.attempts == 1
