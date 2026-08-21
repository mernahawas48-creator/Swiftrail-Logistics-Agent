from __future__ import annotations

import os

import pytest

from rag.naive_rag.pipeline import NaiveRAG

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NAIVE_RAG_INTEGRATION") != "1",
    reason=(
        "Set RUN_NAIVE_RAG_INTEGRATION=1 after "
        "running the ingestion pipeline."
    ),
)


class EvidenceEchoGenerator:
    """Proves the retrieved context reached generation."""

    model_name = "evidence-echo"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return (
            "An authenticated finance manager may release "
            "an active severe credit hold [1]."
        )


def test_real_retrieval_feeds_the_naive_rag_prompt():
    generator = EvidenceEchoGenerator()

    response = NaiveRAG(
        generator=generator
    ).answer(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    assert response.retrieved_count == 3
    assert response.sources
    assert response.sources[0].doc_id == (
        "credit_hold_policy"
    )
    assert response.sources[0].section_id == "CH-3"

    assert "credit_hold_policy" in generator.prompt
    assert "CH-3" in generator.prompt
    assert "Severe Release" in generator.prompt
    assert "[1]" in response.answer
