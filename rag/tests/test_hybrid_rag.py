from __future__ import annotations

from types import SimpleNamespace

from rag.hybrid_rag.pipeline import HybridRAG
from rag.naive_rag.pipeline import NO_CONTEXT_ANSWER


class StubSearch:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results


class StubGenerator:
    model_name = "stub"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self.answer


class SequenceGenerator:
    model_name = "sequence-stub"

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0
        self.prompts = []

    def generate(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        return next(self.answers)


def _result():
    return SimpleNamespace(
        chunk_id="re2",
        fused_score=0.03,
        dense_rank=1,
        sparse_rank=1,
        text=(
            "A discount above 15 percent requires an authenticated "
            "finance manager and explicit human approval or rejection."
        ),
        metadata={
            "doc_id": "rate_exception_policy",
            "title": "Rate Exception Policy",
            "section_id": "RE-2",
            "section_title": "Above-Authority Discount",
            "keywords": ["discount"],
        },
    )


def test_hybrid_rag_generates_verified_answer():
    generator = StubGenerator(
        "A discount above 15 percent requires a finance manager [1]."
    )

    response = HybridRAG(
        searcher=StubSearch([_result()]),
        generator=generator,
    ).answer(
        "RE-2",
        role="finance_manager",
        top_k=3,
    )

    assert generator.calls == 1
    assert response.sources[0].section_id == "RE-2"
    assert response.verification is not None
    assert response.verification.passed


def test_hybrid_rag_abstains_when_retrieval_is_irrelevant():
    generator = StubGenerator(
        "This must not be generated."
    )
    irrelevant = _result()
    irrelevant.metadata["section_id"] = "SP-2"

    response = HybridRAG(
        searcher=StubSearch([irrelevant]),
        generator=generator,
    ).answer(
        "RE-2",
        role="finance_manager",
    )

    assert response.answer == NO_CONTEXT_ANSWER
    assert generator.calls == 0


def test_hybrid_rag_replaces_unsupported_generation():
    generator = StubGenerator(
        "The threshold is 20 percent [1]."
    )

    response = HybridRAG(
        searcher=StubSearch([_result()]),
        generator=generator,
    ).answer(
        "What threshold applies to an above-authority discount?",
        role="finance_manager",
    )

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.verification is not None
    assert not response.verification.passed


def test_hybrid_rag_corrects_citation_format_once_before_abstaining():
    generator = SequenceGenerator(
        [
            "The policy answer is as follows. A finance manager is required [1].",
            "A discount above 15 percent requires a finance manager [1].",
        ]
    )

    response = HybridRAG(
        searcher=StubSearch([_result()]),
        generator=generator,
    ).answer(
        "What threshold applies to an above-authority discount?",
        role="finance_manager",
    )

    assert generator.calls == 2
    assert "VERIFIER FEEDBACK" in generator.prompts[1]
    assert response.answer.endswith("[1].")
    assert response.verification is not None
    assert response.verification.passed
