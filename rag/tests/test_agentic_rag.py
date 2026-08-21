from __future__ import annotations

from types import SimpleNamespace

from rag.agentic_rag.controller import (
    SAFE_NO_EVIDENCE_ANSWER,
    AgenticRAG,
    CorpusQueryRewriter,
    EvidenceAssessment,
)


class StubRetriever:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(
            {
                "query": query,
                **kwargs,
            }
        )

        if not self.responses:
            return []

        return self.responses.pop(0)


class StubGenerator:
    model_name = "stub-generator"

    def __init__(self, answer):
        self.answer_text = answer
        self.calls = 0
        self.prompt = ""

    def generate(self, prompt):
        self.calls += 1
        self.prompt = prompt
        return self.answer_text


class SequenceGrader:
    def __init__(self, assessments):
        self.assessments = list(assessments)
        self.calls = 0
        self.result_counts = []

    def grade(self, original_query, results):
        self.calls += 1
        self.result_counts.append(len(results))
        return self.assessments.pop(0)


class StubRewriter:
    def __init__(self, rewritten):
        self.rewritten = rewritten
        self.calls = 0

    def rewrite(self, original_query, results):
        self.calls += 1
        return self.rewritten


def _result(
    *,
    chunk_id="chunk-ch3",
    doc_id="credit_hold_policy",
    section_id="CH-3",
    section_title="Severe Release",
):
    return SimpleNamespace(
        chunk_id=chunk_id,
        fused_score=0.032,
        dense_rank=1,
        sparse_rank=1,
        text=(
            "Only an authenticated finance manager may "
            "release an active severe credit hold."
        ),
        metadata={
            "doc_id": doc_id,
            "title": "Credit Hold Policy",
            "section_id": section_id,
            "section_title": section_title,
            "access_roles": [
                "finance_manager",
            ],
            "status": "active",
        },
    )


def test_agent_answers_after_one_sufficient_retrieval():
    retriever = StubRetriever(
        [[_result()]]
    )
    generator = StubGenerator(
        "A finance manager may release it [1]."
    )

    response = AgenticRAG(
        retriever=retriever,
        generator=generator,
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=True,
                    reason="Evidence is sufficient.",
                    matched_terms=(
                        "credit",
                        "hold",
                    ),
                )
            ]
        ),
    ).answer(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    assert response.attempts == 1
    assert response.sources[0].section_id == "CH-3"
    assert generator.calls == 1
    assert [
        step.action
        for step in response.trace
    ] == [
        "plan",
        "retrieve",
        "grade_evidence",
        "verify_retrieval",
        "generate_answer",
        "verify_answer",
    ]


def test_agent_rewrites_and_retries_weak_evidence():
    first_result = _result(
        chunk_id="weak",
        doc_id="shipment_pricing_reference",
        section_id="SP-2",
        section_title="Rate Exceptions",
    )
    final_result = _result(
        chunk_id="re2",
        doc_id="rate_exception_policy",
        section_id="RE-2",
        section_title="Above Authority Discount",
    )

    retriever = StubRetriever(
        [
            [first_result],
            [final_result],
        ]
    )
    rewriter = StubRewriter(
        "discount delegated authority approval RE-2"
    )

    response = AgenticRAG(
        retriever=retriever,
        generator=StubGenerator(
            "Finance approval is required [1]."
        ),
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=False,
                    reason="Weak evidence.",
                ),
                EvidenceAssessment(
                    sufficient=True,
                    reason="Strong evidence.",
                    matched_terms=("discount",),
                ),
            ]
        ),
        rewriter=rewriter,
        max_attempts=2,
    ).answer(
        "What discount needs finance approval?",
        role="finance_manager",
        top_k=3,
    )

    assert response.attempts == 2
    assert len(retriever.calls) == 2
    assert retriever.calls[1]["query"] == (
        "discount delegated authority approval RE-2"
    )
    assert rewriter.calls == 1
    assert "rewrite_query" in {
        step.action
        for step in response.trace
    }


def test_exact_section_id_uses_exact_filter_and_sparse_priority():
    retriever = StubRetriever(
        [[
            _result(
                chunk_id="re2",
                doc_id="rate_exception_policy",
                section_id="RE-2",
                section_title=(
                    "Above Authority Discount"
                ),
            )
        ]]
    )

    response = AgenticRAG(
        retriever=retriever,
        generator=StubGenerator(
            "RE-2 requires finance approval [1]."
        ),
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=True,
                    reason="Exact section matched.",
                    matched_terms=("RE-2",),
                )
            ]
        ),
    ).answer(
        "re-2",
        role="finance_manager",
        top_k=2,
    )

    call = retriever.calls[0]

    assert call["query"] == "RE-2"
    assert call["section_ids"] == ("RE-2",)
    assert call["dense_weight"] == 0.5
    assert call["sparse_weight"] == 1.5
    assert response.sources[0].section_id == "RE-2"


def test_agent_stops_safely_after_max_attempts():
    generator = StubGenerator(
        "This answer must not be generated."
    )

    response = AgenticRAG(
        retriever=StubRetriever(
            [[], []]
        ),
        generator=generator,
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=False,
                    reason="No results.",
                ),
                EvidenceAssessment(
                    sufficient=False,
                    reason="Still no results.",
                ),
            ]
        ),
        rewriter=StubRewriter(
            "expanded unavailable query"
        ),
        max_attempts=2,
    ).answer(
        "What is the private password?",
        role="sales_rep",
    )

    assert response.answer == (
        SAFE_NO_EVIDENCE_ANSWER
    )
    assert response.attempts == 2
    assert generator.calls == 0
    assert response.trace[-1].action == "stop"


def test_role_and_metadata_filters_are_forwarded():
    retriever = StubRetriever(
        [[_result()]]
    )

    AgenticRAG(
        retriever=retriever,
        generator=StubGenerator(
            "Answer [1]."
        ),
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=True,
                    reason="Enough.",
                )
            ]
        ),
    ).answer(
        "What is the finance policy?",
        role="finance_manager",
        departments=("finance",),
        document_types=("policy",),
        doc_ids=("credit_hold_policy",),
    )

    call = retriever.calls[0]

    assert call["role"] == "finance_manager"
    assert call["statuses"] == ("active",)
    assert call["departments"] == ("finance",)
    assert call["document_types"] == ("policy",)
    assert call["doc_ids"] == (
        "credit_hold_policy",
    )


def test_agent_accumulates_evidence_across_retrieval_rounds():
    first_result = _result(
        chunk_id="re4",
        doc_id="rate_exception_policy",
        section_id="RE-4",
        section_title="Separate Workflows",
    )
    second_result = _result(
        chunk_id="ch3",
        doc_id="credit_hold_policy",
        section_id="CH-3",
        section_title="Severe Release",
    )
    grader = SequenceGrader(
        [
            EvidenceAssessment(
                sufficient=False,
                reason="One policy facet is still missing.",
            ),
            EvidenceAssessment(
                sufficient=True,
                reason="Combined evidence is sufficient.",
                matched_terms=("hold", "release"),
            ),
        ]
    )

    response = AgenticRAG(
        retriever=StubRetriever(
            [[first_result], [second_result]]
        ),
        generator=StubGenerator(
            "The workflows remain separate [1]. "
            "A finance manager may release the hold [2]."
        ),
        grader=grader,
        rewriter=StubRewriter(
            "severe credit hold release finance manager"
        ),
        max_attempts=2,
    ).answer(
        "Does discount approval release a severe credit hold?",
        role="finance_manager",
        top_k=2,
    )

    assert response.attempts == 2
    assert grader.result_counts == [1, 2]
    assert [source.section_id for source in response.sources] == [
        "RE-4",
        "CH-3",
    ]


def test_rewriter_decomposes_multi_part_discount_and_hold_query():
    rewritten = CorpusQueryRewriter().rewrite(
        (
            "An 18 percent discount is requested for a customer with a "
            "severe credit hold. Who approves the discount, who may "
            "release the hold, and does approval release the hold?"
        ),
        [],
    )

    lowered = rewritten.lower()
    assert "above authority discount" in lowered
    assert "human approval" in lowered
    assert "severe credit hold release" in lowered
    assert "human confirmation" in lowered
    assert "authorization note" in lowered
    assert "separate workflow" in lowered
