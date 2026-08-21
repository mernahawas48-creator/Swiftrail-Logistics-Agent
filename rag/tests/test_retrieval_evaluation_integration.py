from __future__ import annotations

import os

import pytest

from rag.evaluation.dataset import (
    load_evaluation_cases,
)
from rag.evaluation.evaluator import (
    RetrievalEvaluator,
)
from rag.evaluation.retrievers import (
    DenseRetriever,
    HybridRetriever,
)

pytestmark = pytest.mark.skipif(
    os.getenv(
        "RUN_RETRIEVAL_EVALUATION_INTEGRATION"
    ) != "1",
    reason=(
        "Set RUN_RETRIEVAL_EVALUATION_INTEGRATION=1 "
        "after populating Qdrant."
    ),
)


def test_real_dense_and_hybrid_evaluation_runs():
    cases = load_evaluation_cases()
    evaluator = RetrievalEvaluator(
        ks=(1, 3, 5),
    )

    dense_report = evaluator.evaluate(
        DenseRetriever(),
        cases,
    )
    hybrid_report = evaluator.evaluate(
        HybridRetriever(),
        cases,
    )

    assert dense_report.case_count == 28
    assert hybrid_report.case_count == 28
    assert dense_report.relevance_case_count == 26
    assert hybrid_report.access_control_case_count == 2

    assert (
        dense_report.metrics[
            "access_safety_rate"
        ][5]
        == 1.0
    )
    assert (
        hybrid_report.metrics[
            "access_safety_rate"
        ][5]
        == 1.0
    )

    assert (
        dense_report.metrics["hit_rate"][5]
        > 0.0
    )
    assert (
        hybrid_report.metrics["hit_rate"][5]
        > 0.0
    )


def test_hybrid_exact_id_cases_hit_at_one():
    cases = tuple(
        case
        for case in load_evaluation_cases()
        if case.category == "exact_id"
    )

    report = RetrievalEvaluator(
        ks=(1,),
    ).evaluate(
        HybridRetriever(),
        cases,
    )

    assert report.case_count == 4
    assert report.metrics[
        "hit_rate"
    ][1] == 1.0
    assert report.metrics[
        "mrr"
    ][1] == 1.0
