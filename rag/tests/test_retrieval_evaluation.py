from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rag.evaluation.dataset import (
    load_evaluation_cases,
)
from rag.evaluation.evaluator import (
    RetrievalEvaluator,
    write_reports,
)
from rag.evaluation.metrics import (
    access_safe_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from rag.evaluation.models import (
    EvaluationCase,
)


class StubRetriever:
    name = "stub"

    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def search(
        self,
        query,
        *,
        role,
        top_k,
        **kwargs,
    ):
        self.calls.append(
            {
                "query": query,
                "role": role,
                "top_k": top_k,
            }
        )

        return [
            SimpleNamespace(
                metadata={
                    "section_id": section_id
                }
            )
            for section_id in self.rankings[
                query
            ][:top_k]
        ]


def _case(
    *,
    case_id,
    query,
    evaluation_type="relevance",
    expected=("CH-3",),
    forbidden=(),
):
    return EvaluationCase(
        case_id=case_id,
        query=query,
        role="finance_manager",
        evaluation_type=evaluation_type,
        expected_section_ids=expected,
        forbidden_section_ids=forbidden,
        category="test",
    )


def test_metric_functions_use_one_based_ranks():
    retrieved = [
        "CH-1",
        "CH-3",
        "CH-2",
    ]

    assert hit_at_k(
        retrieved,
        {"CH-3"},
        1,
    ) == 0.0
    assert hit_at_k(
        retrieved,
        {"CH-3"},
        2,
    ) == 1.0
    assert recall_at_k(
        retrieved,
        {"CH-3", "CH-2"},
        2,
    ) == 0.5
    assert reciprocal_rank_at_k(
        retrieved,
        {"CH-3"},
        3,
    ) == 0.5
    assert access_safe_at_k(
        retrieved,
        {"PR-1"},
        3,
    ) == 1.0


def test_evaluator_aggregates_relevance_and_access_metrics():
    cases = (
        _case(
            case_id="relevant-1",
            query="q1",
        ),
        _case(
            case_id="relevant-2",
            query="q2",
            expected=("RE-2",),
        ),
        _case(
            case_id="access",
            query="q3",
            evaluation_type="access_control",
            expected=(),
            forbidden=("PR-1",),
        ),
    )

    retriever = StubRetriever(
        {
            "q1": ["CH-3", "CH-1"],
            "q2": ["SP-2", "RE-2"],
            "q3": ["AC-1", "PR-1"],
        }
    )

    report = RetrievalEvaluator(
        ks=(1, 2),
    ).evaluate(
        retriever,
        cases,
    )

    assert report.case_count == 3
    assert report.relevance_case_count == 2
    assert report.access_control_case_count == 1

    assert report.metrics[
        "hit_rate"
    ][1] == 0.5
    assert report.metrics[
        "hit_rate"
    ][2] == 1.0
    assert report.metrics[
        "mrr"
    ][2] == 0.75
    assert report.metrics[
        "access_safety_rate"
    ][1] == 1.0
    assert report.metrics[
        "access_safety_rate"
    ][2] == 0.0
    assert report.metrics[
        "unauthorized_leakage_rate"
    ][2] == 1.0

    assert all(
        call["top_k"] == 2
        for call in retriever.calls
    )


def test_dataset_loader_validates_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "one",
                    "query": "CH-3",
                    "role": "finance_manager",
                    "evaluation_type": "relevance",
                    "expected_section_ids": ["CH-3"],
                    "forbidden_section_ids": [],
                    "category": "exact_id",
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].expected_section_ids == (
        "CH-3",
    )


def test_duplicate_dataset_ids_are_rejected(tmp_path):
    payload = {
        "case_id": "duplicate",
        "query": "CH-3",
        "role": "finance_manager",
        "evaluation_type": "relevance",
        "expected_section_ids": ["CH-3"],
        "forbidden_section_ids": [],
        "category": "exact_id",
    }

    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([payload, payload]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        load_evaluation_cases(path)


def test_report_writer_creates_json_and_markdown(tmp_path):
    report = RetrievalEvaluator(
        ks=(1, 3),
    ).evaluate(
        StubRetriever(
            {
                "q": [
                    "CH-3",
                    "CH-1",
                    "CH-2",
                ]
            }
        ),
        (
            _case(
                case_id="case",
                query="q",
            ),
        ),
    )

    json_path, markdown_path = (
        write_reports(
            [report],
            tmp_path,
        )
    )

    assert json_path.exists()
    assert markdown_path.exists()
    assert "hit_rate" in json_path.read_text(
        encoding="utf-8"
    )
    assert "Swiftrail Retrieval Evaluation" in (
        markdown_path.read_text(
            encoding="utf-8"
        )
    )
