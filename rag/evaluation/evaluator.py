"""Evaluate one or more retrieval methods against gold cases."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from rag.evaluation.metrics import (
    access_safe_at_k,
    first_matching_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank_at_k,
)
from rag.evaluation.models import (
    CaseEvaluation,
    EvaluationCase,
    RetrievalEvaluationReport,
)


class Searcher(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        role: str,
        top_k: int,
        **kwargs: Any,
    ) -> list[Any]:
        ...


class RetrievalEvaluator:
    """Run gold queries and calculate retrieval and access metrics."""

    def __init__(
        self,
        *,
        ks: Sequence[int] = (1, 3, 5),
    ):
        normalized_ks = tuple(
            sorted({int(k) for k in ks})
        )

        if not normalized_ks:
            raise ValueError(
                "At least one K value is required."
            )

        if normalized_ks[0] < 1:
            raise ValueError(
                "All K values must be positive."
            )

        self.ks = normalized_ks

    def evaluate(
        self,
        retriever: Searcher,
        cases: Sequence[EvaluationCase],
    ) -> RetrievalEvaluationReport:
        """Evaluate a retriever over all supplied cases."""

        case_results: list[CaseEvaluation] = []
        max_k = max(self.ks)

        for case in cases:
            results = retriever.search(
                case.query,
                role=case.role,
                top_k=max_k,
            )

            retrieved_section_ids = tuple(
                str(
                    result.metadata.get(
                        "section_id",
                        "",
                    )
                )
                for result in results
            )

            case_results.append(
                self._evaluate_case(
                    case,
                    retrieved_section_ids,
                )
            )

        return self._build_report(
            retriever_name=retriever.name,
            case_results=case_results,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        retrieved_section_ids: tuple[str, ...],
    ) -> CaseEvaluation:
        expected = case.expected_section_ids
        forbidden = case.forbidden_section_ids

        return CaseEvaluation(
            case_id=case.case_id,
            query=case.query,
            role=case.role,
            evaluation_type=case.evaluation_type,
            category=case.category,
            expected_section_ids=expected,
            forbidden_section_ids=forbidden,
            retrieved_section_ids=(
                retrieved_section_ids
            ),
            first_relevant_rank=first_matching_rank(
                retrieved_section_ids,
                expected,
            ),
            first_forbidden_rank=first_matching_rank(
                retrieved_section_ids,
                forbidden,
            ),
            hit_at_k={
                k: hit_at_k(
                    retrieved_section_ids,
                    expected,
                    k,
                )
                for k in self.ks
            },
            recall_at_k={
                k: recall_at_k(
                    retrieved_section_ids,
                    expected,
                    k,
                )
                for k in self.ks
            },
            reciprocal_rank_at_k={
                k: reciprocal_rank_at_k(
                    retrieved_section_ids,
                    expected,
                    k,
                )
                for k in self.ks
            },
            access_safe_at_k={
                k: access_safe_at_k(
                    retrieved_section_ids,
                    forbidden,
                    k,
                )
                for k in self.ks
            },
        )

    def _build_report(
        self,
        *,
        retriever_name: str,
        case_results: Sequence[CaseEvaluation],
    ) -> RetrievalEvaluationReport:
        relevance_cases = [
            case
            for case in case_results
            if case.evaluation_type == "relevance"
        ]
        access_cases = [
            case
            for case in case_results
            if case.evaluation_type
            == "access_control"
        ]

        metrics = {
            "hit_rate": {
                k: mean(
                    [
                        case.hit_at_k[k]
                        for case in relevance_cases
                    ]
                )
                for k in self.ks
            },
            "mean_recall": {
                k: mean(
                    [
                        case.recall_at_k[k]
                        for case in relevance_cases
                    ]
                )
                for k in self.ks
            },
            "mrr": {
                k: mean(
                    [
                        case.reciprocal_rank_at_k[k]
                        for case in relevance_cases
                    ]
                )
                for k in self.ks
            },
            "access_safety_rate": {
                k: mean(
                    [
                        case.access_safe_at_k[k]
                        for case in access_cases
                    ]
                )
                if access_cases
                else 1.0
                for k in self.ks
            },
            "unauthorized_leakage_rate": {
                k: (
                    1.0
                    - mean(
                        [
                            case.access_safe_at_k[k]
                            for case in access_cases
                        ]
                    )
                )
                if access_cases
                else 0.0
                for k in self.ks
            },
        }

        return RetrievalEvaluationReport(
            retriever_name=retriever_name,
            ks=self.ks,
            case_count=len(case_results),
            relevance_case_count=len(
                relevance_cases
            ),
            access_control_case_count=len(
                access_cases
            ),
            metrics=metrics,
            cases=tuple(case_results),
        )


def write_reports(
    reports: Sequence[RetrievalEvaluationReport],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write machine-readable JSON and a Markdown comparison."""

    directory = Path(output_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        directory
        / "retrieval_evaluation.json"
    )
    markdown_path = (
        directory
        / "retrieval_evaluation.md"
    )

    json_payload = [
        asdict(report)
        for report in reports
    ]

    json_path.write_text(
        json.dumps(
            json_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        _build_markdown(reports),
        encoding="utf-8",
    )

    return json_path, markdown_path


def _build_markdown(
    reports: Sequence[RetrievalEvaluationReport],
) -> str:
    if not reports:
        return "# Retrieval Evaluation\n\nNo reports.\n"

    ks = reports[0].ks
    lines = [
        "# Swiftrail Retrieval Evaluation",
        "",
        "## Summary",
        "",
        "| Retriever | Metric | "
        + " | ".join(f"@{k}" for k in ks)
        + " |",
        "|---|---|"
        + "|".join("---:" for _ in ks)
        + "|",
    ]

    metric_order = (
        "hit_rate",
        "mean_recall",
        "mrr",
        "access_safety_rate",
        "unauthorized_leakage_rate",
    )

    for report in reports:
        for metric_name in metric_order:
            values = report.metrics[
                metric_name
            ]
            lines.append(
                "| "
                f"{report.retriever_name}"
                " | "
                f"{metric_name}"
                " | "
                + " | ".join(
                    f"{values[k]:.4f}"
                    for k in ks
                )
                + " |"
            )

    for report in reports:
        lines.extend(
            [
                "",
                f"## {report.retriever_name.title()} failures",
                "",
            ]
        )

        failures = [
            case
            for case in report.cases
            if (
                case.evaluation_type
                == "relevance"
                and case.hit_at_k[max(ks)] == 0.0
            )
            or (
                case.evaluation_type
                == "access_control"
                and case.access_safe_at_k[
                    max(ks)
                ]
                == 0.0
            )
        ]

        if not failures:
            lines.append(
                "No failures at the largest K."
            )
            continue

        for case in failures:
            lines.extend(
                [
                    (
                        f"- **{case.case_id}** "
                        f"({case.evaluation_type})"
                    ),
                    (
                        "  - Expected: "
                        f"{list(case.expected_section_ids)}"
                    ),
                    (
                        "  - Forbidden: "
                        f"{list(case.forbidden_section_ids)}"
                    ),
                    (
                        "  - Retrieved: "
                        f"{list(case.retrieved_section_ids)}"
                    ),
                ]
            )

    return "\n".join(lines) + "\n"
