"""Models used by the retrieval evaluation stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

EvaluationType = Literal[
    "relevance",
    "access_control",
]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One gold retrieval test case."""

    case_id: str
    query: str
    role: str
    evaluation_type: EvaluationType
    expected_section_ids: tuple[str, ...]
    forbidden_section_ids: tuple[str, ...]
    category: str

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EvaluationCase:
        required = {
            "case_id",
            "query",
            "role",
            "evaluation_type",
            "expected_section_ids",
            "forbidden_section_ids",
            "category",
        }

        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "Evaluation case is missing fields: "
                + ", ".join(sorted(missing))
            )

        evaluation_type = str(
            payload["evaluation_type"]
        )

        if evaluation_type not in {
            "relevance",
            "access_control",
        }:
            raise ValueError(
                "evaluation_type must be relevance "
                "or access_control."
            )

        role = str(payload["role"])
        if role not in {
            "sales_rep",
            "finance_manager",
        }:
            raise ValueError(
                f"Unsupported evaluation role: {role}"
            )

        expected = tuple(
            str(value)
            for value in payload[
                "expected_section_ids"
            ]
        )
        forbidden = tuple(
            str(value)
            for value in payload[
                "forbidden_section_ids"
            ]
        )

        if (
            evaluation_type == "relevance"
            and not expected
        ):
            raise ValueError(
                "Relevance cases require at least one "
                "expected section ID."
            )

        if (
            evaluation_type == "access_control"
            and not forbidden
        ):
            raise ValueError(
                "Access-control cases require at least one "
                "forbidden section ID."
            )

        return cls(
            case_id=str(payload["case_id"]),
            query=str(payload["query"]).strip(),
            role=role,
            evaluation_type=evaluation_type,
            expected_section_ids=expected,
            forbidden_section_ids=forbidden,
            category=str(payload["category"]),
        )


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Metrics and ranking details for one case."""

    case_id: str
    query: str
    role: str
    evaluation_type: str
    category: str
    expected_section_ids: tuple[str, ...]
    forbidden_section_ids: tuple[str, ...]
    retrieved_section_ids: tuple[str, ...]
    first_relevant_rank: int | None
    first_forbidden_rank: int | None
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank_at_k: dict[int, float]
    access_safe_at_k: dict[int, float]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Aggregate report for one retrieval method."""

    retriever_name: str
    ks: tuple[int, ...]
    case_count: int
    relevance_case_count: int
    access_control_case_count: int
    metrics: dict[str, dict[int, float]]
    cases: tuple[CaseEvaluation, ...]
