"""Load and validate the gold retrieval dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rag.evaluation.models import EvaluationCase

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "dataset.json"
)


def load_evaluation_cases(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> tuple[EvaluationCase, ...]:
    """Read evaluation cases from JSON."""

    dataset_path = Path(path)

    try:
        payload = json.loads(
            dataset_path.read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Evaluation dataset is invalid JSON: {exc}"
        ) from exc

    if not isinstance(payload, list):
        raise TypeError(
            "Evaluation dataset root must be a list."
        )

    cases = tuple(
        EvaluationCase.from_dict(item)
        for item in payload
    )

    _validate_unique_ids(cases)
    return cases


def _validate_unique_ids(
    cases: Sequence[EvaluationCase],
) -> None:
    ids = [case.case_id for case in cases]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Evaluation case IDs must be unique."
        )
