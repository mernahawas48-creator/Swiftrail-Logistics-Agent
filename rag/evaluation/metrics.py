"""Pure retrieval metrics used by the evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def first_matching_rank(
    retrieved: Sequence[str],
    relevant: Iterable[str],
) -> int | None:
    """Return the one-based rank of the first matching section."""

    relevant_set = set(relevant)

    for rank, section_id in enumerate(
        retrieved,
        start=1,
    ):
        if section_id in relevant_set:
            return rank

    return None


def hit_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Return 1 when at least one relevant result appears in top K."""

    _validate_k(k)
    relevant_set = set(relevant)

    return float(
        any(
            section_id in relevant_set
            for section_id in retrieved[:k]
        )
    )


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Return the fraction of gold sections retrieved in top K."""

    _validate_k(k)
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    found = relevant_set.intersection(
        retrieved[:k]
    )

    return len(found) / len(relevant_set)


def reciprocal_rank_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Return reciprocal rank when the first hit is inside top K."""

    _validate_k(k)
    rank = first_matching_rank(
        retrieved[:k],
        relevant,
    )

    if rank is None:
        return 0.0

    return 1.0 / rank


def access_safe_at_k(
    retrieved: Sequence[str],
    forbidden: Iterable[str],
    k: int,
) -> float:
    """Return 1 when no forbidden section appears in top K."""

    _validate_k(k)
    forbidden_set = set(forbidden)

    return float(
        not any(
            section_id in forbidden_set
            for section_id in retrieved[:k]
        )
    )


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for an empty list."""

    if not values:
        return 0.0

    return sum(values) / len(values)


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive.")
