"""Small in-memory BM25 index for Swiftrail document chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rag.metadata.schema import SearchFilters

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SparseSearchResult:
    """One lexical BM25 result."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class BM25Index:
    """Index chunks and score authorized lexical matches."""

    def __init__(
        self,
        chunks: Sequence[Any],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive.")

        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1.")

        if not chunks:
            raise ValueError(
                "BM25Index requires at least one chunk."
            )

        self.k1 = float(k1)
        self.b = float(b)
        self._records: list[dict[str, Any]] = []
        self._document_frequency: Counter[str] = Counter()

        for chunk in chunks:
            metadata = self._metadata_to_dict(
                chunk.metadata
            )
            searchable_text = self._searchable_text(
                text=chunk.text,
                metadata=metadata,
            )
            tokens = self.tokenize(searchable_text)

            term_frequency = Counter(tokens)

            self._records.append(
                {
                    "chunk_id": str(chunk.chunk_id),
                    "text": str(chunk.text),
                    "metadata": metadata,
                    "tokens": tokens,
                    "term_frequency": term_frequency,
                    "length": len(tokens),
                }
            )

            self._document_frequency.update(
                term_frequency.keys()
            )

        self._document_count = len(self._records)
        self._average_length = (
            sum(record["length"] for record in self._records)
            / self._document_count
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase text while preserving IDs such as CH-3 and RE-2."""

        return [
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
        ]

    def search(
        self,
        query: str,
        filters: SearchFilters,
        *,
        top_k: int = 20,
    ) -> list[SparseSearchResult]:
        """Return authorized chunks ranked by BM25 score."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("The query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        query_terms = self.tokenize(normalized_query)
        if not query_terms:
            return []

        scored: list[SparseSearchResult] = []

        for record in self._records:
            metadata = record["metadata"]

            if not self._is_authorized(
                metadata,
                filters,
            ):
                continue

            score = self._score_record(
                record,
                query_terms,
            )

            if score <= 0.0:
                continue

            scored.append(
                SparseSearchResult(
                    chunk_id=record["chunk_id"],
                    score=score,
                    text=record["text"],
                    metadata=metadata,
                )
            )

        scored.sort(
            key=lambda item: (
                -item.score,
                item.chunk_id,
            )
        )
        return scored[:top_k]

    def _score_record(
        self,
        record: dict[str, Any],
        query_terms: Sequence[str],
    ) -> float:
        score = 0.0
        term_frequency: Counter[str] = record[
            "term_frequency"
        ]
        document_length = record["length"]

        for term in query_terms:
            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            document_frequency = (
                self._document_frequency[term]
            )

            inverse_document_frequency = math.log(
                1.0
                + (
                    self._document_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            denominator = (
                frequency
                + self.k1
                * (
                    1.0
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self._average_length
                    )
                )
            )

            score += (
                inverse_document_frequency
                * frequency
                * (self.k1 + 1.0)
                / denominator
            )

        return score

    @staticmethod
    def _searchable_text(
        *,
        text: str,
        metadata: dict[str, Any],
    ) -> str:
        """Add structured fields so exact IDs and titles remain searchable."""

        keywords = metadata.get("keywords", [])
        keyword_text = " ".join(
            str(keyword)
            for keyword in keywords
        )

        # Repeating the exact section ID gives identifiers such as RE-2
        # stronger lexical evidence without changing the stored chunk text.
        return "\n".join(
            [
                str(text),
                str(metadata.get("title", "")),
                str(metadata.get("section_title", "")),
                str(metadata.get("section_id", "")),
                str(metadata.get("section_id", "")),
                str(metadata.get("section_id", "")),
                keyword_text,
            ]
        )

    @staticmethod
    def _metadata_to_dict(
        metadata: Any,
    ) -> dict[str, Any]:
        if hasattr(metadata, "__dataclass_fields__"):
            payload = asdict(metadata)
        elif hasattr(metadata, "model_dump"):
            payload = metadata.model_dump()
        elif isinstance(metadata, dict):
            payload = dict(metadata)
        else:
            raise TypeError(
                "Chunk metadata must be a dataclass, "
                "Pydantic model, or dictionary."
            )

        return {
            key: list(value)
            if isinstance(value, tuple)
            else value
            for key, value in payload.items()
        }

    @staticmethod
    def _is_authorized(
        metadata: dict[str, Any],
        filters: SearchFilters,
    ) -> bool:
        if filters.role not in metadata.get(
            "access_roles",
            [],
        ):
            return False

        if metadata.get("status") not in filters.statuses:
            return False

        optional_filters: tuple[
            tuple[str, Sequence[str] | None],
            ...
        ] = (
            ("department", filters.departments),
            (
                "document_type",
                filters.document_types,
            ),
            ("doc_id", filters.doc_ids),
            ("section_id", filters.section_ids),
        )

        for field_name, allowed_values in optional_filters:
            if (
                allowed_values is not None
                and metadata.get(field_name)
                not in allowed_values
            ):
                return False

        return True
