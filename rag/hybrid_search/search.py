"""Combine dense Qdrant and lexical BM25 rankings using RRF."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rag.chunking.chunker import MarkdownChunker
from rag.embeddings.embedder import ChunkEmbedder
from rag.hybrid_search.bm25 import BM25Index
from rag.loading.loader import CorpusLoader
from rag.metadata.schema import SearchFilters

SECTION_ID_PATTERN = re.compile(
    r"^[A-Z]{2,5}-\d+(?:\.\d+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """One fused retrieval result."""

    chunk_id: str
    fused_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    text: str
    metadata: dict[str, Any]


class HybridSearch:
    """Run dense and BM25 retrieval, then fuse their ranks."""

    def __init__(
        self,
        *,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        chunks: Sequence[Any] | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self.embedder = embedder or ChunkEmbedder()

        if vector_store is None:
            # Lazy import keeps unit tests independent from qdrant-client.
            from rag.vector_store.qdrant_store import (
                QdrantVectorStore,
            )

            vector_store = QdrantVectorStore()

        self.vector_store = vector_store

        if bm25_index is not None:
            self.bm25_index = bm25_index
        else:
            if chunks is None:
                documents = CorpusLoader().load()
                chunks = (
                    MarkdownChunker()
                    .chunk_documents(documents)
                )

            self.bm25_index = BM25Index(chunks)

    def search(
        self,
        query: str,
        *,
        role: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        statuses: Sequence[str] = ("active",),
        departments: Sequence[str] | None = None,
        document_types: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
    ) -> list[HybridSearchResult]:
        """Return rank-fused dense and lexical results."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("The query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        if rrf_k < 1:
            raise ValueError("rrf_k must be positive.")

        if dense_weight <= 0 or sparse_weight <= 0:
            raise ValueError(
                "dense_weight and sparse_weight "
                "must both be positive."
            )

        resolved_candidate_k = (
            candidate_k
            if candidate_k is not None
            else max(top_k * 4, 20)
        )

        if resolved_candidate_k < top_k:
            raise ValueError(
                "candidate_k must be at least top_k."
            )

        resolved_section_ids = (
            tuple(section_ids)
            if section_ids is not None
            else None
        )

        if (
            resolved_section_ids is None
            and SECTION_ID_PATTERN.fullmatch(
                normalized_query
            )
        ):
            resolved_section_ids = (
                normalized_query.upper(),
            )

        filters = SearchFilters(
            role=role,
            statuses=tuple(statuses),
            departments=(
                tuple(departments)
                if departments is not None
                else None
            ),
            document_types=(
                tuple(document_types)
                if document_types is not None
                else None
            ),
            doc_ids=(
                tuple(doc_ids)
                if doc_ids is not None
                else None
            ),
            section_ids=resolved_section_ids,
        )

        dense_results = self.vector_store.search(
            self.embedder.embed_query(
                normalized_query
            ),
            filters,
            top_k=resolved_candidate_k,
        )

        sparse_results = self.bm25_index.search(
            normalized_query,
            filters,
            top_k=resolved_candidate_k,
        )

        return self._fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    @staticmethod
    def _fuse(
        *,
        dense_results: Sequence[Any],
        sparse_results: Sequence[Any],
        top_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[HybridSearchResult]:
        records: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            record = records.setdefault(
                result.chunk_id,
                HybridSearch._new_record(result),
            )
            record["dense_rank"] = rank
            record["dense_score"] = float(
                result.score
            )
            record["fused_score"] += (
                dense_weight
                / (rrf_k + rank)
            )

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            record = records.setdefault(
                result.chunk_id,
                HybridSearch._new_record(result),
            )
            record["sparse_rank"] = rank
            record["sparse_score"] = float(
                result.score
            )
            record["fused_score"] += (
                sparse_weight
                / (rrf_k + rank)
            )

        results = [
            HybridSearchResult(
                chunk_id=chunk_id,
                fused_score=record["fused_score"],
                dense_rank=record["dense_rank"],
                sparse_rank=record["sparse_rank"],
                dense_score=record["dense_score"],
                sparse_score=record["sparse_score"],
                text=record["text"],
                metadata=record["metadata"],
            )
            for chunk_id, record in records.items()
        ]

        results.sort(
            key=lambda item: (
                -item.fused_score,
                min(
                    rank
                    for rank in (
                        item.dense_rank,
                        item.sparse_rank,
                    )
                    if rank is not None
                ),
                item.chunk_id,
            )
        )

        return results[:top_k]

    @staticmethod
    def _new_record(
        result: Any,
    ) -> dict[str, Any]:
        metadata = result.metadata

        if hasattr(metadata, "__dataclass_fields__"):
            metadata_dict = asdict(metadata)
        elif hasattr(metadata, "model_dump"):
            metadata_dict = metadata.model_dump()
        else:
            metadata_dict = dict(metadata)

        metadata_dict = {
            key: list(value)
            if isinstance(value, tuple)
            else value
            for key, value in metadata_dict.items()
        }

        return {
            "fused_score": 0.0,
            "dense_rank": None,
            "sparse_rank": None,
            "dense_score": None,
            "sparse_score": None,
            "text": str(result.text),
            "metadata": metadata_dict,
        }
