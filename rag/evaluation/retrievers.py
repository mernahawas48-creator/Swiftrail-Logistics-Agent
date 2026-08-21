"""Dense and hybrid retrieval adapters used by evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

SECTION_ID_PATTERN = re.compile(
    r"^[A-Z]{2,5}-\d+(?:\.\d+)?$",
    re.IGNORECASE,
)


class DenseRetriever:
    """Evaluate the existing embedding plus Qdrant dense search."""

    name = "dense"

    def __init__(
        self,
        *,
        embedder: Any | None = None,
        vector_store: Any | None = None,
    ):
        if embedder is None:
            from rag.embeddings.embedder import (
                ChunkEmbedder,
            )

            embedder = ChunkEmbedder()

        if vector_store is None:
            from rag.vector_store.qdrant_store import (
                QdrantVectorStore,
            )

            vector_store = QdrantVectorStore()

        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        *,
        role: str,
        top_k: int,
        statuses: Sequence[str] = ("active",),
        departments: Sequence[str] | None = None,
        document_types: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
    ) -> list[Any]:
        """Run dense search with the same metadata filters."""

        from rag.metadata.schema import SearchFilters

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
            section_ids=(
                tuple(section_ids)
                if section_ids is not None
                else None
            ),
        )

        return self.vector_store.search(
            self.embedder.embed_query(query),
            filters,
            top_k=top_k,
        )


class HybridRetriever:
    """Evaluate the existing dense plus BM25 Hybrid Search."""

    name = "hybrid"

    def __init__(
        self,
        *,
        searcher: Any | None = None,
    ):
        if searcher is None:
            from rag.hybrid_search.search import (
                HybridSearch,
            )

            searcher = HybridSearch()

        self.searcher = searcher

    def search(
        self,
        query: str,
        *,
        role: str,
        top_k: int,
        statuses: Sequence[str] = ("active",),
        departments: Sequence[str] | None = None,
        document_types: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
    ) -> list[Any]:
        """Run Hybrid Search using its exact-ID routing behavior."""

        return self.searcher.search(
            query,
            role=role,
            top_k=top_k,
            candidate_k=max(top_k * 4, 20),
            statuses=tuple(statuses),
            departments=departments,
            document_types=document_types,
            doc_ids=doc_ids,
            section_ids=section_ids,
        )
