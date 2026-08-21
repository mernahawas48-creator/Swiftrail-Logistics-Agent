from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.chunking.chunker import (
    ChunkMetadata,
    DocumentChunk,
)
from rag.hybrid_search.bm25 import BM25Index
from rag.hybrid_search.search import HybridSearch
from rag.metadata.schema import SearchFilters


class StubEmbedder:
    def __init__(self):
        self.last_query = None

    def embed_query(self, query: str):
        self.last_query = query
        return (1.0, 0.0, 0.0)


class StubVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_filters = None
        self.last_top_k = None

    def search(
        self,
        query_vector,
        filters,
        *,
        top_k,
    ):
        self.last_filters = filters
        self.last_top_k = top_k
        return self.results


def _metadata(
    *,
    doc_id: str,
    section_id: str,
    section_title: str,
    access_roles: tuple[str, ...],
) -> ChunkMetadata:
    return ChunkMetadata(
        doc_id=doc_id,
        title=f"{doc_id.replace('_', ' ').title()} Policy",
        version="1.0",
        effective_date="2026-08-01",
        status="active",
        department="finance",
        document_type="policy",
        access_roles=access_roles,
        source_path=f"documents/{doc_id}.md",
        section_id=section_id,
        section_title=section_title,
        chunk_index=0,
        source_checksum="a" * 64,
        keywords=(),
    )


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-ch3",
            text=(
                "Only an authenticated finance manager may "
                "release an active severe credit hold."
            ),
            metadata=_metadata(
                doc_id="credit_hold_policy",
                section_id="CH-3",
                section_title="Severe Release",
                access_roles=("finance_manager",),
            ),
        ),
        DocumentChunk(
            chunk_id="chunk-re2",
            text=(
                "A discount above delegated authority requires "
                "finance review and explicit approval."
            ),
            metadata=_metadata(
                doc_id="rate_exception_policy",
                section_id="RE-2",
                section_title="Above Authority Discount",
                access_roles=(
                    "sales_rep",
                    "finance_manager",
                ),
            ),
        ),
        DocumentChunk(
            chunk_id="chunk-pr2",
            text=(
                "Portfolio risk review priority is restricted "
                "to finance managers."
            ),
            metadata=_metadata(
                doc_id="portfolio_risk_guidelines",
                section_id="PR-2",
                section_title="Review Priority",
                access_roles=("finance_manager",),
            ),
        ),
    ]


def _dense_result(
    chunk: DocumentChunk,
    score: float,
):
    return SimpleNamespace(
        chunk_id=chunk.chunk_id,
        score=score,
        text=chunk.text,
        metadata=chunk.metadata,
    )


def test_bm25_preserves_and_ranks_exact_section_ids():
    chunks = _chunks()
    index = BM25Index(chunks)

    results = index.search(
        "RE-2",
        SearchFilters(
            role="finance_manager"
        ),
        top_k=3,
    )

    assert results
    assert results[0].chunk_id == "chunk-re2"
    assert results[0].metadata[
        "section_id"
    ] == "RE-2"


def test_bm25_applies_role_filtering():
    chunks = _chunks()
    index = BM25Index(chunks)

    results = index.search(
        "portfolio risk review priority",
        SearchFilters(role="sales_rep"),
        top_k=3,
    )

    assert all(
        result.chunk_id != "chunk-pr2"
        for result in results
    )


def test_rrf_promotes_a_result_found_by_both_retrievers():
    chunks = _chunks()

    dense_results = [
        _dense_result(chunks[2], 0.95),
        _dense_result(chunks[0], 0.85),
    ]

    store = StubVectorStore(dense_results)
    searcher = HybridSearch(
        embedder=StubEmbedder(),
        vector_store=store,
        chunks=chunks,
    )

    results = searcher.search(
        "severe credit hold release",
        role="finance_manager",
        top_k=3,
        candidate_k=3,
    )

    assert results[0].chunk_id == "chunk-ch3"
    assert results[0].dense_rank == 2
    assert results[0].sparse_rank == 1
    assert store.last_filters.role == (
        "finance_manager"
    )


def test_hybrid_search_passes_metadata_filters():
    chunks = _chunks()
    store = StubVectorStore([])

    HybridSearch(
        embedder=StubEmbedder(),
        vector_store=store,
        chunks=chunks,
    ).search(
        "discount",
        role="sales_rep",
        top_k=2,
        candidate_k=3,
        section_ids=("RE-2",),
    )

    assert store.last_filters.role == "sales_rep"
    assert store.last_filters.statuses == (
        "active",
    )
    assert store.last_filters.section_ids == (
        "RE-2",
    )


def test_invalid_arguments_are_rejected():
    searcher = HybridSearch(
        embedder=StubEmbedder(),
        vector_store=StubVectorStore([]),
        chunks=_chunks(),
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        searcher.search(
            "   ",
            role="sales_rep",
        )

    with pytest.raises(
        ValueError,
        match="candidate_k",
    ):
        searcher.search(
            "discount",
            role="sales_rep",
            top_k=5,
            candidate_k=3,
        )

    with pytest.raises(
        ValueError,
        match="must both be positive",
    ):
        searcher.search(
            "discount",
            role="sales_rep",
            dense_weight=0,
        )

def test_exact_section_id_becomes_metadata_filter():
    chunks = _chunks()
    store = StubVectorStore([])

    HybridSearch(
        embedder=StubEmbedder(),
        vector_store=store,
        chunks=chunks,
    ).search(
        "re-2",
        role="finance_manager",
        top_k=1,
        candidate_k=3,
    )

    assert store.last_filters.section_ids == (
        "RE-2",
    )

