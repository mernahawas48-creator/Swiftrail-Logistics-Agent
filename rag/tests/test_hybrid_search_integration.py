from __future__ import annotations

import os

import pytest

from rag.hybrid_search.search import HybridSearch

pytestmark = pytest.mark.skipif(
    os.getenv(
        "RUN_HYBRID_SEARCH_INTEGRATION"
    ) != "1",
    reason=(
        "Set RUN_HYBRID_SEARCH_INTEGRATION=1 "
        "after populating Qdrant."
    ),
)


def test_real_hybrid_search_combines_dense_and_bm25():
    searcher = HybridSearch()

    semantic_results = searcher.search(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    assert semantic_results
    assert semantic_results[0].metadata[
        "doc_id"
    ] == "credit_hold_policy"
    assert semantic_results[0].metadata[
        "section_id"
    ] == "CH-3"
    assert (
        semantic_results[0].dense_rank
        is not None
    )
    assert (
        semantic_results[0].sparse_rank
        is not None
    )

    identifier_results = searcher.search(
        "RE-2",
        role="finance_manager",
        top_k=3,
        dense_weight=0.5,
        sparse_weight=1.5,
    )

    assert identifier_results
    assert identifier_results[0].metadata[
        "section_id"
    ] == "RE-2"
