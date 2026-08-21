from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import pytest

from rag.metadata.schema import SearchFilters
from rag.vector_store.config import VectorStoreSettings
from rag.vector_store.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason=(
        "Set RUN_QDRANT_INTEGRATION=1 after starting "
        "the Qdrant Docker container."
    ),
)


@dataclass(frozen=True, slots=True)
class SampleMetadata:
    doc_id: str
    title: str
    version: str
    effective_date: str
    status: str
    department: str
    document_type: str
    access_roles: tuple[str, ...]
    source_path: str
    section_id: str
    section_title: str
    chunk_index: int
    source_checksum: str
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SampleEmbeddedChunk:
    chunk_id: str
    vector: tuple[float, ...]
    text: str
    metadata: SampleMetadata


def _vector(first: float, second: float):
    return (
        first,
        second,
        *((0.0,) * 382),
    )


def test_real_qdrant_ann_and_role_filtering():
    collection_name = (
        "swiftrail_vector_test_"
        + uuid.uuid4().hex[:8]
    )

    settings = VectorStoreSettings(
        url=os.getenv(
            "QDRANT_URL",
            "http://127.0.0.1:6333",
        ),
        collection_name=collection_name,
        vector_size=384,
        default_top_k=5,
    )
    store = QdrantVectorStore(settings=settings)

    sales_chunk = SampleEmbeddedChunk(
        chunk_id=str(uuid.uuid4()),
        vector=_vector(1.0, 0.0),
        text="Sales representatives may approve up to 15 percent.",
        metadata=SampleMetadata(
            doc_id="rate_exception_policy",
            title="Rate Exception Policy",
            version="1.0",
            effective_date="2026-08-01",
            status="active",
            department="finance",
            document_type="policy",
            access_roles=(
                "sales_rep",
                "finance_manager",
            ),
            source_path=(
                "documents/rate_exception_policy.md"
            ),
            section_id="RE-1",
            section_title="Delegated Limit",
            chunk_index=0,
            source_checksum="a" * 64,
            keywords=("discount",),
        ),
    )

    finance_only_chunk = SampleEmbeddedChunk(
        chunk_id=str(uuid.uuid4()),
        vector=_vector(0.0, 1.0),
        text="Finance reviews portfolio risk priority.",
        metadata=SampleMetadata(
            doc_id="portfolio_risk_guidelines",
            title="Portfolio Risk Guidelines",
            version="1.0",
            effective_date="2026-08-01",
            status="active",
            department="finance",
            document_type="guideline",
            access_roles=("finance_manager",),
            source_path=(
                "documents/portfolio_risk_guidelines.md"
            ),
            section_id="PR-2",
            section_title="Review Priority",
            chunk_index=0,
            source_checksum="b" * 64,
            keywords=("portfolio",),
        ),
    )

    try:
        store.health_check()
        store.ensure_collection(recreate=True)
        assert store.upsert(
            [sales_chunk, finance_only_chunk]
        ) == 2
        assert store.count() == 2

        sales_results = store.search(
            _vector(0.0, 1.0),
            SearchFilters(
                role="sales_rep",
                statuses=("active",),
            ),
            top_k=5,
        )

        assert sales_results
        assert all(
            "sales_rep"
            in result.metadata["access_roles"]
            for result in sales_results
        )
        assert all(
            result.metadata["section_id"] != "PR-2"
            for result in sales_results
        )

        finance_results = store.search(
            _vector(0.0, 1.0),
            SearchFilters(
                role="finance_manager",
                statuses=("active",),
            ),
            top_k=5,
        )

        assert finance_results[0].metadata[
            "section_id"
        ] == "PR-2"

        info = store.collection_info()
        assert info["vector_size"] == 384
        assert {
            "access_roles",
            "status",
            "section_id",
        }.issubset(
            set(info["payload_indexes"])
        )
    finally:
        store.delete_collection()
