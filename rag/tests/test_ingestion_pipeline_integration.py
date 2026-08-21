from __future__ import annotations

import os
import uuid

import pytest

from rag.embeddings.embedder import ChunkEmbedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.metadata.schema import SearchFilters
from rag.vector_store.config import VectorStoreSettings
from rag.vector_store.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INGESTION_INTEGRATION") != "1",
    reason=(
        "Set RUN_INGESTION_INTEGRATION=1 after "
        "starting the Qdrant Docker container."
    ),
)


def test_real_ingestion_pipeline_populates_qdrant():
    collection_name = (
        "swiftrail_ingestion_test_"
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

    store = QdrantVectorStore(
        settings=settings
    )
    embedder = ChunkEmbedder()

    try:
        report = IngestionPipeline(
            embedder=embedder,
            vector_store=store,
        ).run(
            recreate_collection=True
        )

        assert report.documents_loaded == 6
        assert report.chunks_created == 22
        assert report.chunks_embedded == 22
        assert report.points_uploaded == 22
        assert report.points_stored == 22
        assert report.vector_size == 384

        results = store.search(
            embedder.embed_query(
                "Who can release a severe credit hold?"
            ),
            SearchFilters(
                role="finance_manager",
                statuses=("active",),
            ),
            top_k=3,
        )

        assert results
        assert results[0].metadata["doc_id"] == (
            "credit_hold_policy"
        )
        assert results[0].metadata["section_id"] == (
            "CH-3"
        )
    finally:
        store.delete_collection()
