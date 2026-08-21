from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.chunking.chunker import MarkdownChunker
from rag.embeddings.embedder import EmbeddedChunk
from rag.ingestion.pipeline import IngestionPipeline
from rag.loading.loader import CorpusLoader


class StubEmbedder:
    """Deterministic no-download embedder for pipeline tests."""

    model_name = "stub-embedding-model"

    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size

    def embed_chunks(self, chunks):
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=(
                    (1.0,)
                    + (0.0,) * (
                        self.vector_size - 1
                    )
                ),
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]


class StubVectorStore:
    """In-memory replacement for Qdrant."""

    def __init__(self, vector_size: int = 384):
        self.settings = SimpleNamespace(
            vector_size=vector_size
        )
        self.collection_name = "stub_collection"
        self.calls: list[str] = []
        self.points: dict[str, object] = {}

    def health_check(self):
        self.calls.append("health_check")
        return True

    def ensure_collection(self, *, recreate=False):
        self.calls.append(
            f"ensure_collection:{recreate}"
        )
        if recreate:
            self.points.clear()

    def upsert(self, embedded_chunks):
        self.calls.append("upsert")
        chunks = list(embedded_chunks)

        for chunk in chunks:
            self.points[chunk.chunk_id] = chunk

        return len(chunks)

    def count(self):
        self.calls.append("count")
        return len(self.points)

    def collection_info(self):
        return {
            "collection_name": self.collection_name,
            "points_count": len(self.points),
            "vector_size": self.settings.vector_size,
        }


def test_pipeline_ingests_the_complete_corpus():
    vector_store = StubVectorStore()

    pipeline = IngestionPipeline(
        loader=CorpusLoader(),
        chunker=MarkdownChunker(),
        embedder=StubEmbedder(),
        vector_store=vector_store,
    )

    report = pipeline.run(
        recreate_collection=True
    )

    assert report.documents_loaded == 6
    assert report.chunks_created == 22
    assert report.chunks_embedded == 22
    assert report.points_uploaded == 22
    assert report.points_stored == 22
    assert report.vector_size == 384
    assert report.recreated_collection is True

    assert vector_store.calls[:3] == [
        "health_check",
        "ensure_collection:True",
        "upsert",
    ]


def test_pipeline_is_idempotent_with_stable_chunk_ids():
    vector_store = StubVectorStore()

    pipeline = IngestionPipeline(
        loader=CorpusLoader(),
        chunker=MarkdownChunker(),
        embedder=StubEmbedder(),
        vector_store=vector_store,
    )

    first = pipeline.run(
        recreate_collection=True
    )
    second = pipeline.run(
        recreate_collection=False
    )

    assert first.points_stored == 22
    assert second.points_uploaded == 22
    assert second.points_stored == 22


def test_pipeline_rejects_vector_size_mismatch():
    pipeline = IngestionPipeline(
        loader=CorpusLoader(),
        chunker=MarkdownChunker(),
        embedder=StubEmbedder(vector_size=12),
        vector_store=StubVectorStore(
            vector_size=384
        ),
    )

    with pytest.raises(
        ValueError,
        match="vector sizes do not match",
    ):
        pipeline.run(
            recreate_collection=True
        )
