from __future__ import annotations

import math

from rag.chunking.chunker import MarkdownChunker
from rag.embeddings.embedder import ChunkEmbedder
from rag.loading.loader import CorpusLoader


class FakeEmbeddingModel:
    """Deterministic model used to test our code without a download."""

    def embed(self, documents: list[str]):
        for index, document in enumerate(documents, start=1):
            yield [
                float(index),
                float(len(document) % 17 + 1),
                2.0,
                3.0,
            ]

    def query_embed(self, query: str):
        yield [
            float(len(query) % 13 + 1),
            2.0,
            4.0,
            8.0,
        ]


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_embedder_preserves_all_chunks_and_metadata():
    documents = CorpusLoader().load()
    chunks = MarkdownChunker().chunk_documents(documents)

    embedder = ChunkEmbedder(
        model_name="fake-model",
        model=FakeEmbeddingModel(),
    )
    embedded = embedder.embed_chunks(chunks)

    assert len(chunks) == 27
    assert len(embedded) == 27
    assert embedder.dimension == 4

    for original, result in zip(
        chunks,
        embedded,
        strict=True,
    ):
        assert result.chunk_id == original.chunk_id
        assert result.text == original.text
        assert result.metadata == original.metadata
        assert len(result.vector) == 4
        assert math.isclose(
            _norm(result.vector),
            1.0,
            rel_tol=1e-9,
        )


def test_query_embedding_uses_one_normalized_vector():
    embedder = ChunkEmbedder(
        model_name="fake-model",
        model=FakeEmbeddingModel(),
    )

    vector = embedder.embed_query(
        "Who can release a severe credit hold?"
    )

    assert len(vector) == 4
    assert math.isclose(
        _norm(vector),
        1.0,
        rel_tol=1e-9,
    )


def test_empty_query_is_rejected():
    embedder = ChunkEmbedder(
        model_name="fake-model",
        model=FakeEmbeddingModel(),
    )

    try:
        embedder.embed_query("   ")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("An empty query should be rejected.")
