from __future__ import annotations

import os

import pytest

from rag.chunking.chunker import MarkdownChunker
from rag.embeddings.embedder import ChunkEmbedder
from rag.loading.loader import CorpusLoader

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EMBEDDING_INTEGRATION") != "1",
    reason=(
        "Set RUN_EMBEDDING_INTEGRATION=1 to download and test "
        "the real FastEmbed model."
    ),
)


def test_real_fastembed_model_embeds_all_chunks():
    documents = CorpusLoader().load()
    chunks = MarkdownChunker().chunk_documents(documents)

    embedder = ChunkEmbedder()
    embedded = embedder.embed_chunks(chunks)
    query_vector = embedder.embed_query(
        "What does RE-2 require?"
    )

    assert len(embedded) == 22
    assert embedder.dimension == 384
    assert len(query_vector) == 384
    assert all(
        len(item.vector) == 384
        for item in embedded
    )
