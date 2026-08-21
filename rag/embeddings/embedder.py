"""Convert Swiftrail document chunks and user queries into dense vectors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.chunking.chunker import ChunkMetadata, DocumentChunk

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel(Protocol):
    """Small protocol that allows FastEmbed or a test model to be used."""

    def embed(
        self,
        documents: list[str],
    ) -> Iterable[Sequence[float]]:
        ...

    def query_embed(
        self,
        query: str,
    ) -> Iterable[Sequence[float]]:
        ...


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """One chunk paired with its normalized dense vector."""

    chunk_id: str
    vector: tuple[float, ...]
    text: str
    metadata: ChunkMetadata


class ChunkEmbedder:
    """Generate separate document and query embeddings with FastEmbed."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: EmbeddingModel | None = None,
    ):
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.model_name = model_name
        self._model = model
        self._dimension: int | None = None

    @property
    def model(self) -> EmbeddingModel:
        """Load the real FastEmbed model only when it is first needed."""

        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    'FastEmbed is not installed. Run: '
                    'pip install "fastembed>=0.8,<0.9"'
                ) from exc

            self._model = TextEmbedding(
                model_name=self.model_name,
            )

        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector size after embedding at least one item."""

        if self._dimension is None:
            self.embed_query("Swiftrail embedding dimension probe")
        assert self._dimension is not None
        return self._dimension

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """Embed chunk texts while preserving IDs, text, and metadata."""

        if not chunks:
            return []

        for chunk in chunks:
            if not chunk.text.strip():
                raise ValueError(
                    f"Chunk {chunk.chunk_id} contains empty text."
                )

        raw_vectors = list(
            self.model.embed(
                [chunk.text for chunk in chunks]
            )
        )

        if len(raw_vectors) != len(chunks):
            raise RuntimeError(
                "The embedding model returned a different number of "
                "vectors than the number of chunks."
            )

        embedded: list[EmbeddedChunk] = []

        for chunk, raw_vector in zip(
            chunks,
            raw_vectors,
            strict=True,
        ):
            vector = self._prepare_vector(raw_vector)

            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    vector=vector,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )

        return embedded

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Create a query vector using the model's query-specific method."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("The query cannot be empty.")

        raw_vectors = list(
            self.model.query_embed(normalized_query)
        )

        if len(raw_vectors) != 1:
            raise RuntimeError(
                "The embedding model must return exactly one query vector."
            )

        return self._prepare_vector(raw_vectors[0])

    def _prepare_vector(
        self,
        raw_vector: Sequence[float],
    ) -> tuple[float, ...]:
        values = tuple(float(value) for value in raw_vector)

        if not values:
            raise ValueError("The embedding model returned an empty vector.")

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "The embedding model returned a non-finite vector value."
            )

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if norm == 0.0:
            raise ValueError("The embedding model returned a zero vector.")

        normalized = tuple(
            value / norm
            for value in values
        )

        if self._dimension is None:
            self._dimension = len(normalized)
        elif len(normalized) != self._dimension:
            raise ValueError(
                "The embedding model returned inconsistent vector sizes."
            )

        return normalized


def main() -> None:
    from rag.chunking.chunker import MarkdownChunker
    from rag.loading.loader import CorpusLoader

    documents = CorpusLoader().load()
    chunks = MarkdownChunker().chunk_documents(documents)

    embedder = ChunkEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)
    query_vector = embedder.embed_query(
        "Who can release a severe credit hold?"
    )

    print(f"Model: {embedder.model_name}")
    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"Vector dimension: {embedder.dimension}")
    print(f"Query vector dimension: {len(query_vector)}")
    print(
        "First embedded chunk: "
        f"{embedded_chunks[0].metadata.doc_id} | "
        f"{embedded_chunks[0].metadata.section_id} | "
        f"{embedded_chunks[0].chunk_id}"
    )


if __name__ == "__main__":
    main()
