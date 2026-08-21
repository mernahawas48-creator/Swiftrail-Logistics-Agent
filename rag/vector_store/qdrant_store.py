"""Qdrant storage, indexing, filtering, and vector search."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from qdrant_client import QdrantClient, models

from rag.embeddings.embedder import EmbeddedChunk
from rag.metadata.schema import SearchFilters
from rag.vector_store.config import VectorStoreSettings


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One result returned from Qdrant."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class QdrantVectorStore:
    """Store and retrieve Swiftrail chunk embeddings in Qdrant."""

    PAYLOAD_INDEXES: ClassVar[dict[
        str,
        models.PayloadSchemaType,
    ]] = {
        "doc_id": models.PayloadSchemaType.KEYWORD,
        "status": models.PayloadSchemaType.KEYWORD,
        "department": models.PayloadSchemaType.KEYWORD,
        "document_type": models.PayloadSchemaType.KEYWORD,
        "access_roles": models.PayloadSchemaType.KEYWORD,
        "section_id": models.PayloadSchemaType.KEYWORD,
        "source_path": models.PayloadSchemaType.KEYWORD,
        "version": models.PayloadSchemaType.KEYWORD,
    }

    def __init__(
        self,
        settings: VectorStoreSettings | None = None,
        client: QdrantClient | None = None,
    ):
        self.settings = settings or VectorStoreSettings()
        self.settings.validate()

        self.client = client or QdrantClient(
            url=self.settings.url,
            api_key=self.settings.api_key,
            timeout=30,
        )

    @property
    def collection_name(self) -> str:
        return self.settings.collection_name

    def health_check(self) -> bool:
        """Confirm that the Qdrant server can answer a request."""

        self.client.get_collections()
        return True

    def ensure_collection(
        self,
        *,
        recreate: bool = False,
    ) -> None:
        """Create the collection and payload indexes.

        Payload indexes are created before vectors are uploaded so filtered
        HNSW search can benefit from them during index construction.
        """

        exists = self.client.collection_exists(
            collection_name=self.collection_name
        )

        if exists and recreate:
            self.client.delete_collection(
                collection_name=self.collection_name
            )
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.settings.vector_size,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=1,
                ),
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        collection = self.client.get_collection(
            collection_name=self.collection_name
        )
        payload_schema = getattr(
            collection,
            "payload_schema",
            {},
        ) or {}
        existing_fields = set(payload_schema.keys())

        for field_name, field_schema in self.PAYLOAD_INDEXES.items():
            if field_name in existing_fields:
                continue

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    def upsert(
        self,
        embedded_chunks: Iterable[EmbeddedChunk],
    ) -> int:
        """Insert or replace embedded chunks using stable chunk UUIDs."""

        points: list[models.PointStruct] = []

        for embedded_chunk in embedded_chunks:
            vector = self._validate_vector(
                embedded_chunk.vector
            )

            metadata = self._metadata_to_payload(
                embedded_chunk.metadata
            )

            payload = {
                **metadata,
                "text": embedded_chunk.text,
            }

            points.append(
                models.PointStruct(
                    id=embedded_chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return 0

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query_vector: Sequence[float],
        filters: SearchFilters,
        *,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Run ANN search with metadata filtering inside Qdrant."""

        vector = self._validate_vector(query_vector)
        limit = top_k or self.settings.default_top_k

        if limit < 1:
            raise ValueError("top_k must be positive.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=self._build_filter(filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
            search_params=models.SearchParams(
                hnsw_ef=64,
                exact=False,
            ),
        )

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))

            results.append(
                VectorSearchResult(
                    chunk_id=str(point.id),
                    score=float(point.score),
                    text=text,
                    metadata=payload,
                )
            )

        return results

    def count(self) -> int:
        """Return the exact number of stored points."""

        response = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(response.count)

    def collection_info(self) -> dict[str, Any]:
        """Return configuration evidence for tests and the demo."""

        collection = self.client.get_collection(
            collection_name=self.collection_name
        )

        vectors = collection.config.params.vectors
        hnsw = collection.config.hnsw_config
        payload_schema = getattr(
            collection,
            "payload_schema",
            {},
        ) or {}

        return {
            "collection_name": self.collection_name,
            "points_count": collection.points_count,
            "indexed_vectors_count": (
                collection.indexed_vectors_count
            ),
            "vector_size": getattr(vectors, "size", None),
            "distance": str(
                getattr(vectors, "distance", None)
            ),
            "hnsw_m": getattr(hnsw, "m", None),
            "hnsw_ef_construct": getattr(
                hnsw,
                "ef_construct",
                None,
            ),
            "payload_indexes": sorted(
                payload_schema.keys()
            ),
        }

    def delete_collection(self) -> None:
        """Delete the configured collection when it exists."""

        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            self.client.delete_collection(
                collection_name=self.collection_name
            )

    def _validate_vector(
        self,
        vector: Sequence[float],
    ) -> list[float]:
        values = [float(value) for value in vector]

        if len(values) != self.settings.vector_size:
            raise ValueError(
                "Vector size does not match the Qdrant collection. "
                f"Expected {self.settings.vector_size}, "
                f"received {len(values)}."
            )

        return values

    @staticmethod
    def _metadata_to_payload(
        metadata: Any,
    ) -> dict[str, Any]:
        if hasattr(metadata, "__dataclass_fields__"):
            payload = asdict(metadata)
        elif hasattr(metadata, "model_dump"):
            payload = metadata.model_dump()
        elif isinstance(metadata, dict):
            payload = dict(metadata)
        else:
            raise TypeError(
                "Chunk metadata must be a dataclass, Pydantic model, "
                "or dictionary."
            )

        return {
            key: list(value)
            if isinstance(value, tuple)
            else value
            for key, value in payload.items()
        }

    @staticmethod
    def _match_condition(
        field_name: str,
        values: Sequence[str],
    ) -> models.FieldCondition:
        if len(values) == 1:
            match: models.Match = models.MatchValue(
                value=values[0]
            )
        else:
            match = models.MatchAny(
                any=list(values)
            )

        return models.FieldCondition(
            key=field_name,
            match=match,
        )

    def _build_filter(
        self,
        filters: SearchFilters,
    ) -> models.Filter:
        must: list[models.Condition] = [
            self._match_condition(
                "access_roles",
                [filters.role],
            ),
            self._match_condition(
                "status",
                list(filters.statuses),
            ),
        ]

        optional_fields = {
            "department": filters.departments,
            "document_type": filters.document_types,
            "doc_id": filters.doc_ids,
            "section_id": filters.section_ids,
        }

        for field_name, values in optional_fields.items():
            if values:
                must.append(
                    self._match_condition(
                        field_name,
                        list(values),
                    )
                )

        return models.Filter(must=must)
