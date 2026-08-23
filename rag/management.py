from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any

from rag.ingestion.pipeline import IngestionPipeline
from rag.loading.loader import CorpusLoader
from rag.metadata.schema import DocumentMetadataSchema
from rag.vector_store.qdrant_store import QdrantVectorStore


class RAGDocumentManager:
    """Manage the real corpus and keep it synchronized with Qdrant."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        *,
        vector_store: Any | None = None,
        pipeline_factory: Callable[[], Any] | None = None,
    ):
        root = Path(__file__).resolve().parent / "corpus"
        self.manifest_path = (manifest_path or root / "manifest.json").resolve()
        self.corpus_root = self.manifest_path.parent
        self.vector_store = vector_store or QdrantVectorStore()
        self._pipeline_factory = pipeline_factory or self._build_pipeline
        self._lock = RLock()

    def list_documents(self) -> list[dict[str, Any]]:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError("Corpus manifest must contain a JSON list.")
        return data

    def add_document(self, metadata: dict[str, Any], text: str) -> dict[str, Any]:
        normalized = self._validate_metadata(metadata)
        clean_text = self._validate_text(text)
        with self._lock:
            docs = self.list_documents()
            if any(doc["doc_id"] == normalized["doc_id"] for doc in docs):
                raise ValueError(f"Document already exists: {normalized['doc_id']}")
            target = self._source_path(normalized["source_path"])
            if target.exists():
                raise ValueError(f"Corpus file already exists: {normalized['source_path']}")
            manifest_before = self.manifest_path.read_bytes()
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(clean_text + "\n", encoding="utf-8")
                self._write_manifest([*docs, normalized])
                report = self.reindex()
            except Exception:
                self.manifest_path.write_bytes(manifest_before)
                target.unlink(missing_ok=True)
                raise
            return {"document": normalized, "reindex": report}

    def remove_document(self, doc_id: str) -> dict[str, Any]:
        with self._lock:
            docs = self.list_documents()
            match = next((doc for doc in docs if doc["doc_id"] == doc_id), None)
            if match is None:
                raise KeyError(doc_id)
            source = self._source_path(match["source_path"])
            manifest_before = self.manifest_path.read_bytes()
            source_before = source.read_bytes() if source.exists() else None
            try:
                source.unlink(missing_ok=True)
                self._write_manifest([doc for doc in docs if doc["doc_id"] != doc_id])
                report = self.reindex()
            except Exception:
                self.manifest_path.write_bytes(manifest_before)
                if source_before is not None:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(source_before)
                raise
            return {"deleted": doc_id, "reindex": report}

    def update_document(self, doc_id: str, text: str) -> dict[str, Any]:
        clean_text = self._validate_text(text)
        with self._lock:
            docs = self.list_documents()
            match = next((doc for doc in docs if doc["doc_id"] == doc_id), None)
            if match is None:
                raise KeyError(doc_id)
            target = self._source_path(match["source_path"])
            previous = target.read_bytes()
            try:
                target.write_text(clean_text + "\n", encoding="utf-8")
                report = self.reindex()
            except Exception:
                target.write_bytes(previous)
                raise
            return {"document": match, "reindex": report}

    def reindex(self) -> dict[str, Any]:
        report = self._pipeline_factory().run(recreate_collection=True)
        return {
            "documents": report.documents_loaded,
            "chunks": report.chunks_created,
            "points": report.points_uploaded,
            "collection": report.collection_name,
        }

    def _build_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            loader=CorpusLoader(self.manifest_path),
            vector_store=self.vector_store,
        )

    def _source_path(self, relative: str) -> Path:
        target = (self.corpus_root / relative).resolve()
        try:
            target.relative_to(self.corpus_root)
        except ValueError as exc:
            raise ValueError("Document path must stay inside the corpus.") from exc
        return target

    def _write_manifest(self, docs: list[dict[str, Any]]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(docs, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        validated = DocumentMetadataSchema.model_validate(metadata)
        return validated.model_dump(mode="json")

    @staticmethod
    def _validate_text(text: str) -> str:
        clean = text.strip()
        if not clean:
            raise ValueError("Document text cannot be empty.")
        return clean
