from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.ingestion.pipeline import IngestionPipeline
from rag.vector_store.qdrant_store import QdrantVectorStore


class RAGDocumentManager:
    """Admin-side document lifecycle for the existing Swiftrail RAG corpus."""

    def __init__(self, manifest_path: Path | None = None):
        root = Path(__file__).resolve().parent / "corpus"
        self.manifest_path = (manifest_path or root / "manifest.json").resolve()
        self.corpus_root = self.manifest_path.parent
        self.vector_store = QdrantVectorStore()

    def list_documents(self) -> list[dict[str, Any]]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def add_document(self, metadata: dict[str, Any], text: str) -> dict[str, Any]:
        self._validate_metadata(metadata)
        docs = self.list_documents()
        if any(d["doc_id"] == metadata["doc_id"] for d in docs):
            raise ValueError(f"Document already exists: {metadata['doc_id']}")
        relative = metadata["source_path"]
        target = (self.corpus_root / relative).resolve()
        target.relative_to(self.corpus_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.strip() + "\n", encoding="utf-8")
        docs.append(metadata)
        self._write_manifest(docs)
        self.reindex()
        return metadata

    def remove_document(self, doc_id: str) -> None:
        docs = self.list_documents()
        match = next((d for d in docs if d["doc_id"] == doc_id), None)
        if match is None:
            raise KeyError(doc_id)
        source = (self.corpus_root / match["source_path"]).resolve()
        source.relative_to(self.corpus_root)
        if source.exists():
            source.unlink()
        self._write_manifest([d for d in docs if d["doc_id"] != doc_id])
        # Rebuild the existing collection from the remaining corpus so both
        # dense Qdrant and lexical BM25 reflect the deletion.
        self.reindex()

    def reindex(self) -> dict[str, Any]:
        report = IngestionPipeline(vector_store=self.vector_store).run(recreate_collection=True)
        return {
            "documents": report.documents_loaded,
            "chunks": report.chunks_created,
            "points": report.points_uploaded,
            "collection": report.collection_name,
        }

    def update_document(self, doc_id: str, text: str) -> dict[str, Any]:
        docs = self.list_documents()
        match = next((d for d in docs if d["doc_id"] == doc_id), None)
        if match is None:
            raise KeyError(doc_id)
        target = (self.corpus_root / match["source_path"]).resolve()
        target.relative_to(self.corpus_root)
        target.write_text(text.strip() + "\n", encoding="utf-8")
        return self.reindex()

    def _write_manifest(self, docs: list[dict[str, Any]]) -> None:
        self.manifest_path.write_text(json.dumps(docs, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        required = {"doc_id", "title", "version", "effective_date", "status", "department", "document_type", "access_roles", "source_path", "section_prefix"}
        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"Missing metadata: {sorted(missing)}")
        if not metadata["access_roles"]:
            raise ValueError("access_roles cannot be empty")
