from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from rag.management import RAGDocumentManager


class PlatformRAGService:
    """Translate admin UI requests into real corpus mutations."""

    def __init__(
        self,
        manager_factory: Callable[[], RAGDocumentManager] = RAGDocumentManager,
    ):
        self._manager_factory = manager_factory
        self._manager: RAGDocumentManager | None = None

    @property
    def manager(self) -> RAGDocumentManager:
        if self._manager is None:
            self._manager = self._manager_factory()
        return self._manager

    def list_documents(self) -> list[dict[str, Any]]:
        return [self._for_ui(doc) for doc in self.manager.list_documents()]

    def add_document(
        self, *, title: str, body: str, department: str,
        document_type: str, access_roles: list[str], section_prefix: str,
    ) -> dict[str, Any]:
        prefix = section_prefix.strip().upper()
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:45]
        doc_id = f"admin_{slug}_{uuid.uuid4().hex[:8]}"
        metadata = {
            "doc_id": doc_id, "title": title, "version": "1.0",
            "effective_date": datetime.now(UTC).date().isoformat(),
            "status": "active",
            "department": department, "document_type": document_type,
            "access_roles": access_roles,
            "source_path": f"documents/admin/{doc_id}.md",
            "section_prefix": prefix,
            "keywords": sorted(set(re.findall(r"[a-z0-9]+", title.lower()))),
        }
        markdown = f"# {title.strip()}\n\n## {prefix}-1 — Admin Policy\n\n{body.strip()}"
        result = self.manager.add_document(metadata, markdown)
        result["document"] = self._for_ui(result["document"])
        return result

    def remove_document(self, doc_id: str) -> dict[str, Any]:
        doc = self._find(doc_id)
        if not doc["source_path"].startswith("documents/admin/"):
            raise PermissionError("Built-in course corpus documents are read-only.")
        return self.manager.remove_document(doc_id)

    def update_document(self, doc_id: str, body: str) -> dict[str, Any]:
        doc = self._find(doc_id)
        if not doc["source_path"].startswith("documents/admin/"):
            raise PermissionError("Built-in course corpus documents are read-only.")
        prefix = doc["section_prefix"]
        markdown = f"# {doc['title']}\n\n## {prefix}-1 — Admin Policy\n\n{body.strip()}"
        return self.manager.update_document(doc_id, markdown)

    def reindex(self) -> dict[str, Any]:
        return self.manager.reindex()

    def _find(self, doc_id: str) -> dict[str, Any]:
        match = next(
            (
                doc
                for doc in self.manager.list_documents()
                if doc["doc_id"] == doc_id
            ),
            None,
        )
        if match is None:
            raise KeyError(doc_id)
        return match

    @staticmethod
    def _for_ui(doc: dict[str, Any]) -> dict[str, Any]:
        return {
            **doc,
            "removable": doc["source_path"].startswith("documents/admin/"),
        }
