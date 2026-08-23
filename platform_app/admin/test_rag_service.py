import pytest

from platform_app.admin.rag_service import PlatformRAGService


class FakeManager:
    def __init__(self):
        self.docs = [{
            "doc_id": "core_policy", "title": "Core Policy",
            "source_path": "documents/core.md", "section_prefix": "COR",
        }]
        self.added = None

    def list_documents(self):
        return self.docs

    def add_document(self, metadata, text):
        self.added = (metadata, text)
        return {"document": metadata, "reindex": {"points": 8}}

    def remove_document(self, doc_id):
        return {"deleted": doc_id, "reindex": {"points": 7}}


def test_platform_rag_service_builds_valid_corpus_document():
    manager = FakeManager()
    service = PlatformRAGService(lambda: manager)
    result = service.add_document(
        title="New Rail Policy", body="Route shipments by rail.",
        department="operations", document_type="policy",
        access_roles=["sales_rep"], section_prefix="new",
    )

    metadata, markdown = manager.added
    assert metadata["source_path"].startswith("documents/admin/")
    assert metadata["section_prefix"] == "NEW"
    assert "## NEW-1 — Admin Policy" in markdown
    assert result["document"]["removable"] is True


def test_platform_rag_service_protects_built_in_documents():
    service = PlatformRAGService(FakeManager)

    with pytest.raises(PermissionError, match="read-only"):
        service.remove_document("core_policy")
