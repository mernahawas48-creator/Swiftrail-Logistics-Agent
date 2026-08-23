import json
from types import SimpleNamespace

import pytest

from rag.management import RAGDocumentManager


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail

    def run(self, *, recreate_collection):
        assert recreate_collection is True
        if self.fail:
            raise RuntimeError("Qdrant unavailable")
        return SimpleNamespace(
            documents_loaded=2, chunks_created=3,
            points_uploaded=3, collection_name="test-corpus",
        )


def metadata(doc_id="admin_policy_1"):
    return {
        "doc_id": doc_id, "title": "Admin Test Policy", "version": "1.0",
        "effective_date": "2026-08-23", "status": "active",
        "department": "operations", "document_type": "policy",
        "access_roles": ["sales_rep"],
        "source_path": f"documents/admin/{doc_id}.md",
        "section_prefix": "ADM", "keywords": ["admin"],
    }


def make_manager(tmp_path, *, fail=False):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]\n", encoding="utf-8")
    manager = RAGDocumentManager(
        manifest, vector_store=object(),
        pipeline_factory=lambda: FakePipeline(fail),
    )
    return manager, manifest


def test_add_document_updates_corpus_and_returns_reindex_evidence(tmp_path):
    manager, manifest = make_manager(tmp_path)
    result = manager.add_document(metadata(), "## ADM-1 — Policy\n\nUse rail.")

    assert result["reindex"]["points"] == 3
    assert json.loads(manifest.read_text(encoding="utf-8"))[0]["doc_id"] == "admin_policy_1"
    assert (tmp_path / "documents/admin/admin_policy_1.md").exists()


def test_add_document_rolls_back_when_reindex_fails(tmp_path):
    manager, manifest = make_manager(tmp_path, fail=True)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        manager.add_document(metadata(), "## ADM-1 — Policy\n\nUse rail.")

    assert manifest.read_bytes() == before
    assert not (tmp_path / "documents/admin/admin_policy_1.md").exists()


def test_remove_document_rolls_back_when_reindex_fails(tmp_path):
    manager, manifest = make_manager(tmp_path, fail=True)
    doc = metadata()
    manifest.write_text(json.dumps([doc]) + "\n", encoding="utf-8")
    source = tmp_path / doc["source_path"]
    source.parent.mkdir(parents=True)
    source.write_text("original\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        manager.remove_document(doc["doc_id"])

    assert json.loads(manifest.read_text(encoding="utf-8")) == [doc]
    assert source.read_text(encoding="utf-8") == "original\n"
