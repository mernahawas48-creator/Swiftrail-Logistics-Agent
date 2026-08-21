from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from rag.chunking.chunker import MarkdownChunker
from rag.loading.loader import CorpusLoader
from rag.metadata.schema import (
    ChunkMetadataSchema,
    DocumentMetadataSchema,
    SearchFilters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "rag" / "corpus" / "manifest.json"


def test_manifest_entries_match_document_metadata_schema():
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    documents = TypeAdapter(
        list[DocumentMetadataSchema]
    ).validate_python(manifest)

    assert len(documents) == 7
    assert len({document.doc_id for document in documents}) == 7

    for document in documents:
        assert document.status == "active"
        assert document.source_path.startswith("documents/")
        assert document.source_path.endswith(".md")


def test_generated_chunks_match_chunk_metadata_schema():
    documents = CorpusLoader().load()
    chunks = MarkdownChunker().chunk_documents(documents)

    validated = [
        ChunkMetadataSchema.model_validate(
            asdict(chunk.metadata)
        )
        for chunk in chunks
    ]

    assert len(validated) == 27
    assert all(
        len(metadata.source_checksum) == 64
        for metadata in validated
    )


def test_search_filters_accept_valid_values():
    filters = SearchFilters(
        role="finance_manager",
        statuses=("active",),
        departments=("finance",),
        section_ids=("RE-2", "CH-3"),
    )

    assert filters.role == "finance_manager"
    assert filters.statuses == ("active",)


def test_unsafe_source_path_is_rejected():
    with pytest.raises(ValidationError):
        DocumentMetadataSchema(
            doc_id="unsafe_document",
            title="Unsafe Document Example",
            version="1.0",
            effective_date="2026-08-01",
            status="active",
            department="finance",
            document_type="policy",
            access_roles=("sales_rep",),
            source_path="../../.env",
            section_prefix="UN",
            keywords=(),
        )


def test_unknown_metadata_field_is_rejected():
    with pytest.raises(ValidationError):
        SearchFilters(
            role="sales_rep",
            unknown_filter="not_allowed",
        )
