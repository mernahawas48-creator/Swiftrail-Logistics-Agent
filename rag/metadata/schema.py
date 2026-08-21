"""Pydantic metadata schemas used by the Swiftrail RAG pipeline."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

Role = Literal["sales_rep", "finance_manager"]
DocumentStatus = Literal["active", "archived"]


class StrictMetadataModel(BaseModel):
    """Reject unknown fields and normalize surrounding whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )


class DocumentMetadataSchema(StrictMetadataModel):
    """Metadata stored once for every document in manifest.json."""

    doc_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_]*$",
    )
    title: str = Field(min_length=5, max_length=180)
    version: str = Field(pattern=r"^\d+\.\d+(?:\.\d+)?$")
    effective_date: date
    status: DocumentStatus
    department: str = Field(min_length=2, max_length=80)
    document_type: str = Field(min_length=2, max_length=80)
    access_roles: tuple[Role, ...] = Field(min_length=1)
    source_path: str = Field(min_length=5, max_length=240)
    section_prefix: str = Field(pattern=r"^[A-Z]{2,5}$")
    keywords: tuple[str, ...] = ()

    @field_validator("access_roles")
    @classmethod
    def access_roles_must_be_unique(
        cls,
        value: tuple[Role, ...],
    ) -> tuple[Role, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "access_roles must not contain duplicate roles."
            )
        return value

    @field_validator("source_path")
    @classmethod
    def source_path_must_be_safe(
        cls,
        value: str,
    ) -> str:
        normalized = value.replace("\\", "/")

        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError(
                "source_path must stay inside rag/corpus."
            )

        if not normalized.endswith(".md"):
            raise ValueError(
                "source_path must reference a Markdown file."
            )

        return normalized

    @field_validator("keywords")
    @classmethod
    def keywords_must_be_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            keyword.lower()
            for keyword in value
            if keyword.strip()
        )

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "keywords must not contain duplicates."
            )

        return normalized


class ChunkMetadataSchema(StrictMetadataModel):
    """Metadata attached to every searchable document chunk."""

    doc_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_]*$",
    )
    title: str = Field(min_length=5, max_length=180)
    version: str = Field(pattern=r"^\d+\.\d+(?:\.\d+)?$")
    effective_date: date
    status: DocumentStatus
    department: str = Field(min_length=2, max_length=80)
    document_type: str = Field(min_length=2, max_length=80)
    access_roles: tuple[Role, ...] = Field(min_length=1)
    source_path: str = Field(min_length=5, max_length=240)
    section_id: str = Field(
        pattern=r"^[A-Z]{2,5}-\d+(?:\.\d+)?$",
    )
    section_title: str = Field(min_length=2, max_length=180)
    chunk_index: int = Field(ge=0)
    source_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    keywords: tuple[str, ...] = ()

    @field_validator("access_roles")
    @classmethod
    def chunk_roles_must_be_unique(
        cls,
        value: tuple[Role, ...],
    ) -> tuple[Role, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "access_roles must not contain duplicate roles."
            )
        return value

    @field_validator("source_path")
    @classmethod
    def chunk_source_path_must_be_safe(
        cls,
        value: str,
    ) -> str:
        normalized = value.replace("\\", "/")

        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError(
                "source_path must stay inside rag/corpus."
            )

        if not normalized.endswith(".md"):
            raise ValueError(
                "source_path must reference a Markdown file."
            )

        return normalized


class SearchFilters(StrictMetadataModel):
    """Metadata filters used before or during vector retrieval."""

    role: Role
    statuses: tuple[DocumentStatus, ...] = ("active",)
    departments: tuple[str, ...] | None = None
    document_types: tuple[str, ...] | None = None
    doc_ids: tuple[str, ...] | None = None
    section_ids: tuple[str, ...] | None = None

    @field_validator(
        "statuses",
        "departments",
        "document_types",
        "doc_ids",
        "section_ids",
    )
    @classmethod
    def filter_values_must_be_unique(
        cls,
        value: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None

        if len(value) != len(set(value)):
            raise ValueError(
                "Filter values must not contain duplicates."
            )

        return value
