"""Split loaded Swiftrail Markdown documents into section-aware chunks."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from rag.loading.loader import LoadedDocument

SECTION_PATTERN = re.compile(
    r"^##\s+([A-Z]{2,5}-\d+)\s+[—-]\s+(.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata copied from the source document and its policy section."""

    doc_id: str
    title: str
    version: str
    effective_date: str
    status: str
    department: str
    document_type: str
    access_roles: tuple[str, ...]
    source_path: str
    section_id: str
    section_title: str
    chunk_index: int
    source_checksum: str
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One searchable section or section fragment."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata


class MarkdownChunker:
    """Create section-aware chunks with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 120,
    ):
        if chunk_size < 200:
            raise ValueError("chunk_size must be at least 200 characters.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(
        self,
        documents: list[LoadedDocument],
    ) -> list[DocumentChunk]:
        """Chunk all loaded documents and return one combined list."""

        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.chunk_document(document))

        return chunks

    def chunk_document(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        """Split one loaded document by its Markdown policy sections."""

        matches = list(SECTION_PATTERN.finditer(document.text))

        if not matches:
            raise ValueError(
                f"Document {document.metadata.get('doc_id', '<unknown>')} "
                "does not contain valid section headings."
            )

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section_position, match in enumerate(matches):
            section_id = match.group(1)
            section_title = match.group(2).strip()

            section_start = match.end()
            section_end = (
                matches[section_position + 1].start()
                if section_position + 1 < len(matches)
                else len(document.text)
            )

            section_text = document.text[
                section_start:section_end
            ].strip()

            if not section_text:
                raise ValueError(
                    f"Section {section_id} in "
                    f"{document.metadata['doc_id']} is empty."
                )

            for piece in self._split_long_section(section_text):
                chunk_text = (
                    f"Document: {document.metadata['title']}\n"
                    f"Section: {section_id} — {section_title}\n\n"
                    f"{piece}"
                )

                stable_value = (
                    f"{document.metadata['doc_id']}|"
                    f"{section_id}|"
                    f"{chunk_index}|"
                    f"{chunk_text}"
                )

                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        stable_value,
                    )
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=self._build_metadata(
                            document=document,
                            section_id=section_id,
                            section_title=section_title,
                            chunk_index=chunk_index,
                        ),
                    )
                )

                chunk_index += 1

        return chunks

    def _split_long_section(self, text: str) -> list[str]:
        """Keep short sections whole and split long ones with overlap."""

        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(text), step):
            piece = text[start:start + self.chunk_size].strip()

            if piece:
                pieces.append(piece)

            if start + self.chunk_size >= len(text):
                break

        return pieces

    @staticmethod
    def _build_metadata(
        document: LoadedDocument,
        section_id: str,
        section_title: str,
        chunk_index: int,
    ) -> ChunkMetadata:
        metadata: dict[str, Any] = document.metadata

        return ChunkMetadata(
            doc_id=str(metadata["doc_id"]),
            title=str(metadata["title"]),
            version=str(metadata["version"]),
            effective_date=str(metadata["effective_date"]),
            status=str(metadata["status"]),
            department=str(metadata["department"]),
            document_type=str(metadata["document_type"]),
            access_roles=tuple(metadata["access_roles"]),
            source_path=str(metadata["source_path"]),
            section_id=section_id,
            section_title=section_title,
            chunk_index=chunk_index,
            source_checksum=document.checksum,
            keywords=tuple(metadata.get("keywords", [])),
        )


def main() -> None:
    from rag.loading.loader import CorpusLoader

    documents = CorpusLoader().load()
    chunks = MarkdownChunker().chunk_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks:
        print(
            f"- {chunk.metadata.doc_id} | "
            f"{chunk.metadata.section_id} | "
            f"chunk_index={chunk.metadata.chunk_index}"
        )


if __name__ == "__main__":
    main()
