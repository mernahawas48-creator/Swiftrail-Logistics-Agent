"""Load Swiftrail Markdown corpus files into Python objects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = PROJECT_ROOT / "rag" / "corpus"
DEFAULT_MANIFEST_PATH = CORPUS_ROOT / "manifest.json"

REQUIRED_METADATA_FIELDS = {
    "doc_id",
    "title",
    "version",
    "effective_date",
    "status",
    "department",
    "document_type",
    "access_roles",
    "source_path",
    "section_prefix",
}


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """One corpus document after reading it from disk."""

    metadata: dict[str, Any]
    text: str
    checksum: str
    source_file: Path


class CorpusLoader:
    """Read and validate documents listed in the corpus manifest."""

    def __init__(self, manifest_path: Path | None = None):
        self.manifest_path = (
            manifest_path.resolve()
            if manifest_path
            else DEFAULT_MANIFEST_PATH.resolve()
        )
        self.corpus_root = self.manifest_path.parent.resolve()

    def load(self) -> list[LoadedDocument]:
        manifest_entries = self._read_manifest()
        self._validate_unique_document_ids(manifest_entries)

        documents: list[LoadedDocument] = []
        for entry in manifest_entries:
            self._validate_metadata(entry)

            source_file = self._resolve_source_path(entry["source_path"])
            text = self._read_document(source_file)
            checksum = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            documents.append(
                LoadedDocument(
                    metadata=entry,
                    text=text,
                    checksum=checksum,
                    source_file=source_file,
                )
            )

        return documents

    def _read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Corpus manifest was not found: {self.manifest_path}"
            )

        try:
            data = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corpus manifest contains invalid JSON: {exc}"
            ) from exc

        if not isinstance(data, list):
            raise TypeError(
                "Corpus manifest must contain a JSON list of documents."
            )

        if not data:
            raise ValueError("Corpus manifest does not contain documents.")

        if not all(isinstance(item, dict) for item in data):
            raise TypeError(
                "Every corpus manifest entry must be a JSON object."
            )

        return data

    @staticmethod
    def _validate_unique_document_ids(
        entries: list[dict[str, Any]],
    ) -> None:
        document_ids = [
            entry.get("doc_id")
            for entry in entries
            if entry.get("doc_id")
        ]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                "Corpus manifest contains duplicate doc_id values."
            )

    @staticmethod
    def _validate_metadata(entry: dict[str, Any]) -> None:
        missing = sorted(REQUIRED_METADATA_FIELDS - entry.keys())
        if missing:
            document_name = entry.get("doc_id", "<unknown>")
            raise ValueError(
                f"Document {document_name} is missing metadata fields: "
                f"{', '.join(missing)}"
            )

        if not isinstance(entry["access_roles"], list):
            raise TypeError(
                f"Document {entry['doc_id']} access_roles must be a list."
            )

        if not entry["access_roles"]:
            raise ValueError(
                f"Document {entry['doc_id']} must allow at least one role."
            )

    def _resolve_source_path(self, source_path: str) -> Path:
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be a non-empty string.")

        source_file = (
            self.corpus_root / source_path
        ).resolve()

        try:
            source_file.relative_to(self.corpus_root)
        except ValueError as exc:
            raise ValueError(
                f"Unsafe corpus source path: {source_path}"
            ) from exc

        if not source_file.exists():
            raise FileNotFoundError(
                f"Corpus document was not found: {source_file}"
            )

        if not source_file.is_file():
            raise ValueError(
                f"Corpus source path is not a file: {source_file}"
            )

        return source_file

    @staticmethod
    def _read_document(source_file: Path) -> str:
        try:
            text = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Corpus document must use UTF-8 encoding: {source_file}"
            ) from exc

        # Use the same line endings on Windows, Linux, and macOS.
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not normalized:
            raise ValueError(
                f"Corpus document is empty: {source_file}"
            )

        return normalized


def main() -> None:
    documents = CorpusLoader().load()

    print(f"Loaded documents: {len(documents)}")
    for document in documents:
        print(
            f"- {document.metadata['doc_id']}: "
            f"{len(document.text)} characters, "
            f"checksum={document.checksum[:12]}"
        )


if __name__ == "__main__":
    main()
