from __future__ import annotations

import json
import re
from collections import Counter

from pydantic import TypeAdapter
from rag.config import CORPUS_ROOT, MANIFEST_PATH
from rag.models import DocumentMetadata


def validate_corpus() -> dict:
    errors: list[str] = []
    entries = TypeAdapter(list[DocumentMetadata]).validate_python(
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    )

    ids = [entry.doc_id for entry in entries]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate doc_id values: {duplicates}")

    for entry in entries:
        source = (CORPUS_ROOT / entry.source_path).resolve()
        if not source.exists():
            errors.append(f"Missing source: {entry.source_path}")
            continue

        text = source.read_text(encoding="utf-8").strip()
        if len(text) < 200:
            errors.append(f"{entry.doc_id}: document is too short.")

        pattern = re.compile(
            rf"^##\s+{re.escape(entry.section_prefix)}-\d+",
            re.MULTILINE,
        )
        if not pattern.search(text):
            errors.append(
                f"{entry.doc_id}: missing section IDs with prefix "
                f"{entry.section_prefix}."
            )

        if re.search(
            r"(DB_PASSWORD|API_KEY)\s*=\s*\S+",
            text,
            re.IGNORECASE,
        ):
            errors.append(f"{entry.doc_id}: possible secret found.")

    return {
        "documents": len(entries),
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    report = validate_corpus()
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
