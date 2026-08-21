"""Environment-based Qdrant configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True, slots=True)
class VectorStoreSettings:
    """Connection and collection settings for Qdrant."""

    url: str = field(
        default_factory=lambda: os.getenv(
            "QDRANT_URL",
            "http://127.0.0.1:6333",
        )
    )
    api_key: str | None = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY") or None
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv(
            "QDRANT_COLLECTION",
            "swiftrail_knowledge",
        )
    )
    vector_size: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    )
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_TOP_K", "5"))
    )

    def validate(self) -> None:
        if not self.url.strip():
            raise ValueError("QDRANT_URL cannot be empty.")

        if not self.collection_name.strip():
            raise ValueError(
                "QDRANT_COLLECTION cannot be empty."
            )

        if self.vector_size < 1:
            raise ValueError(
                "QDRANT_VECTOR_SIZE must be positive."
            )

        if self.default_top_k < 1:
            raise ValueError(
                "QDRANT_TOP_K must be positive."
            )
