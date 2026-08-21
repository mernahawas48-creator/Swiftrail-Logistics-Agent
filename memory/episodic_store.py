"""Episodic memory: durable store for events promoted by the router.

Only PromoteDropRouter (router.py) writes here. This store never writes
to SemanticMemory -- that only happens through ConsolidationLayer's
periodic pass (consolidation.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

DEFAULT_DB_PATH = os.environ.get(
    "MEMORY_DB_PATH",
    os.path.join(os.path.dirname(__file__), "memory_store.db"),
)


@dataclass
class Episode:
    id: int | None
    customer_id: int | None
    event_type: str
    content: dict[str, Any]
    source_session_id: str | None
    reason: str
    created_at: str
    consolidated: bool = False


class EpisodicMemory:
    """SQLite-backed episodic store, queryable by customer."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_session_id TEXT,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consolidated INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_customer "
                "ON episodes(customer_id)"
            )

    def add_episode(
        self,
        event_type: str,
        content: dict[str, Any],
        reason: str,
        customer_id: int | None = None,
        source_session_id: str | None = None,
    ) -> Episode:
        """Promote one event into durable episodic storage.

        `reason` is the promote-or-drop router's justification for why
        this event was promoted (not dropped) -- kept here so a grader
        can trace every episode back to why it exists.
        """

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodes
                    (customer_id, event_type, content, source_session_id,
                     reason, created_at, consolidated)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    customer_id,
                    event_type,
                    json.dumps(content),
                    source_session_id,
                    reason,
                    created_at,
                ),
            )
            episode_id = cur.lastrowid

        return Episode(
            id=episode_id,
            customer_id=customer_id,
            event_type=event_type,
            content=content,
            source_session_id=source_session_id,
            reason=reason,
            created_at=created_at,
        )

    def get_by_customer(self, customer_id: int) -> list[Episode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE customer_id = ? "
                "ORDER BY created_at ASC",
                (customer_id,),
            ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def get_unconsolidated(self) -> list[Episode]:
        """Episodes not yet processed by a consolidation pass."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE consolidated = 0 "
                "ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def mark_consolidated(self, episode_ids: list[int]) -> None:
        if not episode_ids:
            return
        placeholders = ",".join("?" for _ in episode_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE episodes SET consolidated = 1 "
                f"WHERE id IN ({placeholders})",
                episode_ids,
            )

    def all(self) -> list[Episode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            customer_id=row["customer_id"],
            event_type=row["event_type"],
            content=json.loads(row["content"]),
            source_session_id=row["source_session_id"],
            reason=row["reason"],
            created_at=row["created_at"],
            consolidated=bool(row["consolidated"]),
        )
