"""Semantic memory: versioned, stable facts derived from episodic memory.

Nothing outside ConsolidationLayer is allowed to call ``upsert_fact``.
Facts are never overwritten silently:
    - Updates create a NEW row with an incremented version.
    - The previous active row is marked ``superseded`` and points at the
      row that replaced it, so the old value is never lost.
    - Facts carry an ``expires_at``; stale facts are marked ``expired``.
    - Conflicts create a new version and keep the full fact history.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

DEFAULT_DB_PATH = os.environ.get(
    "MEMORY_DB_PATH",
    os.path.join(os.path.dirname(__file__), "memory_store.db"),
)
DEFAULT_TTL_DAYS = 90


@dataclass
class SemanticFact:
    id: int | None
    customer_id: int
    fact_key: str
    fact_value: str
    version: int
    status: str
    source_episode_id: int | None
    conflict_reason: str | None
    superseded_by: int | None
    created_at: str
    expires_at: str


class SemanticMemory:
    """SQLite-backed semantic fact store."""

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
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    source_episode_id INTEGER,
                    conflict_reason TEXT,
                    superseded_by INTEGER,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_lookup "
                "ON semantic_facts(customer_id, fact_key, status)"
            )

    def get_active_fact(self, customer_id: int, fact_key: str) -> SemanticFact | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_facts WHERE customer_id = ? "
                "AND fact_key = ? AND status = 'active' "
                "ORDER BY version DESC LIMIT 1",
                (customer_id, fact_key),
            ).fetchone()
        return self._row_to_fact(row) if row else None

    def get_active_facts(self, customer_id: int) -> list[SemanticFact]:
        """Return all currently active facts for one customer.

        Verified memory recall needs the active semantic set before it can
        perform the explicit relevance check required by the project.
        """

        self.expire_stale_facts()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_facts WHERE customer_id = ? "
                "AND status = 'active' ORDER BY fact_key, version DESC",
                (customer_id,),
            ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def fact_history(self, customer_id: int, fact_key: str) -> list[SemanticFact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_facts WHERE customer_id = ? "
                "AND fact_key = ? ORDER BY version ASC",
                (customer_id, fact_key),
            ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def upsert_fact(
        self,
        customer_id: int,
        fact_key: str,
        fact_value: str,
        source_episode_id: int | None = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> SemanticFact:
        """Insert, reaffirm, or version a semantic fact."""

        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=ttl_days)).isoformat()
        current = self.get_active_fact(customer_id, fact_key)

        if current is None:
            return self._insert(
                customer_id,
                fact_key,
                fact_value,
                version=1,
                source_episode_id=source_episode_id,
                conflict_reason=None,
                expires_at=expires_at,
            )

        if current.fact_value == fact_value:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE semantic_facts SET expires_at = ? WHERE id = ?",
                    (expires_at, current.id),
                )
            current.expires_at = expires_at
            return current

        conflict_reason = (
            f"Superseded version {current.version} "
            f"({current.fact_value!r} -> {fact_value!r}) "
            f"based on episode {source_episode_id}."
        )
        new_fact = self._insert(
            customer_id,
            fact_key,
            fact_value,
            version=current.version + 1,
            source_episode_id=source_episode_id,
            conflict_reason=conflict_reason,
            expires_at=expires_at,
        )

        with self._connect() as conn:
            conn.execute(
                "UPDATE semantic_facts SET status = 'superseded', "
                "superseded_by = ? WHERE id = ?",
                (new_fact.id, current.id),
            )
        return new_fact

    def expire_stale_facts(
        self,
        as_of: datetime | None = None,
    ) -> list[SemanticFact]:
        """Mark active facts past ``expires_at`` as expired, not deleted."""

        as_of = as_of or datetime.now(timezone.utc)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_facts WHERE status = 'active'"
            ).fetchall()
            expired: list[SemanticFact] = []
            for row in rows:
                fact = self._row_to_fact(row)
                if datetime.fromisoformat(fact.expires_at) < as_of:
                    conn.execute(
                        "UPDATE semantic_facts SET status = 'expired' WHERE id = ?",
                        (fact.id,),
                    )
                    fact.status = "expired"
                    expired.append(fact)
        return expired

    def _insert(
        self,
        customer_id: int,
        fact_key: str,
        fact_value: str,
        version: int,
        source_episode_id: int | None,
        conflict_reason: str | None,
        expires_at: str,
    ) -> SemanticFact:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO semantic_facts
                    (customer_id, fact_key, fact_value, version, status,
                     source_episode_id, conflict_reason, superseded_by,
                     created_at, expires_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?, NULL, ?, ?)
                """,
                (
                    customer_id,
                    fact_key,
                    fact_value,
                    version,
                    source_episode_id,
                    conflict_reason,
                    created_at,
                    expires_at,
                ),
            )
            fact_id = cur.lastrowid

        return SemanticFact(
            id=fact_id,
            customer_id=customer_id,
            fact_key=fact_key,
            fact_value=fact_value,
            version=version,
            status="active",
            source_episode_id=source_episode_id,
            conflict_reason=conflict_reason,
            superseded_by=None,
            created_at=created_at,
            expires_at=expires_at,
        )

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> SemanticFact:
        return SemanticFact(
            id=row["id"],
            customer_id=row["customer_id"],
            fact_key=row["fact_key"],
            fact_value=row["fact_value"],
            version=row["version"],
            status=row["status"],
            source_episode_id=row["source_episode_id"],
            conflict_reason=row["conflict_reason"],
            superseded_by=row["superseded_by"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
