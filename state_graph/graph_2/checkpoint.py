from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteCheckpointStore:
    """Durable checkpoints plus persisted HITL tasks and failure tickets."""

    def __init__(self, path: str | Path = "state_graph/checkpoints.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    node TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, sequence)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TEXT
                )
                """
            )

    def save(self, run_id: str, node: str, state: dict[str, Any]) -> int:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM graph_checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row[0])
            conn.execute(
                "INSERT INTO graph_checkpoints(run_id, sequence, node, state_json) VALUES (?, ?, ?, ?)",
                (run_id, sequence, node, json.dumps(state, default=str)),
            )
        return sequence

    def latest(self, run_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT state_json FROM graph_checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, run_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT sequence, node, state_json FROM graph_checkpoints WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return [{"sequence": seq, "node": node, "state": json.loads(raw)} for seq, node, raw in rows]

    def create_task(self, task_id: str, run_id: str, task_type: str, state: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_tasks(task_id, run_id, task_type, status, state_json) VALUES (?, ?, ?, 'open', ?)",
                (task_id, run_id, task_type, json.dumps(state, default=str)),
            )

    def list_tasks(self, task_type: str | None = None, status: str = "open") -> list[dict[str, Any]]:
        query = "SELECT task_id, run_id, task_type, status, state_json, created_at, resolved_at FROM graph_tasks WHERE status = ?"
        params: list[Any] = [status]
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY created_at"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"task_id": r[0], "run_id": r[1], "task_type": r[2], "status": r[3],
             "state": json.loads(r[4]), "created_at": r[5], "resolved_at": r[6]}
            for r in rows
        ]

    def resolve_task(self, task_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE graph_tasks SET status='resolved', resolved_at=CURRENT_TIMESTAMP WHERE task_id=?",
                (task_id,),
            )
