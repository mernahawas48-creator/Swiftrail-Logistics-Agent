from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from state_graph.core.state import SharedGraphState
from state_graph.core.types import HITLStatus, TicketStatus


class SQLiteCheckpointStore:
    """SQLite implementation for unit tests and single-process demos."""

    def __init__(self, path: str | Path = "state_graph/checkpoints.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        script = """
        CREATE TABLE IF NOT EXISTS graph_runs (
            run_id TEXT PRIMARY KEY,
            graph_name TEXT NOT NULL,
            status TEXT NOT NULL,
            current_node TEXT NOT NULL,
            revision INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS graph_checkpoints (
            checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            node TEXT NOT NULL,
            event TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, revision)
        );
        CREATE TABLE IF NOT EXISTS graph_node_executions (
            execution_key TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node TEXT NOT NULL,
            result_json TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS graph_hitl_tasks (
            task_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            request_json TEXT NOT NULL,
            state_json TEXT NOT NULL,
            decision_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT
        );
        CREATE TABLE IF NOT EXISTS graph_failure_tickets (
            ticket_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            failed_node TEXT NOT NULL,
            status TEXT NOT NULL,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            state_json TEXT NOT NULL,
            resolution_note TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            investigating_at TEXT,
            resolved_at TEXT
        );
        """
        with self._connect() as connection:
            connection.executescript(script)

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, default=str, sort_keys=True)

    @staticmethod
    def _load(value: str | None) -> Any:
        return json.loads(value) if value else None

    def _upsert_run(self, connection: sqlite3.Connection, state: SharedGraphState) -> None:
        connection.execute(
            """
            INSERT INTO graph_runs(
                run_id, graph_name, status, current_node, revision,
                state_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                current_node=excluded.current_node,
                revision=excluded.revision,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (
                state.run_id,
                state.graph_name,
                state.status.value,
                state.current_node,
                state.revision,
                self._dump(state.to_dict()),
                state.created_at,
                state.updated_at,
            ),
        )

    def create_run(self, state: SharedGraphState) -> None:
        with self._connect() as connection:
            self._upsert_run(connection, state)
            connection.execute(
                """
                INSERT INTO graph_checkpoints(
                    run_id, revision, node, event, state_json
                ) VALUES (?, ?, ?, 'run_started', ?)
                """,
                (
                    state.run_id,
                    state.revision,
                    state.current_node,
                    self._dump(state.to_dict()),
                ),
            )

    def load_run(self, run_id: str) -> SharedGraphState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM graph_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return SharedGraphState.from_dict(self._load(row[0])) if row else None

    def save_checkpoint(
        self, state: SharedGraphState, *, node: str, event: str
    ) -> int:
        with self._connect() as connection:
            self._upsert_run(connection, state)
            cursor = connection.execute(
                """
                INSERT INTO graph_checkpoints(
                    run_id, revision, node, event, state_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    state.run_id,
                    state.revision,
                    node,
                    event,
                    self._dump(state.to_dict()),
                ),
            )
            return int(cursor.lastrowid)

    def checkpoint_history(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT checkpoint_id, revision, node, event, state_json, created_at
                FROM graph_checkpoints
                WHERE run_id = ?
                ORDER BY revision
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "checkpoint_id": row["checkpoint_id"],
                "revision": row["revision"],
                "node": row["node"],
                "event": row["event"],
                "state": self._load(row["state_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_node_result(
        self,
        execution_key: str,
        *,
        run_id: str,
        node: str,
        result: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO graph_node_executions(
                    execution_key, run_id, node, result_json
                ) VALUES (?, ?, ?, ?)
                """,
                (execution_key, run_id, node, self._dump(result)),
            )

    def load_node_result(self, execution_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM graph_node_executions
                WHERE execution_key = ?
                """,
                (execution_key,),
            ).fetchone()
        return self._load(row[0]) if row else None

    def create_hitl_task(self, task: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO graph_hitl_tasks(
                    task_id, run_id, node, status, reason,
                    request_json, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["task_id"],
                    task["run_id"],
                    task["node"],
                    task["status"],
                    task["reason"],
                    self._dump(task["request"]),
                    self._dump(task["state"]),
                ),
            )

    def get_hitl_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM graph_hitl_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._hitl_row(row) if row else None

    def list_hitl_tasks(
        self, status: HITLStatus = HITLStatus.PENDING
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_hitl_tasks
                WHERE status = ? ORDER BY created_at
                """,
                (status.value,),
            ).fetchall()
        return [self._hitl_row(row) for row in rows]

    def _hitl_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "run_id": row["run_id"],
            "node": row["node"],
            "status": row["status"],
            "reason": row["reason"],
            "request": self._load(row["request_json"]),
            "state": self._load(row["state_json"]),
            "decision": self._load(row["decision_json"]),
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    def update_hitl_task(
        self,
        task_id: str,
        *,
        status: HITLStatus,
        decision: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE graph_hitl_tasks
                SET status = ?, decision_json = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND status = 'pending'
                """,
                (status.value, self._dump(decision), task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("HITL task is missing or is no longer pending.")

    def create_ticket(self, ticket: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO graph_failure_tickets(
                    ticket_id, run_id, failed_node, status, error_type,
                    error_message, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket["ticket_id"],
                    ticket["run_id"],
                    ticket["failed_node"],
                    ticket["status"],
                    ticket["error_type"],
                    ticket["error_message"],
                    self._dump(ticket["state"]),
                ),
            )

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM graph_failure_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return self._ticket_row(row) if row else None

    def list_tickets(
        self, status: TicketStatus | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM graph_failure_tickets"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._ticket_row(row) for row in rows]

    def _ticket_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "ticket_id": row["ticket_id"],
            "run_id": row["run_id"],
            "failed_node": row["failed_node"],
            "status": row["status"],
            "error_type": row["error_type"],
            "error_message": row["error_message"],
            "state": self._load(row["state_json"]),
            "resolution_note": row["resolution_note"],
            "created_at": row["created_at"],
            "investigating_at": row["investigating_at"],
            "resolved_at": row["resolved_at"],
        }

    def update_ticket(
        self,
        ticket_id: str,
        *,
        status: TicketStatus,
        resolution_note: str | None = None,
    ) -> None:
        current = self.get_ticket(ticket_id)
        if current is None:
            raise ValueError("Failure ticket was not found.")
        allowed = {
            TicketStatus.OPEN.value: TicketStatus.INVESTIGATING,
            TicketStatus.INVESTIGATING.value: TicketStatus.RESOLVED,
        }
        if allowed.get(current["status"]) != status:
            raise ValueError(
                f"Ticket cannot move from {current['status']} to {status.value}."
            )

        timestamp_column = (
            "investigating_at"
            if status is TicketStatus.INVESTIGATING
            else "resolved_at"
        )
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE graph_failure_tickets
                SET status = ?, resolution_note = ?,
                    {timestamp_column} = CURRENT_TIMESTAMP
                WHERE ticket_id = ?
                """,
                (status.value, resolution_note, ticket_id),
            )
