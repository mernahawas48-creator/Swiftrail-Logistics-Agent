from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mcp_server.db import get_connection
from state_graph.core.state import SharedGraphState
from state_graph.core.types import HITLStatus, TicketStatus


class MySQLCheckpointStore:
    """Production checkpoint store backed by the existing Swiftrail MySQL DB."""

    def __init__(self, connection_factory: Callable[[], Any] = get_connection) -> None:
        self.connection_factory = connection_factory

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, default=str, sort_keys=True)

    @staticmethod
    def _load(value: str | bytes | None) -> Any:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def _execute(self, operation):
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                result = operation(cursor)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _upsert_run(self, cursor, state: SharedGraphState) -> None:
        cursor.execute(
            """
            INSERT INTO graph_runs(
                run_id, graph_name, status, current_node, revision,
                state_json, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                status=VALUES(status), current_node=VALUES(current_node),
                revision=VALUES(revision), state_json=VALUES(state_json),
                updated_at=VALUES(updated_at)
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
        def operation(cursor):
            self._upsert_run(cursor, state)
            cursor.execute(
                """
                INSERT INTO graph_checkpoints(
                    run_id, revision, node, event, state_json
                ) VALUES (%s, %s, %s, 'run_started', %s)
                """,
                (
                    state.run_id,
                    state.revision,
                    state.current_node,
                    self._dump(state.to_dict()),
                ),
            )

        self._execute(operation)

    def load_run(self, run_id: str) -> SharedGraphState | None:
        def operation(cursor):
            cursor.execute(
                "SELECT state_json FROM graph_runs WHERE run_id = %s", (run_id,)
            )
            return cursor.fetchone()

        row = self._execute(operation)
        return SharedGraphState.from_dict(self._load(row["state_json"])) if row else None

    def list_runs(self, graph_name: str | None = None) -> list[SharedGraphState]:
        def operation(cursor):
            if graph_name is None:
                cursor.execute(
                    "SELECT state_json FROM graph_runs ORDER BY updated_at DESC"
                )
            else:
                cursor.execute(
                    """
                    SELECT state_json FROM graph_runs
                    WHERE graph_name = %s ORDER BY updated_at DESC
                    """,
                    (graph_name,),
                )
            return cursor.fetchall()

        return [
            SharedGraphState.from_dict(self._load(row["state_json"]))
            for row in self._execute(operation)
        ]

    def save_checkpoint(
        self, state: SharedGraphState, *, node: str, event: str
    ) -> int:
        def operation(cursor):
            self._upsert_run(cursor, state)
            cursor.execute(
                """
                INSERT INTO graph_checkpoints(
                    run_id, revision, node, event, state_json
                ) VALUES (%s, %s, %s, %s, %s)
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

        return self._execute(operation)

    def checkpoint_history(self, run_id: str) -> list[dict[str, Any]]:
        def operation(cursor):
            cursor.execute(
                """
                SELECT checkpoint_id, revision, node, event, state_json, created_at
                FROM graph_checkpoints WHERE run_id = %s ORDER BY revision
                """,
                (run_id,),
            )
            return cursor.fetchall()

        rows = self._execute(operation)
        return [
            {
                **row,
                "state": self._load(row.pop("state_json")),
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
        self._execute(
            lambda cursor: cursor.execute(
                """
                INSERT IGNORE INTO graph_node_executions(
                    execution_key, run_id, node, result_json
                ) VALUES (%s, %s, %s, %s)
                """,
                (execution_key, run_id, node, self._dump(result)),
            )
        )

    def load_node_result(self, execution_key: str) -> dict[str, Any] | None:
        def operation(cursor):
            cursor.execute(
                """
                SELECT result_json FROM graph_node_executions
                WHERE execution_key = %s
                """,
                (execution_key,),
            )
            return cursor.fetchone()

        row = self._execute(operation)
        return self._load(row["result_json"]) if row else None

    def create_hitl_task(self, task: dict[str, Any]) -> None:
        self._execute(
            lambda cursor: cursor.execute(
                """
                INSERT INTO graph_hitl_tasks(
                    task_id, run_id, node, status, reason,
                    request_json, state_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task["task_id"], task["run_id"], task["node"],
                    task["status"], task["reason"],
                    self._dump(task["request"]), self._dump(task["state"]),
                ),
            )
        )

    def _hitl_row(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["request"] = self._load(result.pop("request_json"))
        result["state"] = self._load(result.pop("state_json"))
        result["decision"] = self._load(result.pop("decision_json"))
        return result

    def get_hitl_task(self, task_id: str) -> dict[str, Any] | None:
        def operation(cursor):
            cursor.execute(
                "SELECT * FROM graph_hitl_tasks WHERE task_id = %s", (task_id,)
            )
            return cursor.fetchone()

        row = self._execute(operation)
        return self._hitl_row(row) if row else None

    def list_hitl_tasks(
        self, status: HITLStatus = HITLStatus.PENDING
    ) -> list[dict[str, Any]]:
        def operation(cursor):
            cursor.execute(
                """
                SELECT * FROM graph_hitl_tasks
                WHERE status = %s ORDER BY created_at
                """,
                (status.value,),
            )
            return cursor.fetchall()

        return [self._hitl_row(row) for row in self._execute(operation)]

    def update_hitl_task(
        self,
        task_id: str,
        *,
        status: HITLStatus,
        decision: dict[str, Any],
    ) -> None:
        def operation(cursor):
            cursor.execute(
                """
                UPDATE graph_hitl_tasks
                SET status=%s, decision_json=%s, resolved_at=CURRENT_TIMESTAMP
                WHERE task_id=%s AND status='pending'
                """,
                (status.value, self._dump(decision), task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("HITL task is missing or is no longer pending.")

        self._execute(operation)

    def create_ticket(self, ticket: dict[str, Any]) -> None:
        self._execute(
            lambda cursor: cursor.execute(
                """
                INSERT INTO graph_failure_tickets(
                    ticket_id, run_id, failed_node, status, error_type,
                    error_message, state_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    ticket["ticket_id"], ticket["run_id"], ticket["failed_node"],
                    ticket["status"], ticket["error_type"],
                    ticket["error_message"], self._dump(ticket["state"]),
                ),
            )
        )

    def _ticket_row(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["state"] = self._load(result.pop("state_json"))
        return result

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        def operation(cursor):
            cursor.execute(
                "SELECT * FROM graph_failure_tickets WHERE ticket_id = %s",
                (ticket_id,),
            )
            return cursor.fetchone()

        row = self._execute(operation)
        return self._ticket_row(row) if row else None

    def list_tickets(
        self, status: TicketStatus | None = None
    ) -> list[dict[str, Any]]:
        def operation(cursor):
            if status is None:
                cursor.execute(
                    "SELECT * FROM graph_failure_tickets ORDER BY created_at"
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM graph_failure_tickets
                    WHERE status = %s ORDER BY created_at
                    """,
                    (status.value,),
                )
            return cursor.fetchall()

        return [self._ticket_row(row) for row in self._execute(operation)]

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
        column = (
            "investigating_at"
            if status is TicketStatus.INVESTIGATING
            else "resolved_at"
        )
        self._execute(
            lambda cursor: cursor.execute(
                f"""
                UPDATE graph_failure_tickets
                SET status=%s, resolution_note=%s, {column}=CURRENT_TIMESTAMP
                WHERE ticket_id=%s
                """,
                (status.value, resolution_note, ticket_id),
            )
        )
