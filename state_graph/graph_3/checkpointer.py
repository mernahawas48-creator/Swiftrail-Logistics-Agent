"""
checkpointer.py — durable, crash-survivable state storage.

This is the ONE place state ever gets written. Every meaningful transition
in engine.py calls Checkpointer.save() before the engine moves on, so a
`kill -9` between two nodes can never lose more than the node currently
executing (which hasn't committed yet anyway).

Swapping this for MySQL later is a one-file change: replace the sqlite3
calls with your db/ connection pool and keep the same method signatures.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "swiftrail_state.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    graph_name TEXT NOT NULL,
    status TEXT NOT NULL,           -- running | paused_hitl | paused_wait | ticketed | completed
    current_node TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    seq INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS hitl_tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    options_json TEXT NOT NULL,
    status TEXT NOT NULL,           -- pending | approved | rejected
    decision TEXT,
    decided_by TEXT,
    created_at REAL NOT NULL,
    decided_at REAL
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    status TEXT NOT NULL,           -- open | investigating | resolved
    created_at REAL NOT NULL,
    resolved_at REAL
);

CREATE TABLE IF NOT EXISTS agent_tools (
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (agent_name, tool_name)
);

CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    added_at REAL NOT NULL
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript(SCHEMA)


class Checkpointer:
    """Every write here is a committed sqlite transaction, so a process
    killed the instant after save() returns has already durably persisted
    that transition — that's what makes crash-and-resume real rather than
    a resume-within-the-same-process trick."""

    def start_run(self, run_id: str, graph_name: str, first_node: str, initial_state: dict) -> None:
        now = time.time()
        with _conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, graph_name, status, current_node, created_at, updated_at) "
                "VALUES (?, ?, 'running', ?, ?, ?)",
                (run_id, graph_name, first_node, now, now),
            )
            conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, run_id, node_name, state_json, created_at, seq) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (str(uuid.uuid4()), run_id, first_node, json.dumps(initial_state), now),
            )

    def save(self, run_id: str, node_name: str, state: dict, status: str = "running") -> None:
        now = time.time()
        with _conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            next_seq = row["next_seq"]
            conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, run_id, node_name, state_json, created_at, seq) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), run_id, node_name, json.dumps(state), now, next_seq),
            )
            conn.execute(
                "UPDATE runs SET status = ?, current_node = ?, updated_at = ? WHERE run_id = ?",
                (status, node_name, now, run_id),
            )

    def latest_checkpoint(self, run_id: str) -> dict | None:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "node_name": row["node_name"],
                "state": json.loads(row["state_json"]),
                "seq": row["seq"],
            }

    def history(self, run_id: str) -> list[dict]:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT node_name, state_json, created_at, seq FROM checkpoints "
                "WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
            return [
                {"node_name": r["node_name"], "state": json.loads(r["state_json"]),
                 "created_at": r["created_at"], "seq": r["seq"]}
                for r in rows
            ]

    def get_run(self, run_id: str) -> dict | None:
        with _conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self, graph_name: str | None = None) -> list[dict]:
        with _conn() as conn:
            if graph_name:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE graph_name = ? ORDER BY updated_at DESC", (graph_name,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM runs ORDER BY updated_at DESC").fetchall()
            return [dict(r) for r in rows]

    # ---- HITL ----
    def create_hitl_task(self, run_id: str, node_name: str, reason: str, options: list[str]) -> str:
        task_id = str(uuid.uuid4())
        with _conn() as conn:
            conn.execute(
                "INSERT INTO hitl_tasks (task_id, run_id, node_name, reason, options_json, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (task_id, run_id, node_name, reason, json.dumps(options), time.time()),
            )
        return task_id

    def decide_hitl_task(self, task_id: str, decision: str, decided_by: str) -> dict:
        with _conn() as conn:
            conn.execute(
                "UPDATE hitl_tasks SET status = ?, decision = ?, decided_by = ?, decided_at = ? WHERE task_id = ?",
                ("approved" if decision == "approve" else "rejected", decision, decided_by, time.time(), task_id),
            )
            row = conn.execute("SELECT * FROM hitl_tasks WHERE task_id = ?", (task_id,)).fetchone()
            return dict(row)

    def list_hitl_tasks(self, status: str | None = None) -> list[dict]:
        with _conn() as conn:
            if status:
                rows = conn.execute("SELECT * FROM hitl_tasks WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM hitl_tasks ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # ---- Tickets ----
    def create_ticket(self, run_id: str, node_name: str, error_type: str, error_message: str) -> str:
        ticket_id = str(uuid.uuid4())
        with _conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_id, run_id, node_name, error_type, error_message, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'open', ?)",
                (ticket_id, run_id, node_name, error_type, error_message, time.time()),
            )
        return ticket_id

    def set_ticket_status(self, ticket_id: str, status: str) -> dict:
        with _conn() as conn:
            resolved_at = time.time() if status == "resolved" else None
            conn.execute(
                "UPDATE tickets SET status = ?, resolved_at = COALESCE(?, resolved_at) WHERE ticket_id = ?",
                (status, resolved_at, ticket_id),
            )
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            return dict(row)

    def list_tickets(self, status: str | None = None) -> list[dict]:
        with _conn() as conn:
            if status:
                rows = conn.execute("SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]


init_db()
