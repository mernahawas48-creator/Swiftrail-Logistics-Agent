"""
registry.py — runtime-registered tools per agent, driven from the admin
panel, not edited by hand and redeployed.

This is intentionally the thing execute_remediation_action() checks before
calling a tool. When the admin disables 'release_credit_hold' for
'graph3_credit_hold_remediation' through the platform, the very next node
execution actually refuses to call it — the toggle reaches live execution,
it isn't cosmetic.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "swiftrail_state.db"

DEFAULT_TOOLS = {
    "graph3_credit_hold_remediation": [
        "list_customer_invoices",
        "list_customer_credit_holds",
        "release_credit_hold",
        "record_payment_evidence",
        "create_invoice_dispute",
    ],
    "graph1_delivery_exception": [
        "get_shipment_status",
        "list_reroute_options",
        "confirm_reroute",
    ],
    "graph2_rate_exception": [
        "get_rate_exception",
        "approve_rate_exception",
        "request_revised_discount",
    ],
    "memory_rag_agent": ["search_customer", "hybrid_retrieve"],
    "planning_agent": ["get_shipment_status", "list_customer_invoices", "release_credit_hold", "approve_rate_exception"],
}


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_seeded():
    with _conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_tools ("
            "agent_name TEXT NOT NULL, tool_name TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,"
            "PRIMARY KEY (agent_name, tool_name))"
        )
        row = conn.execute("SELECT COUNT(*) AS c FROM agent_tools").fetchone()
        if row["c"] == 0:
            for agent, tools in DEFAULT_TOOLS.items():
                for t in tools:
                    conn.execute(
                        "INSERT OR IGNORE INTO agent_tools (agent_name, tool_name, enabled) VALUES (?, ?, 1)",
                        (agent, t),
                    )


def is_tool_enabled(agent_name: str, tool_name: str) -> bool:
    ensure_seeded()
    with _conn() as conn:
        row = conn.execute(
            "SELECT enabled FROM agent_tools WHERE agent_name = ? AND tool_name = ?",
            (agent_name, tool_name),
        ).fetchone()
        return bool(row["enabled"]) if row else False


def list_tools(agent_name: str) -> list[dict]:
    ensure_seeded()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT tool_name, enabled FROM agent_tools WHERE agent_name = ? ORDER BY tool_name", (agent_name,)
        ).fetchall()
        return [dict(r) for r in rows]


def list_all_agents() -> list[str]:
    ensure_seeded()
    with _conn() as conn:
        rows = conn.execute("SELECT DISTINCT agent_name FROM agent_tools").fetchall()
        return [r["agent_name"] for r in rows]


def set_tool_enabled(agent_name: str, tool_name: str, enabled: bool) -> None:
    ensure_seeded()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO agent_tools (agent_name, tool_name, enabled) VALUES (?, ?, ?) "
            "ON CONFLICT(agent_name, tool_name) DO UPDATE SET enabled = excluded.enabled",
            (agent_name, tool_name, 1 if enabled else 0),
        )


def add_tool(agent_name: str, tool_name: str) -> None:
    set_tool_enabled(agent_name, tool_name, True)


def remove_tool(agent_name: str, tool_name: str) -> None:
    set_tool_enabled(agent_name, tool_name, False)


ensure_seeded()
