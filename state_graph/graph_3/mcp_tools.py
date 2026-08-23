"""
mcp_tools.py — the ONLY place Graph 3's nodes talk to "the database".

In the real repo, these five functions become thin wrappers around
agent/client.py calling the live mcp_server tools of the same name
(list_customer_invoices, list_customer_credit_holds, release_credit_hold,
record_payment_evidence, create_invoice_dispute). Keeping this as its own
module means Constrained ReAct in graph3_credit_hold.py can be pointed at
either this mock or the real MCP client without touching graph logic.

The mock store is intentionally tiny and file-backed (sqlite) so it
persists across the crash-and-resume demo the same way the real MySQL
db/ would.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "swiftrail_state.db"

ALLOWED_TOOLS = {
    "list_customer_invoices",
    "list_customer_credit_holds",
    "release_credit_hold",
    "record_payment_evidence",
    "create_invoice_dispute",
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


def _ensure_schema():
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS mock_invoices (
                invoice_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL      -- overdue | disputed | paid | corrected
            );
            CREATE TABLE IF NOT EXISTS mock_credit_holds (
                customer_id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,   -- none | standard | severe
                reason TEXT NOT NULL,
                released INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS mock_disputes (
                dispute_id TEXT PRIMARY KEY,
                invoice_id TEXT NOT NULL,
                evidence TEXT,
                status TEXT NOT NULL     -- open | insufficient | approved | rejected
            );
            """
        )


def seed_customer(customer_id: str, overdue_amount: float, severity: str, invoice_note: str = ""):
    """Used by the demo / API to stand up a fresh scenario for a customer."""
    _ensure_schema()
    with _conn() as conn:
        conn.execute("DELETE FROM mock_invoices WHERE customer_id = ?", (customer_id,))
        conn.execute("DELETE FROM mock_credit_holds WHERE customer_id = ?", (customer_id,))
        inv_id = f"INV-{customer_id}-1"
        conn.execute(
            "INSERT INTO mock_invoices (invoice_id, customer_id, amount, status) VALUES (?, ?, ?, 'overdue')",
            (inv_id, customer_id, overdue_amount),
        )
        conn.execute(
            "INSERT INTO mock_credit_holds (customer_id, severity, reason, released) VALUES (?, ?, ?, 0)",
            (customer_id, severity, invoice_note or "90+ days overdue"),
        )


# ---- the five whitelisted MCP tools ----

def list_customer_invoices(customer_id: str) -> list[dict]:
    _ensure_schema()
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM mock_invoices WHERE customer_id = ?", (customer_id,)).fetchall()
        return [dict(r) for r in rows]


def list_customer_credit_holds(customer_id: str) -> Optional[dict]:
    _ensure_schema()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM mock_credit_holds WHERE customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None


def release_credit_hold(customer_id: str, *, simulate_failure: bool = False) -> dict:
    if simulate_failure:
        raise RuntimeError("MCP tool 'release_credit_hold' timed out contacting the holds service")
    _ensure_schema()
    with _conn() as conn:
        conn.execute("UPDATE mock_credit_holds SET released = 1 WHERE customer_id = ?", (customer_id,))
        row = conn.execute("SELECT * FROM mock_credit_holds WHERE customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else {}


def record_payment_evidence(customer_id: str, invoice_id: str, amount_paid: float) -> dict:
    _ensure_schema()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM mock_invoices WHERE invoice_id = ?", (invoice_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown invoice {invoice_id}")
        remaining = row["amount"] - amount_paid
        status = "paid" if remaining <= 0 else "overdue"
        conn.execute(
            "UPDATE mock_invoices SET amount = ?, status = ? WHERE invoice_id = ?",
            (max(remaining, 0), status, invoice_id),
        )
        return {"invoice_id": invoice_id, "remaining_balance": max(remaining, 0), "status": status}


def create_invoice_dispute(invoice_id: str, evidence: str) -> dict:
    _ensure_schema()
    dispute_id = f"DSP-{invoice_id}-{uuid.uuid4().hex[:8]}"
    status = "insufficient" if len(evidence.strip()) < 15 else "open"
    with _conn() as conn:
        conn.execute(
            "INSERT INTO mock_disputes (dispute_id, invoice_id, evidence, status) VALUES (?, ?, ?, ?)",
            (dispute_id, invoice_id, evidence, status),
        )
        conn.execute("UPDATE mock_invoices SET status = 'disputed' WHERE invoice_id = ?", (invoice_id,))
    return {"dispute_id": dispute_id, "status": status}
