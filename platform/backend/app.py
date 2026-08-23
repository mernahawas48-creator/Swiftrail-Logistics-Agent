"""
platform/backend/app.py — the product surface described in the project
brief: a real website talking to a live MCP-style backend, not a mockup.

Run with:
    cd platform/backend && python -m uvicorn app:app --reload --port 8000

Endpoints are grouped:
  /api/agents, /api/chat, /api/runs           -> USER surface
  /api/admin/agents, /api/admin/tools         -> admin: runtime tool mgmt
  /api/admin/rag                              -> admin: RAG doc mgmt
  /api/admin/hitl                             -> admin: HITL queue
  /api/admin/tickets                          -> admin: failure tickets
  /api/demo/seed                              -> convenience for the demo

Only Graph 3 is a real, live state-graph agent in this codebase (that is
Person 3's scope). The memory/RAG and planning agents, and Graphs 1/2, are
represented here as real rows in the same tables (agents, tools, runs) so
the SAME admin/user surface manages them — wiring each one's chat handler
to the real agent/agent_loop.py and state_graph/graph1_*.py /
graph2_*.py modules is a one-function change per agent (see
`AGENT_CHAT_HANDLERS` below), not a platform rewrite.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from state_graph.graph_3 import mcp_tools, registry
from state_graph.graph_3.checkpointer import Checkpointer
from state_graph.graph_3.graph3_credit_hold import GRAPH_NAME as G3_NAME
from state_graph.graph_3.graph3_credit_hold import graph as g3

cp = Checkpointer()

app = FastAPI(title="Swiftrail Platform API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Agent catalog (drives both the user's agent switcher and the admin's tool
# management screen). Graph 3 is live; the rest are declared here with the
# same shape so the platform is genuinely agent-agnostic.
# ---------------------------------------------------------------------------
AGENTS = [
    {"id": G3_NAME, "name": "Credit-Hold Remediation", "kind": "state_graph",
     "description": "Resolves overdue-invoice credit holds and invoice disputes. Owner: Person 3."},
    {"id": "graph1_delivery_exception", "name": "Delivery Exception Recovery", "kind": "state_graph",
     "description": "Reroutes shipments around delivery exceptions. Owner: Person 1."},
    {"id": "graph2_rate_exception", "name": "Rate Exception Approval", "kind": "state_graph",
     "description": "Routes discount/rate exceptions to finance approval. Owner: Person 2."},
    {"id": "memory_rag_agent", "name": "Memory & RAG Agent", "kind": "memory_rag",
     "description": "Policy questions and cross-session customer recall."},
    {"id": "planning_agent", "name": "Decomposition & Planning Agent", "kind": "planning",
     "description": "Multi-blocker shipment resolution planning."},
]


class ChatRequest(BaseModel):
    agent_id: str
    message: str
    run_id: Optional[str] = None


class ChatResponse(BaseModel):
    run_id: Optional[str] = None
    reply: str
    status: Optional[str] = None
    current_node: Optional[str] = None


# ---------------------------------------------------------------------------
# USER SURFACE
# ---------------------------------------------------------------------------
@app.get("/api/agents")
def list_agents():
    return AGENTS


@app.get("/api/runs")
def list_runs(graph_name: Optional[str] = None):
    return cp.list_runs(graph_name)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = cp.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    history = cp.history(run_id)
    latest = cp.latest_checkpoint(run_id)
    return {"run": run, "history": history, "state": latest["state"] if latest else {}}


_START_RE = re.compile(
    r"start\s+customer\s+(?P<cust>[^\s,]+)(?:\s*,?\s*claim:\s*(?P<claim>.+))?", re.IGNORECASE
)


def _handle_graph3_chat(message: str, run_id: Optional[str]) -> ChatResponse:
    m = _START_RE.match(message.strip())
    if m:
        customer_id = m.group("cust")
        claim = m.group("claim")
        holds = mcp_tools.list_customer_credit_holds(customer_id)
        if holds is None:
            return ChatResponse(reply=f"No account found for customer {customer_id}. "
                                       f"Try 'seed customer {customer_id} amount 5000 severity severe' first.")
        new_run_id = g3.start("load_account_state", {"customer_id": customer_id, "customer_claim": claim, "log": []})
        run = cp.get_run(new_run_id)
        return ChatResponse(
            run_id=new_run_id, status=run["status"], current_node=run["current_node"],
            reply=_status_reply(new_run_id),
        )

    if not run_id:
        return ChatResponse(reply=(
            "Tell me which customer to work, e.g. "
            "'start customer 3, claim: one invoice is incorrect'."
        ))

    run = cp.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")

    if run["status"] == "paused_hitl":
        return ChatResponse(run_id=run_id, status=run["status"], current_node=run["current_node"], reply=(
            "This run is waiting on a finance-admin decision — it can only continue "
            "once an admin acts on it from the Admin Dashboard."
        ))
    if run["status"] == "ticketed":
        return ChatResponse(run_id=run_id, status=run["status"], current_node=run["current_node"], reply=(
            "This run hit an unexpected failure and is now a ticket — an admin needs to "
            "resolve it from the Admin Dashboard before it can continue."
        ))
    if run["status"] == "completed":
        return ChatResponse(run_id=run_id, status=run["status"], current_node=run["current_node"],
                             reply="This run is already complete. Start a new customer to begin another.")

    # paused_wait: the free-text message IS the customer's action.
    latest = cp.latest_checkpoint(run_id)
    waiting_on = latest["state"].get("_waiting_on")
    extra = {}
    if waiting_on == "dispute_evidence":
        extra["customer_evidence"] = message
    elif waiting_on == "payment_confirmation":
        amt_match = re.search(r"[\d.,]+", message.replace(",", ""))
        amt = float(amt_match.group()) if amt_match else latest["state"].get("overdue_amount", 0)
        extra["payment_confirmed"] = amt
    else:
        return ChatResponse(run_id=run_id, status=run["status"], current_node=run["current_node"],
                             reply="I'm not currently waiting on anything from you for this run.")

    g3.resume(run_id, extra_state=extra)
    return ChatResponse(run_id=run_id, status=cp.get_run(run_id)["status"],
                         current_node=cp.get_run(run_id)["current_node"], reply=_status_reply(run_id))


def _status_reply(run_id: str) -> str:
    run = cp.get_run(run_id)
    latest = cp.latest_checkpoint(run_id)
    log_tail = latest["state"].get("log", [])[-2:]
    tail = " ".join(log_tail)
    if run["status"] == "paused_wait":
        waiting_on = latest["state"].get("_waiting_on")
        if waiting_on == "dispute_evidence":
            return f"{tail} Waiting on you: please submit dispute evidence for this invoice."
        return f"{tail} Waiting on you: please confirm payment (amount)."
    if run["status"] == "paused_hitl":
        return f"{tail} This now needs finance-admin sign-off — an admin has been notified."
    if run["status"] == "ticketed":
        return f"{tail} Something went wrong on our end — a ticket has been opened for an admin."
    if run["status"] == "completed":
        return f"{tail} All done."
    return tail or "Working on it..."


AGENT_CHAT_HANDLERS = {
    G3_NAME: _handle_graph3_chat,
    # "graph1_delivery_exception": _handle_graph1_chat,   # Person 1 wires this in
    # "graph2_rate_exception": _handle_graph2_chat,        # Person 2 wires this in
}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    handler = AGENT_CHAT_HANDLERS.get(req.agent_id)
    if handler:
        return handler(req.message, req.run_id)

    agent = next((a for a in AGENTS if a["id"] == req.agent_id), None)
    if not agent:
        raise HTTPException(404, "unknown agent")
    return ChatResponse(reply=(
        f"'{agent['name']}' isn't wired to a live backend in this checkout of the repo yet — "
        f"its owner connects it the same way Graph 3 is connected here (see AGENT_CHAT_HANDLERS "
        f"in platform/backend/app.py)."
    ))


# ---------------------------------------------------------------------------
# ADMIN SURFACE — tools per agent (runtime registry)
# ---------------------------------------------------------------------------
@app.get("/api/admin/agents")
def admin_agents():
    out = []
    for a in AGENTS:
        out.append({**a, "tools": registry.list_tools(a["id"])})
    return out


class ToolChange(BaseModel):
    tool_name: str
    enabled: bool


@app.post("/api/admin/agents/{agent_id}/tools")
def set_tool(agent_id: str, change: ToolChange):
    registry.set_tool_enabled(agent_id, change.tool_name, change.enabled)
    return {"agent_id": agent_id, "tools": registry.list_tools(agent_id)}


# ---------------------------------------------------------------------------
# ADMIN SURFACE — RAG documents
# ---------------------------------------------------------------------------
class RagDoc(BaseModel):
    title: str
    body: str


@app.get("/api/admin/rag/documents")
def list_rag_docs():
    import sqlite3
    conn = sqlite3.connect(mcp_tools.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM rag_documents ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/admin/rag/documents")
def add_rag_doc(doc: RagDoc):
    import sqlite3
    import time
    import uuid
    doc_id = str(uuid.uuid4())
    conn = sqlite3.connect(mcp_tools.DB_PATH)
    conn.execute(
        "INSERT INTO rag_documents (doc_id, title, body, added_at) VALUES (?, ?, ?, ?)",
        (doc_id, doc.title, doc.body, time.time()),
    )
    conn.commit()
    conn.close()
    return {"doc_id": doc_id}


@app.delete("/api/admin/rag/documents/{doc_id}")
def delete_rag_doc(doc_id: str):
    import sqlite3
    conn = sqlite3.connect(mcp_tools.DB_PATH)
    conn.execute("DELETE FROM rag_documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return {"deleted": doc_id}


# ---------------------------------------------------------------------------
# ADMIN SURFACE — HITL queue
# ---------------------------------------------------------------------------
@app.get("/api/admin/hitl")
def list_hitl(status: Optional[str] = None):
    tasks = cp.list_hitl_tasks(status)
    for t in tasks:
        run = cp.get_run(t["run_id"])
        t["graph_status"] = run["status"] if run else None
    return tasks


class HitlDecision(BaseModel):
    decision: str  # "approve" | "reject"
    decided_by: str = "admin"


@app.post("/api/admin/hitl/{task_id}/decide")
def decide_hitl(task_id: str, body: HitlDecision):
    tasks = {t["task_id"]: t for t in cp.list_hitl_tasks()}
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "hitl task not found")
    cp.decide_hitl_task(task_id, body.decision, body.decided_by)
    g3.resume(task["run_id"], extra_state={"finance_decision": body.decision})
    return {"task_id": task_id, "run": cp.get_run(task["run_id"])}


# ---------------------------------------------------------------------------
# ADMIN SURFACE — tickets
# ---------------------------------------------------------------------------
@app.get("/api/admin/tickets")
def list_tickets(status: Optional[str] = None):
    tickets = cp.list_tickets(status)
    for t in tickets:
        run = cp.get_run(t["run_id"])
        t["graph_status"] = run["status"] if run else None
    return tickets


class TicketStatusChange(BaseModel):
    status: str  # open | investigating | resolved


@app.post("/api/admin/tickets/{ticket_id}/status")
def set_ticket_status(ticket_id: str, body: TicketStatusChange):
    tickets = {t["ticket_id"]: t for t in cp.list_tickets()}
    ticket = tickets.get(ticket_id)
    if not ticket:
        raise HTTPException(404, "ticket not found")
    updated = cp.set_ticket_status(ticket_id, body.status)
    if body.status == "resolved":
        g3.resume(ticket["run_id"])
    return {"ticket": updated, "run": cp.get_run(ticket["run_id"])}


# ---------------------------------------------------------------------------
# Demo convenience
# ---------------------------------------------------------------------------
class SeedRequest(BaseModel):
    customer_id: str
    amount: float
    severity: str = "severe"


@app.post("/api/demo/seed")
def seed(req: SeedRequest):
    mcp_tools.seed_customer(req.customer_id, req.amount, req.severity)
    return {"seeded": req.customer_id}


@app.post("/api/demo/force-tool-failure/{run_id}")
def force_tool_failure(run_id: str):
    """Test-only: flags the next execute_remediation_action call on this run
    to fail, so the ticket path can be exercised on demand."""
    latest = cp.latest_checkpoint(run_id)
    if not latest:
        raise HTTPException(404, "run not found")
    state = latest["state"]
    state["_force_tool_failure"] = True
    cp.save(run_id, latest["node_name"], state, status=cp.get_run(run_id)["status"])
    return {"flagged": run_id}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
