"""
platform/backend/app.py — the product surface described in the project
brief: a real website talking to a live MCP-style backend, not a mockup.

Run with:
    cd platform/backend && python -m uvicorn app:app --reload --port 8080

Endpoints are grouped:
  /api/agents, /api/chat, /api/runs           -> USER surface
  /api/admin/agents, /api/admin/tools         -> admin: runtime tool mgmt
  /api/admin/rag                              -> admin: RAG doc mgmt
  /api/admin/hitl                             -> admin: HITL queue
  /api/admin/tickets                          -> admin: failure tickets
All three state graphs and the existing memory/RAG and planning agents are
connected to the shared user surface. The state graphs also expose their
persisted HITL tasks and failure tickets through the admin surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from platform_app.admin.rag_service import PlatformRAGService
from platform_app.admin.runtime_client import (
    RuntimeAdminClient,
    RuntimeAdminClientError,
)
from platform_app.agent_integration import (
    MEMORY_RAG_AGENT_ID,
    PLANNING_AGENT_ID,
    PlatformAgentIntegration,
)
from platform_app.graph_integration import (
    GRAPH_1_AGENT_ID,
    GRAPH_2_AGENT_ID,
    GRAPH_3_AGENT_ID,
    PlatformGraphIntegration,
)

platform_graphs = PlatformGraphIntegration()
platform_agents = PlatformAgentIntegration()
runtime_admin = RuntimeAdminClient.from_env()
rag_admin = PlatformRAGService()

app = FastAPI(title="Swiftrail Platform API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Agent catalog (drives both the user's agent switcher and the admin's tool
# management screen). All state graphs are live; the remaining agent types
# use the same shape so the platform stays agent-agnostic.
# ---------------------------------------------------------------------------
AGENTS = [
    {"id": GRAPH_3_AGENT_ID, "name": "Credit-Hold Remediation", "kind": "state_graph",
     "description": "Resolves overdue-invoice credit holds and invoice disputes. Owner: Person 3."},
    {"id": GRAPH_1_AGENT_ID, "name": "Delivery Exception Recovery", "kind": "state_graph",
     "description": "Reroutes shipments around delivery exceptions. Owner: Person 1."},
    {"id": GRAPH_2_AGENT_ID, "name": "Rate Exception Approval", "kind": "state_graph",
     "description": "Routes discount/rate exceptions to finance approval. Owner: Person 2."},
    {"id": MEMORY_RAG_AGENT_ID, "name": "Memory & RAG Agent", "kind": "memory_rag",
     "description": "Policy questions and cross-session customer recall."},
    {"id": PLANNING_AGENT_ID, "name": "Decomposition & Planning Agent", "kind": "planning",
     "description": "Multi-blocker shipment resolution planning."},
]


class ChatRequest(BaseModel):
    agent_id: str
    message: str
    run_id: str | None = None


class ChatResponse(BaseModel):
    run_id: str | None = None
    reply: str
    status: str | None = None
    current_node: str | None = None


# ---------------------------------------------------------------------------
# USER SURFACE
# ---------------------------------------------------------------------------
@app.get("/api/agents")
def list_agents():
    return AGENTS


@app.get("/api/runs")
def list_runs(graph_name: str | None = None):
    return platform_graphs.list_runs(graph_name)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    integrated = platform_graphs.get_run(run_id)
    if integrated is not None:
        return integrated
    agent_run = platform_agents.get_run(run_id)
    if agent_run is not None:
        return agent_run
    raise HTTPException(404, "run not found")


AGENT_CHAT_HANDLERS = {
    GRAPH_3_AGENT_ID: lambda message, run_id: ChatResponse(
        **platform_graphs.chat(GRAPH_3_AGENT_ID, message, run_id)
    ),
    GRAPH_1_AGENT_ID: lambda message, run_id: ChatResponse(
        **platform_graphs.chat(GRAPH_1_AGENT_ID, message, run_id)
    ),
    GRAPH_2_AGENT_ID: lambda message, run_id: ChatResponse(
        **platform_graphs.chat(GRAPH_2_AGENT_ID, message, run_id)
    ),
    MEMORY_RAG_AGENT_ID: lambda message, run_id: ChatResponse(
        **platform_agents.chat(MEMORY_RAG_AGENT_ID, message, run_id)
    ),
    PLANNING_AGENT_ID: lambda message, run_id: ChatResponse(
        **platform_agents.chat(PLANNING_AGENT_ID, message, run_id)
    ),
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
    try:
        return runtime_admin.list_agents()
    except RuntimeAdminClientError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@app.get("/api/admin/runtime/health")
def admin_runtime_health():
    try:
        return runtime_admin.health()
    except RuntimeAdminClientError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


class ToolChange(BaseModel):
    tool_name: str
    enabled: bool


@app.post("/api/admin/agents/{agent_id}/tools")
def set_tool(agent_id: str, change: ToolChange):
    try:
        return runtime_admin.set_tool(
            agent_id,
            change.tool_name,
            change.enabled,
        )
    except RuntimeAdminClientError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


# ---------------------------------------------------------------------------
# ADMIN SURFACE — RAG documents
# ---------------------------------------------------------------------------
class RagDoc(BaseModel):
    title: str = Field(min_length=5, max_length=180)
    body: str = Field(min_length=10)
    department: str = Field(default="operations", min_length=2, max_length=80)
    document_type: str = Field(default="policy", min_length=2, max_length=80)
    access_roles: list[str] = Field(
        default_factory=lambda: ["sales_rep", "finance_manager"],
        min_length=1,
    )
    section_prefix: str = Field(default="ADM", pattern=r"^[A-Za-z]{2,5}$")


class RagUpdate(BaseModel):
    body: str = Field(min_length=10)


def _rag_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, "RAG document was not found.")
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(503, f"RAG/Qdrant update failed: {exc}")


@app.get("/api/admin/rag/documents")
def list_rag_docs():
    try:
        return rag_admin.list_documents()
    except Exception as exc:
        raise _rag_error(exc) from exc


@app.post("/api/admin/rag/documents")
def add_rag_doc(doc: RagDoc):
    try:
        return rag_admin.add_document(**doc.model_dump())
    except Exception as exc:
        raise _rag_error(exc) from exc


@app.put("/api/admin/rag/documents/{doc_id}")
def update_rag_doc(doc_id: str, update: RagUpdate):
    try:
        return rag_admin.update_document(doc_id, update.body)
    except Exception as exc:
        raise _rag_error(exc) from exc


@app.delete("/api/admin/rag/documents/{doc_id}")
def delete_rag_doc(doc_id: str):
    try:
        return rag_admin.remove_document(doc_id)
    except Exception as exc:
        raise _rag_error(exc) from exc


@app.post("/api/admin/rag/reindex")
def reindex_rag_docs():
    try:
        return rag_admin.reindex()
    except Exception as exc:
        raise _rag_error(exc) from exc


# ---------------------------------------------------------------------------
# ADMIN SURFACE — HITL queue
# ---------------------------------------------------------------------------
@app.get("/api/admin/hitl")
def list_hitl(status: str | None = None):
    tasks = platform_graphs.hitl_tasks()
    if status is not None:
        tasks = [task for task in tasks if task["status"] == status]
    return tasks


class HitlDecision(BaseModel):
    decision: str  # "approve" | "reject"
    decided_by: str = "admin"
    admin_employee_id: int = 3


@app.post("/api/admin/hitl/{task_id}/decide")
def decide_hitl(task_id: str, body: HitlDecision):
    result = platform_graphs.decide_hitl(
        task_id,
        decision=body.decision,
        decided_by=body.decided_by,
        admin_employee_id=body.admin_employee_id,
    )
    if result is None:
        raise HTTPException(404, "hitl task not found")
    return {"task_id": task_id, "run": result}


# ---------------------------------------------------------------------------
# ADMIN SURFACE — tickets
# ---------------------------------------------------------------------------
@app.get("/api/admin/tickets")
def list_tickets(status: str | None = None):
    tickets = platform_graphs.tickets()
    if status is not None:
        tickets = [ticket for ticket in tickets if ticket["status"] == status]
    return tickets


class TicketStatusChange(BaseModel):
    status: str  # open | investigating | resolved


@app.post("/api/admin/tickets/{ticket_id}/status")
def set_ticket_status(ticket_id: str, body: TicketStatusChange):
    result = platform_graphs.set_ticket_status(ticket_id, body.status)
    if result is None:
        raise HTTPException(404, "ticket not found")
    return result


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
