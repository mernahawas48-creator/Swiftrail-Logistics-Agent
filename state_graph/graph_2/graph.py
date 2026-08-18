from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx

# The existing MCP modules use top-level imports because the MCP server is
# launched from its own directory. Add that directory here so Graph 2 reuses
# the real handlers without duplicating them.
_MCP_ROOT = Path(__file__).resolve().parents[2] / "mcp_server"
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

from .checkpoint import SQLiteCheckpointStore
from .state import RateExceptionState


AUTO_APPROVAL_LIMIT = 15.0


class GraphStatus(str, Enum):
    RUNNING = "running"
    WAITING_HITL = "waiting_hitl"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class _ElicitResult:
    action: str
    data: Any | None


class GraphDecisionContext:
    """Adapter that feeds the real MCP write tool the already-made admin decision."""

    def __init__(self, state: RateExceptionState):
        self.state = state

    async def elicit(self, *, message: str, schema: Any) -> _ElicitResult:
        if self.state.admin_decision not in {"approve", "reject"}:
            return _ElicitResult(action="decline", data=None)
        from schemas import RateExceptionDecision
        decision = RateExceptionDecision(
            approve=self.state.admin_decision == "approve",
            reviewer_note=self.state.admin_note or "Admin reviewed the rate exception in the platform.",
        )
        return _ElicitResult(action="accept", data=decision)


def _mcp_handlers():
    # Imported lazily so checkpoint/graph-structure tests do not require the
    # MCP runtime package. Production execution imports the existing handlers.
    from schemas import (
        ApproveRateExceptionInput,
        ShipmentRateExceptionInput,
        ShipmentStatusInput,
    )
    from tools.read_tools import get_shipment_rate_exception, get_shipment_status
    from tools.rate_exception import approve_rate_exception
    return (
        ApproveRateExceptionInput,
        ShipmentRateExceptionInput,
        ShipmentStatusInput,
        get_shipment_rate_exception,
        get_shipment_status,
        approve_rate_exception,
    )


class RateExceptionGraph:
    """Graph 2: stateful rate-exception approval and recovery.

    LLM additions:
      1. RAG retrieval in ``retrieve_policy``.
      2. Constrained ReAct is represented by the MCP action boundary: only the
         registered rate-exception read/write tools may be selected.

    The graph branches on discount authority, pauses for an external admin
    decision, checkpoints each transition, and resumes after a process restart.
    """

    def __init__(self, checkpoint_store: SQLiteCheckpointStore | None = None, *, live_mcp: bool = False, mcp_url: str = "http://127.0.0.1:8000/mcp"):
        self.checkpoints = checkpoint_store or SQLiteCheckpointStore()
        self.live_mcp = live_mcp
        self.mcp_url = mcp_url
        self.graph = self._build_graph()

    @staticmethod
    def _build_graph() -> nx.DiGraph:
        graph = nx.DiGraph()
        edges = [
            ("START", "load_shipment"),
            ("load_shipment", "load_rate_exception"),
            ("load_rate_exception", "retrieve_policy"),
            ("retrieve_policy", "classify_authority"),
            ("classify_authority", "auto_approve"),
            ("classify_authority", "wait_for_admin"),
            ("wait_for_admin", "apply_admin_decision"),
            ("apply_admin_decision", "complete"),
            ("auto_approve", "complete"),
            ("complete", "END"),
            # Recovery cycle: a resolved failure returns to the exact node.
            ("failure_ticket", "resume_from_checkpoint"),
            ("resume_from_checkpoint", "load_shipment"),
        ]
        graph.add_edges_from(edges)
        return graph

    def start(self, shipment_id: int, session_id: str, *, run_id: str | None = None, employee_id: int = 3) -> RateExceptionState:
        state = RateExceptionState(
            run_id=run_id or uuid.uuid4().hex,
            shipment_id=shipment_id,
            session_id=session_id,
            employee_id=employee_id,
            mcp_url=self.mcp_url,
        )
        self._checkpoint(state, "START")
        return self.run(state)

    def resume(
        self,
        run_id: str,
        *,
        admin_decision: str | None = None,
        admin_note: str | None = None,
    ) -> RateExceptionState:
        """Resume a HITL run after an admin decision."""
        data = self.checkpoints.latest(run_id)
        if data is None:
            raise ValueError(f"No checkpoint exists for run {run_id}.")
        state = RateExceptionState.from_dict(data)
        if state.current_node != "wait_for_admin":
            raise ValueError(f"Run {run_id} is not waiting for admin action.")
        if admin_decision not in {"approve", "reject"}:
            raise ValueError("admin_decision must be 'approve' or 'reject'.")
        state.admin_decision = admin_decision
        state.admin_note = admin_note
        state.current_node = "apply_admin_decision"
        if state.hitl_task_id:
            self.checkpoints.resolve_task(state.hitl_task_id)
        self._checkpoint(state, "apply_admin_decision")
        return self.run(state)

    def resolve_failure(self, run_id: str) -> RateExceptionState:
        """Resolve a failure ticket and resume from the exact failed node."""
        data = self.checkpoints.latest(run_id)
        if data is None:
            raise ValueError(f"No checkpoint exists for run {run_id}.")
        state = RateExceptionState.from_dict(data)
        if state.current_node != "failure_ticket" or not state.failed_node:
            raise ValueError(f"Run {run_id} has no open failure ticket.")
        failed_node = state.failed_node
        if state.ticket_id:
            self.checkpoints.resolve_task(state.ticket_id)
        state.ticket_status = "resolved"
        state.error = None
        state.current_node = failed_node
        self._checkpoint(state, "resume_from_checkpoint")
        return self.run(state)

    def run(self, state: RateExceptionState) -> RateExceptionState:
        while state.current_node not in {"END", "wait_for_admin", "failure_ticket"}:
            try:
                if state.current_node == "START":
                    self._transition(state, "load_shipment")
                elif state.current_node == "load_shipment":
                    if self.live_mcp:
                        result = self._live_call(state, "get_shipment_status", {"shipment_id": state.shipment_id})
                    else:
                        _, _, ShipmentStatusInput, _, get_shipment_status, _ = _mcp_handlers()
                        result = get_shipment_status(
                            ShipmentStatusInput(session_id=state.session_id, shipment_id=state.shipment_id)
                        )
                    if not result.get("success"):
                        raise RuntimeError(result.get("message", "Shipment lookup failed"))
                    state.shipment = result["data"]["shipment"]
                    self._transition(state, "load_rate_exception")
                elif state.current_node == "load_rate_exception":
                    if self.live_mcp:
                        result = self._live_call(state, "get_shipment_rate_exception", {"shipment_id": state.shipment_id})
                    else:
                        _, ShipmentRateExceptionInput, _, get_shipment_rate_exception, _, _ = _mcp_handlers()
                        result = get_shipment_rate_exception(
                            ShipmentRateExceptionInput(session_id=state.session_id, shipment_id=state.shipment_id)
                        )
                    if not result.get("success"):
                        raise RuntimeError(result.get("message", "Rate exception lookup failed"))
                    state.rate_exception = result["data"].get("rate_exception")
                    if not state.rate_exception:
                        state.final_status = "no_exception"
                        self._transition(state, "END")
                    else:
                        state.discount_pct = float(state.rate_exception["discount_pct"])
                        self._transition(state, "retrieve_policy")
                elif state.current_node == "retrieve_policy":
                    from rag.hybrid_search.search import HybridSearch
                    searcher = HybridSearch()
                    results = searcher.search(
                        "rate exception discount approval authority policy",
                        role="finance_manager",
                        top_k=3,
                        doc_ids=("rate_exception_policy",),
                    )
                    state.policy_evidence = [
                        {"chunk_id": r.chunk_id, "doc_id": r.metadata.get("doc_id"),
                         "section_id": r.metadata.get("section_id"), "text": r.text}
                        for r in results
                    ]
                    self._transition(state, "classify_authority")
                elif state.current_node == "classify_authority":
                    from .react import ConstrainedReActPlanner
                    decision = ConstrainedReActPlanner().decide(
                        shipment=state.shipment or {}, exception=state.rate_exception or {}, policy=state.policy_evidence,
                    )
                    state.requires_human = decision.decision == "human_review"
                    self._transition(state, "wait_for_admin" if state.requires_human else "auto_approve")
                    if state.requires_human:
                        state.hitl_task_id = state.hitl_task_id or f"HITL-{uuid.uuid4().hex[:10]}"
                        self.checkpoints.create_task(state.hitl_task_id, state.run_id, "hitl", state.to_dict())
                        self._checkpoint(state, "wait_for_admin")
                elif state.current_node == "auto_approve":
                    if self.live_mcp:
                        result = self._live_call(state, "approve_rate_exception", {"exception_id": int(state.rate_exception["id"])})
                    else:
                        ApproveRateExceptionInput, _, _, _, _, approve_rate_exception = _mcp_handlers()
                        result = self._call_async(
                            approve_rate_exception(
                                ApproveRateExceptionInput(session_id=state.session_id, exception_id=int(state.rate_exception["id"])),
                                GraphDecisionContext(state),
                            )
                        )
                    if not result.get("success"):
                        raise RuntimeError(result.get("message", "Rate exception approval failed"))
                    state.final_status = result["data"].get("status", "auto_approved")
                    self._transition(state, "complete")
                elif state.current_node == "apply_admin_decision":
                    if self.live_mcp:
                        result = self._live_call(
                            state,
                            "approve_rate_exception",
                            {
                                "exception_id": int(state.rate_exception["id"]),
                                "decision": {
                                    "approve": state.admin_decision == "approve",
                                    "reviewer_note": state.admin_note or "Admin reviewed the rate exception in the platform.",
                                },
                            },
                        )
                    else:
                        ApproveRateExceptionInput, _, _, _, _, approve_rate_exception = _mcp_handlers()
                        result = self._call_async(
                            approve_rate_exception(
                                ApproveRateExceptionInput(session_id=state.session_id, exception_id=int(state.rate_exception["id"])),
                                GraphDecisionContext(state),
                            )
                        )
                    if not result.get("success"):
                        raise RuntimeError(result.get("message", "Admin decision could not be applied"))
                    state.final_status = result["data"].get("status")
                    self._transition(state, "complete")
                elif state.current_node == "complete":
                    self._transition(state, "END")
                else:
                    raise RuntimeError(f"Unsupported graph node: {state.current_node}")
            except Exception as exc:
                state.error = str(exc)
                state.failed_node = state.current_node
                state.ticket_id = state.ticket_id or f"FT-{uuid.uuid4().hex[:10]}"
                state.ticket_status = "open"
                state.current_node = "failure_ticket"
                self._checkpoint(state, "failure_ticket")
                self.checkpoints.create_task(state.ticket_id, state.run_id, "failure", state.to_dict())
                return state
        return state

    def _live_call(self, state: RateExceptionState, tool_name: str, request: dict[str, Any]) -> dict[str, Any]:
        """Authenticate and call one real MCP tool in the same client session."""
        from agent.mcp_graph_client import GraphMCPClient

        async def operation():
            client = GraphMCPClient(state.mcp_url)
            try:
                auth = await client.authenticate(state.session_id, state.employee_id)
                if not auth.get("success"):
                    raise RuntimeError(auth.get("message", "MCP authentication failed"))
                return await client.call(tool_name, {"session_id": state.session_id, **request})
            finally:
                await client.close()

        return self._call_async(operation())

    @staticmethod
    def _call_async(awaitable):
        """Run an MCP coroutine from both sync scripts and active event loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        import threading
        box: list[Any] = []
        error: list[BaseException] = []
        def runner():
            try:
                box.append(asyncio.run(awaitable))
            except BaseException as exc:
                error.append(exc)
        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return box[0]

    def _transition(self, state: RateExceptionState, next_node: str) -> None:
        """Move to the next graph state and durably checkpoint the transition."""
        state.current_node = next_node
        self._checkpoint(state, next_node)

    def pending_hitl_tasks(self) -> list[dict[str, Any]]:
        return self.checkpoints.list_tasks(task_type="hitl", status="open")

    def pending_failure_tickets(self) -> list[dict[str, Any]]:
        return self.checkpoints.list_tasks(task_type="failure", status="open")

    def _checkpoint(self, state: RateExceptionState, node: str) -> None:
        self.checkpoints.save(state.run_id, node, state.to_dict())
