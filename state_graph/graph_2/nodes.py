from __future__ import annotations

from typing import Any

from state_graph.core.nodes import NodeContext, NodeResult
from state_graph.core.state import SharedGraphState
from state_graph.core.types import NodeDirective
from state_graph.graph_2.state import RateExceptionRequest


def _request(state: SharedGraphState) -> RateExceptionRequest:
    return RateExceptionRequest.from_input(state.input_data)


def load_shipment(state: SharedGraphState, context: NodeContext) -> NodeResult:
    request = _request(state)
    shipment = context.require("rate_tools").load_shipment(
        session_id=request.session_id,
        employee_id=request.employee_id,
        shipment_id=request.shipment_id,
    )
    return NodeResult("load_rate_exception", {"shipment": shipment})


def load_rate_exception(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    request = _request(state)
    exception = context.require("rate_tools").load_rate_exception(
        session_id=request.session_id,
        employee_id=request.employee_id,
        shipment_id=request.shipment_id,
    )
    if exception is None:
        return NodeResult("complete", {"final_status": "no_exception"})
    return NodeResult(
        "retrieve_policy",
        {
            "rate_exception": exception,
            "discount_pct": float(exception["discount_pct"]),
        },
    )


def retrieve_policy(state: SharedGraphState, context: NodeContext) -> NodeResult:
    results = context.require("policy_search").search(
        "rate exception discount approval authority policy",
        role="finance_manager",
        top_k=3,
        doc_ids=("rate_exception_policy",),
    )
    evidence = [
        {
            "chunk_id": result.chunk_id,
            "doc_id": result.metadata.get("doc_id"),
            "section_id": result.metadata.get("section_id"),
            "text": result.text,
        }
        for result in results
    ]
    if not evidence:
        raise RuntimeError("No rate-exception policy evidence was retrieved.")
    analysis = context.require("policy_analyst").analyze(evidence=evidence)
    return NodeResult(
        "classify_authority",
        {
            "policy_evidence": evidence,
            "policy_analysis": analysis.to_dict(),
        },
    )


def classify_authority(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    decision = context.require("decision_planner").decide(
        shipment=state.data["shipment"],
        exception=state.data["rate_exception"],
        policy=state.data["policy_evidence"],
        analysis=state.data["policy_analysis"],
    )
    requires_human = decision.decision == "human_review"
    return NodeResult(
        "wait_for_admin" if requires_human else "apply_rate_decision",
        {
            "requires_human": requires_human,
            "planner_decision": {
                "decision": decision.decision,
                "rationale": decision.rationale,
                "tool": decision.tool,
            },
        },
    )


def admin_reason(state: SharedGraphState) -> str:
    return (
        f"Rate exception {state.data['discount_pct']}% exceeds delegated "
        "authority and requires finance-manager review."
    )


def admin_request(state: SharedGraphState) -> dict[str, Any]:
    return {
        "shipment": state.data["shipment"],
        "rate_exception": state.data["rate_exception"],
        "policy_evidence": state.data["policy_evidence"],
        "policy_analysis": state.data["policy_analysis"],
        "planner_decision": state.data["planner_decision"],
    }


def apply_rate_decision(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    request = _request(state)
    human_decision = state.data.get("admin_decision")
    approve: bool | None = None
    note: str | None = None
    employee_id = request.employee_id
    if state.data.get("requires_human"):
        if not isinstance(human_decision, dict):
            raise TypeError("Finance-manager decision is missing.")
        approve = bool(human_decision.get("approved"))
        note = str(human_decision.get("note", "")).strip()
        employee_id = int(human_decision.get("admin_employee_id") or employee_id)
    result = context.require("rate_tools").apply_decision(
        session_id=request.session_id,
        employee_id=employee_id,
        exception_id=int(state.data["rate_exception"]["id"]),
        approve=approve,
        note=note,
    )
    final_status = result.get("status")
    if final_status is None:
        final_status = (
            "auto_approved"
            if approve is None
            else "approved" if approve else "rejected"
        )
    return NodeResult("complete", {"decision_result": result, "final_status": final_status})


def complete(state: SharedGraphState, context: NodeContext) -> NodeResult:
    del state, context
    return NodeResult("END", directive=NodeDirective.COMPLETE)
