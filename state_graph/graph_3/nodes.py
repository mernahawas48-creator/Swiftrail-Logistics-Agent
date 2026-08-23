from __future__ import annotations

from typing import Any

from state_graph.core.nodes import NodeContext, NodeResult
from state_graph.core.state import SharedGraphState
from state_graph.core.types import NodeDirective
from state_graph.graph_3.state import CreditHoldRequest


def _request(state):
    return CreditHoldRequest.from_input(state.input_data)


def load_account_state(state: SharedGraphState, context: NodeContext) -> NodeResult:
    request = _request(state)
    invoices, holds = context.require("credit_tools").load_account(
        session_id=request.session_id,
        employee_id=request.employee_id,
        customer_id=request.customer_id,
    )
    active_hold = holds[0] if holds else None
    overdue = sum(
        float(invoice["amount"])
        for invoice in invoices
        if invoice["paid_status"] == "overdue"
    )
    if active_hold is None:
        return NodeResult("complete", {"final_status": "no_active_hold"})
    return NodeResult(
        "build_remediation_plan",
        {"invoices": invoices, "credit_hold": active_hold, "overdue_amount": overdue},
    )


def build_remediation_plan(state: SharedGraphState, context: NodeContext) -> NodeResult:
    request = _request(state)
    plan = context.require("remediation_planner").plan(
        invoices=state.data["invoices"],
        hold=state.data["credit_hold"],
        overdue_amount=float(state.data["overdue_amount"]),
        customer_claim=request.customer_claim,
    )
    return NodeResult(
        "prepare_customer_wait",
        {"plan": plan.action, "lats_plan": plan.to_dict()},
    )


def prepare_customer_wait(state: SharedGraphState, context: NodeContext) -> NodeResult:
    del context
    waiting_on = (
        "dispute_evidence"
        if state.data["plan"] == "dispute_review"
        else "payment_confirmation"
    )
    return NodeResult("wait_for_customer", {"waiting_on": waiting_on})


def customer_request(state: SharedGraphState) -> dict[str, Any]:
    return {"waiting_on": state.data["waiting_on"], "plan": state.data["plan"]}


def process_customer_input(state: SharedGraphState, context: NodeContext) -> NodeResult:
    del context
    payload = state.data["external_input"]
    if state.data["waiting_on"] == "dispute_evidence":
        evidence = str(payload.get("evidence", "")).strip()
        if len(evidence) < 15:
            return NodeResult(
                "prepare_customer_wait",
                {
                    "evidence_review": "insufficient",
                    "evidence_attempts": int(state.data.get("evidence_attempts", 0)) + 1,
                },
            )
        updates = {"customer_evidence": evidence, "evidence_review": "accepted"}
    else:
        amount = float(payload.get("amount", 0))
        if amount < float(state.data["overdue_amount"]):
            return NodeResult(
                "complete",
                {"payment_confirmed": amount, "final_status": "partial_payment_hold"},
            )
        updates = {"payment_confirmed": amount}
    return NodeResult("classify_release_action", updates)


def classify_release_action(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    lats_plan = state.data["lats_plan"]
    decision = context.require("release_planner").decide(
        hold=state.data["credit_hold"],
        overdue_amount=float(state.data["overdue_amount"]),
        plan={
            key: lats_plan[key]
            for key in ("action", "narrative", "score", "iterations")
        },
        customer_evidence=state.data.get("customer_evidence"),
        payment_confirmed=state.data.get("payment_confirmed"),
    )
    requires_human = decision.decision == "human_review"
    return NodeResult(
        "wait_for_finance_admin" if requires_human else "execute_remediation_action",
        {
            "requires_human": requires_human,
            "react_decision": {
                "decision": decision.decision,
                "rationale": decision.rationale,
                "tool": decision.tool,
            },
        },
    )


def admin_reason(state: SharedGraphState) -> str:
    return (
        f"Severe hold {state.data['credit_hold']['id']} requires finance-manager "
        "approval before release."
    )


def admin_request(state: SharedGraphState) -> dict[str, Any]:
    return {
        "credit_hold": state.data["credit_hold"],
        "overdue_amount": state.data["overdue_amount"],
        "plan": state.data["plan"],
        "lats_plan": state.data["lats_plan"],
        "react_decision": state.data["react_decision"],
    }


def apply_admin_decision(state: SharedGraphState, context: NodeContext) -> NodeResult:
    del context
    if not state.data["admin_decision"]["approved"]:
        return NodeResult("complete", {"final_status": "admin_rejected"})
    return NodeResult("execute_remediation_action")


def execute_remediation_action(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    request = _request(state)
    decision = state.data.get("admin_decision")
    employee_id = request.employee_id
    approved = None
    note = None
    if decision is not None:
        employee_id = int(decision["admin_employee_id"])
        approved = bool(decision["approved"])
        note = str(decision["note"])
    result = context.require("credit_tools").release_hold(
        session_id=request.session_id,
        employee_id=employee_id,
        hold_id=int(state.data["credit_hold"]["id"]),
        approved=approved,
        note=note,
    )
    return NodeResult("complete", {"release_result": result, "final_status": "released"})


def complete(state: SharedGraphState, context: NodeContext) -> NodeResult:
    del state, context
    return NodeResult("END", directive=NodeDirective.COMPLETE)
