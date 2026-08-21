from __future__ import annotations

from typing import Any

from state_graph.core.nodes import NodeContext, NodeResult
from state_graph.core.state import SharedGraphState
from state_graph.core.types import NodeDirective
from state_graph.graph_1.state import DeliveryRecoveryRequest


def _request(state: SharedGraphState) -> DeliveryRecoveryRequest:
    return DeliveryRecoveryRequest.from_input(state.input_data)


def load_shipment(state: SharedGraphState, context: NodeContext) -> NodeResult:
    request = _request(state)
    shipment = context.require("delivery_tools").load_shipment(
        session_id=request.session_id,
        employee_id=request.employee_id,
        shipment_id=request.shipment_id,
    )
    return NodeResult("validate_delivery_exception", {"shipment": shipment})


def validate_delivery_exception(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    del context
    shipment = state.data["shipment"]
    if shipment.get("status") != "delivery_exception":
        raise RuntimeError(
            "Shipment is not in the delivery_exception state required by Graph 1."
        )
    return NodeResult("decompose_recovery_plan")


def decompose_recovery_plan(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    request = _request(state)
    plan = context.require("task_decomposer").decompose(
        shipment=state.data["shipment"],
        failure_reason=request.failure_reason,
    )
    return NodeResult("retrieve_rerouting_policy", {"recovery_plan": plan.to_dict()})


def retrieve_rerouting_policy(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    response = context.require("policy_rag").answer(
        state.data["recovery_plan"]["policy_query"],
        role="sales_rep",
        top_k=4,
        doc_ids=("delivery_exception_policy",),
    )
    verification = getattr(response, "verification", None)
    if verification is None or not verification.passed:
        reason = getattr(verification, "reason", "RAG verification was unavailable.")
        raise RuntimeError(f"Delivery policy evidence failed verification: {reason}")
    sources = [
        {
            "doc_id": source.doc_id,
            "section_id": source.section_id,
            "number": source.number,
        }
        for source in response.sources
    ]
    return NodeResult(
        "create_recovery_case",
        {
            "policy_answer": response.answer,
            "policy_sources": sources,
        },
    )


def create_recovery_case(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    request = _request(state)
    recovery_case = context.require("delivery_tools").create_case(
        session_id=request.session_id,
        employee_id=request.employee_id,
        shipment_id=request.shipment_id,
        failure_reason=request.failure_reason,
    )
    return NodeResult(
        "generate_recovery_options",
        {"recovery_case": recovery_case, "option_round": 0},
    )


def generate_recovery_options(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    del context
    option_round = int(state.data.get("option_round", 0)) + 1
    shipment = state.data["shipment"]
    options = [
        {
            "action": "redeliver",
            "label": "Retry delivery to the verified destination",
            "destination": shipment["destination"],
            "destination_verified": True,
            "estimated_cost": 0.0,
        },
        {
            "action": "reroute",
            "label": "Reroute to a customer-provided destination",
            "destination_verified": False,
            "estimated_cost": None,
        },
    ]
    return NodeResult(
        "wait_for_customer",
        {"recovery_options": options, "option_round": option_round},
    )


def customer_wait_request(state: SharedGraphState) -> dict[str, Any]:
    return {
        "question": state.data["recovery_plan"]["customer_question"],
        "options": state.data["recovery_options"],
        "option_round": state.data["option_round"],
    }


def evaluate_customer_choice(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    del context
    choice = state.data.get("external_input")
    if not isinstance(choice, dict):
        raise TypeError("Customer choice is missing or invalid.")
    action = choice.get("action")
    if action == "request_new_options":
        return NodeResult(
            "generate_recovery_options",
            {"customer_choice": choice},
        )
    if action not in {"redeliver", "reroute"}:
        raise RuntimeError("Customer selected an unsupported recovery action.")

    shipment = state.data["shipment"]
    normalized = {
        "action": action,
        "new_destination": str(
            choice.get("new_destination") or shipment["destination"]
        ).strip(),
        "destination_verified": bool(
            choice.get("destination_verified", action == "redeliver")
        ),
        "estimated_cost": float(choice.get("estimated_cost") or 0),
        "customs_change": bool(choice.get("customs_change", False)),
        "high_value": bool(choice.get("high_value", False)),
    }
    if len(normalized["new_destination"]) < 3:
        raise RuntimeError("Customer destination is invalid.")
    requires_admin = (
        not normalized["destination_verified"]
        or normalized["estimated_cost"] > 500
        or normalized["customs_change"]
        or normalized["high_value"]
    )
    return NodeResult(
        "wait_for_admin" if requires_admin else "apply_reroute",
        {
            "customer_choice": normalized,
            "requires_admin": requires_admin,
        },
    )


def admin_reason(state: SharedGraphState) -> str:
    choice = state.data["customer_choice"]
    reasons = []
    if not choice["destination_verified"]:
        reasons.append("destination is not verified")
    if choice["estimated_cost"] > 500:
        reasons.append("reroute cost exceeds $500")
    if choice["customs_change"]:
        reasons.append("reroute changes the customs region")
    if choice["high_value"]:
        reasons.append("shipment is high-value")
    return "Admin review required because " + ", ".join(reasons) + "."


def admin_request(state: SharedGraphState) -> dict[str, Any]:
    return {
        "shipment": state.data["shipment"],
        "customer_choice": state.data["customer_choice"],
        "policy_sources": state.data["policy_sources"],
    }


def apply_admin_decision(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    del context
    decision = state.data.get("admin_decision")
    if not isinstance(decision, dict):
        raise TypeError("Admin decision is missing.")
    if decision.get("approved") is True:
        return NodeResult("apply_reroute")
    return NodeResult(
        "generate_recovery_options",
        {"last_rejection_note": decision.get("note", "Admin rejected option.")},
    )


def apply_reroute(state: SharedGraphState, context: NodeContext) -> NodeResult:
    request = _request(state)
    choice = state.data["customer_choice"]
    decision = state.data.get("admin_decision") or {}
    employee_id = int(decision.get("admin_employee_id") or request.employee_id)
    authorization = None
    if state.data.get("requires_admin"):
        authorization = {
            "admin_employee_id": employee_id,
            "authorization_note": decision.get("note", ""),
        }
    result = context.require("delivery_tools").apply_reroute(
        session_id=request.session_id,
        employee_id=employee_id,
        request={
            "case_id": int(state.data["recovery_case"]["id"]),
            "new_destination": choice["new_destination"],
            "estimated_cost": choice["estimated_cost"],
            "destination_verified": choice["destination_verified"],
            "customs_change": choice["customs_change"],
            "high_value": choice["high_value"],
            "idempotency_key": f"{state.run_id}:apply_reroute",
            "authorization": authorization,
        },
    )
    return NodeResult("verify_shipment_update", {"reroute_result": result})


def verify_shipment_update(
    state: SharedGraphState, context: NodeContext
) -> NodeResult:
    request = _request(state)
    shipment = context.require("delivery_tools").load_shipment(
        session_id=request.session_id,
        employee_id=request.employee_id,
        shipment_id=request.shipment_id,
    )
    expected = state.data["customer_choice"]["new_destination"]
    if shipment.get("destination") != expected:
        raise RuntimeError("Shipment destination did not reflect the approved reroute.")
    return NodeResult("complete", {"shipment": shipment})


def complete(state: SharedGraphState, context: NodeContext) -> NodeResult:
    del state, context
    return NodeResult("END", {"final_status": "resolved"}, NodeDirective.COMPLETE)
