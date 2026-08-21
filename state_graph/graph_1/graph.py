from __future__ import annotations

from state_graph.core.nodes import ExternalWaitNode, FunctionNode, HITLNode
from state_graph.core.transitions import GraphDefinition
from state_graph.graph_1 import nodes

GRAPH_NAME = "delivery_exception_recovery"


def build_delivery_recovery_graph() -> GraphDefinition:
    graph = GraphDefinition(
        name=GRAPH_NAME,
        start_node="load_shipment",
        nodes={
            "load_shipment": FunctionNode("load_shipment", nodes.load_shipment),
            "validate_delivery_exception": FunctionNode(
                "validate_delivery_exception", nodes.validate_delivery_exception
            ),
            "decompose_recovery_plan": FunctionNode(
                "decompose_recovery_plan", nodes.decompose_recovery_plan
            ),
            "retrieve_rerouting_policy": FunctionNode(
                "retrieve_rerouting_policy", nodes.retrieve_rerouting_policy
            ),
            "create_recovery_case": FunctionNode(
                "create_recovery_case", nodes.create_recovery_case
            ),
            "generate_recovery_options": FunctionNode(
                "generate_recovery_options", nodes.generate_recovery_options
            ),
            "wait_for_customer": ExternalWaitNode(
                "wait_for_customer",
                "evaluate_customer_choice",
                "Waiting for the customer to select a delivery recovery option.",
                nodes.customer_wait_request,
            ),
            "evaluate_customer_choice": FunctionNode(
                "evaluate_customer_choice", nodes.evaluate_customer_choice
            ),
            "wait_for_admin": HITLNode(
                "wait_for_admin",
                "apply_admin_decision",
                nodes.admin_reason,
                nodes.admin_request,
            ),
            "apply_admin_decision": FunctionNode(
                "apply_admin_decision", nodes.apply_admin_decision
            ),
            "apply_reroute": FunctionNode("apply_reroute", nodes.apply_reroute),
            "verify_shipment_update": FunctionNode(
                "verify_shipment_update", nodes.verify_shipment_update
            ),
            "complete": FunctionNode("complete", nodes.complete),
        },
        transitions={
            "load_shipment": frozenset({"validate_delivery_exception"}),
            "validate_delivery_exception": frozenset({"decompose_recovery_plan"}),
            "decompose_recovery_plan": frozenset({"retrieve_rerouting_policy"}),
            "retrieve_rerouting_policy": frozenset({"create_recovery_case"}),
            "create_recovery_case": frozenset({"generate_recovery_options"}),
            "generate_recovery_options": frozenset({"wait_for_customer"}),
            "wait_for_customer": frozenset({"evaluate_customer_choice"}),
            "evaluate_customer_choice": frozenset(
                {"generate_recovery_options", "wait_for_admin", "apply_reroute"}
            ),
            "wait_for_admin": frozenset({"apply_admin_decision"}),
            "apply_admin_decision": frozenset(
                {"generate_recovery_options", "apply_reroute"}
            ),
            "apply_reroute": frozenset({"verify_shipment_update"}),
            "verify_shipment_update": frozenset({"complete"}),
            "complete": frozenset({"END"}),
        },
    )
    if not graph.has_cycle():
        raise RuntimeError("Delivery recovery graph must contain a business cycle.")
    return graph
