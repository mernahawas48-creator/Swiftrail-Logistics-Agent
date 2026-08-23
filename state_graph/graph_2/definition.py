from __future__ import annotations

from state_graph.core.nodes import FunctionNode, HITLNode
from state_graph.core.transitions import GraphDefinition
from state_graph.graph_2 import nodes

GRAPH_NAME = "rate_exception_approval"


def build_rate_exception_graph() -> GraphDefinition:
    return GraphDefinition(
        name=GRAPH_NAME,
        start_node="load_shipment",
        nodes={
            "load_shipment": FunctionNode("load_shipment", nodes.load_shipment),
            "load_rate_exception": FunctionNode(
                "load_rate_exception", nodes.load_rate_exception
            ),
            "retrieve_policy": FunctionNode("retrieve_policy", nodes.retrieve_policy),
            "classify_authority": FunctionNode(
                "classify_authority", nodes.classify_authority
            ),
            "wait_for_admin": HITLNode(
                "wait_for_admin",
                "apply_rate_decision",
                nodes.admin_reason,
                nodes.admin_request,
            ),
            "apply_rate_decision": FunctionNode(
                "apply_rate_decision", nodes.apply_rate_decision
            ),
            "complete": FunctionNode("complete", nodes.complete),
        },
        transitions={
            "load_shipment": frozenset({"load_rate_exception"}),
            "load_rate_exception": frozenset({"retrieve_policy", "complete"}),
            "retrieve_policy": frozenset({"classify_authority"}),
            "classify_authority": frozenset(
                {"apply_rate_decision", "wait_for_admin"}
            ),
            "wait_for_admin": frozenset({"apply_rate_decision"}),
            "apply_rate_decision": frozenset({"complete"}),
            "complete": frozenset({"END"}),
        },
    )
