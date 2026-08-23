from state_graph.core.nodes import ExternalWaitNode, FunctionNode, HITLNode
from state_graph.core.transitions import GraphDefinition
from state_graph.graph_3 import nodes

GRAPH_NAME = "graph3_credit_hold_remediation"


def build_credit_hold_graph() -> GraphDefinition:
    return GraphDefinition(
        name=GRAPH_NAME,
        start_node="load_account_state",
        nodes={
            name: FunctionNode(name, function)
            for name, function in {
                "load_account_state": nodes.load_account_state,
                "build_remediation_plan": nodes.build_remediation_plan,
                "prepare_customer_wait": nodes.prepare_customer_wait,
                "process_customer_input": nodes.process_customer_input,
                "apply_admin_decision": nodes.apply_admin_decision,
                "execute_remediation_action": nodes.execute_remediation_action,
                "complete": nodes.complete,
            }.items()
        }
        | {
            "wait_for_customer": ExternalWaitNode(
                "wait_for_customer",
                "process_customer_input",
                "Waiting for customer payment confirmation or dispute evidence.",
                nodes.customer_request,
            ),
            "wait_for_finance_admin": HITLNode(
                "wait_for_finance_admin",
                "apply_admin_decision",
                nodes.admin_reason,
                nodes.admin_request,
            ),
        },
        transitions={
            "load_account_state": frozenset({"build_remediation_plan", "complete"}),
            "build_remediation_plan": frozenset({"prepare_customer_wait"}),
            "prepare_customer_wait": frozenset({"wait_for_customer"}),
            "wait_for_customer": frozenset({"process_customer_input"}),
            "process_customer_input": frozenset(
                {
                    "prepare_customer_wait",
                    "wait_for_finance_admin",
                    "execute_remediation_action",
                    "complete",
                }
            ),
            "wait_for_finance_admin": frozenset({"apply_admin_decision"}),
            "apply_admin_decision": frozenset(
                {"execute_remediation_action", "complete"}
            ),
            "execute_remediation_action": frozenset({"complete"}),
            "complete": frozenset({"END"}),
        },
    )
