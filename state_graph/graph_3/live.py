from state_graph.core.engine import GraphEngine
from state_graph.core.mysql_store import MySQLCheckpointStore
from state_graph.core.registry import GraphRegistry
from state_graph.core.service import GraphService
from state_graph.graph_3.definition import build_credit_hold_graph
from state_graph.graph_3.llm import MistralLATSRemediationPlanner
from state_graph.graph_3.react import ConstrainedCreditHoldReActPlanner
from state_graph.graph_3.tools import LiveCreditHoldTools


def build_live_service(
    *, mcp_url="http://127.0.0.1:8000/mcp", permission_checker=None
) -> GraphService:
    registry = GraphRegistry()
    registry.register(build_credit_hold_graph())
    engine = GraphEngine(
        registry,
        MySQLCheckpointStore(),
        services={
            "credit_tools": LiveCreditHoldTools(
                mcp_url, permission_checker=permission_checker
            ),
            "remediation_planner": MistralLATSRemediationPlanner(),
            "release_planner": ConstrainedCreditHoldReActPlanner(),
        },
    )
    return GraphService(engine)
