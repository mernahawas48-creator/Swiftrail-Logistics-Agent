from __future__ import annotations

from rag.hybrid_rag.pipeline import HybridRAG
from state_graph.core.engine import GraphEngine
from state_graph.core.mysql_store import MySQLCheckpointStore
from state_graph.core.registry import GraphRegistry
from state_graph.core.service import GraphService
from state_graph.graph_1.graph import build_delivery_recovery_graph
from state_graph.graph_1.llm import MistralRecoveryDecomposer
from state_graph.graph_1.tools import LiveDeliveryRecoveryTools


def build_live_service(
    *, mcp_url: str = "http://127.0.0.1:8000/mcp"
) -> GraphService:
    registry = GraphRegistry()
    registry.register(build_delivery_recovery_graph())
    engine = GraphEngine(
        registry,
        MySQLCheckpointStore(),
        services={
            "delivery_tools": LiveDeliveryRecoveryTools(mcp_url),
            "task_decomposer": MistralRecoveryDecomposer(),
            "policy_rag": HybridRAG(),
        },
    )
    return GraphService(engine)
