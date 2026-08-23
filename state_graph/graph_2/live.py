from __future__ import annotations

from typing import Any

from agent.mcp_graph_client import GraphMCPClient
from rag.hybrid_search.search import HybridSearch
from state_graph.core.async_utils import run_async
from state_graph.core.engine import GraphEngine
from state_graph.core.mysql_store import MySQLCheckpointStore
from state_graph.core.registry import GraphRegistry
from state_graph.core.service import GraphService
from state_graph.graph_2.definition import build_rate_exception_graph
from state_graph.graph_2.llm import MistralPolicyAnalyst
from state_graph.graph_2.react import ConstrainedReActPlanner


class LiveRateExceptionTools:
    """Synchronous Graph 2 adapter around the existing live MCP client."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000/mcp",
        *,
        agent_id: str = "graph2_rate_exception",
        permission_checker: Any | None = None,
    ) -> None:
        self.url = url
        self.agent_id = agent_id
        self.permission_checker = permission_checker

    def _call(
        self,
        *,
        session_id: str,
        employee_id: int,
        tool_name: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            client = GraphMCPClient(
                self.url,
                agent_id=self.agent_id,
                permission_checker=self.permission_checker,
            )
            try:
                authentication = await client.authenticate(session_id, employee_id)
                if authentication.get("success") is not True:
                    raise RuntimeError(
                        authentication.get("message", "MCP authentication failed")
                    )
                response = await client.call(
                    tool_name,
                    {"session_id": session_id, **request},
                )
                if response.get("success") is not True:
                    raise RuntimeError(
                        response.get("message", f"MCP tool {tool_name} failed")
                    )
                return response["data"]
            finally:
                await client.close()

        return run_async(operation())

    def load_shipment(
        self, *, session_id: str, employee_id: int, shipment_id: int
    ) -> dict[str, Any]:
        return self._call(
            session_id=session_id,
            employee_id=employee_id,
            tool_name="get_shipment_status",
            request={"shipment_id": shipment_id},
        )["shipment"]

    def load_rate_exception(
        self, *, session_id: str, employee_id: int, shipment_id: int
    ) -> dict[str, Any] | None:
        return self._call(
            session_id=session_id,
            employee_id=employee_id,
            tool_name="get_shipment_rate_exception",
            request={"shipment_id": shipment_id},
        ).get("rate_exception")

    def apply_decision(
        self,
        *,
        session_id: str,
        employee_id: int,
        exception_id: int,
        approve: bool | None,
        note: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"exception_id": exception_id}
        if approve is not None:
            request["decision"] = {
                "approve": approve,
                "reviewer_note": note,
            }
        return self._call(
            session_id=session_id,
            employee_id=employee_id,
            tool_name="approve_rate_exception",
            request=request,
        )


def build_live_service(
    *,
    mcp_url: str = "http://127.0.0.1:8000/mcp",
    permission_checker: Any | None = None,
) -> GraphService:
    """Build Graph 2 with the shared production MySQL state runtime."""
    registry = GraphRegistry()
    registry.register(build_rate_exception_graph())
    engine = GraphEngine(
        registry,
        MySQLCheckpointStore(),
        services={
            "rate_tools": LiveRateExceptionTools(
                mcp_url,
                permission_checker=permission_checker,
            ),
            "policy_search": HybridSearch(),
            "policy_analyst": MistralPolicyAnalyst(),
            "decision_planner": ConstrainedReActPlanner(),
        },
    )
    return GraphService(engine)
