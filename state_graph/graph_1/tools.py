from __future__ import annotations

from typing import Any, Protocol

from agent.mcp_graph_client import GraphMCPClient
from state_graph.core.async_utils import run_async


class DeliveryRecoveryTools(Protocol):
    def load_shipment(
        self, *, session_id: str, employee_id: int, shipment_id: int
    ) -> dict[str, Any]: ...

    def create_case(
        self,
        *,
        session_id: str,
        employee_id: int,
        shipment_id: int,
        failure_reason: str,
    ) -> dict[str, Any]: ...

    def apply_reroute(
        self,
        *,
        session_id: str,
        employee_id: int,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...


class LiveDeliveryRecoveryTools:
    """Graph 1 adapter for the existing live MCP server."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000/mcp",
        *,
        agent_id: str = "graph1_delivery_exception",
        permission_checker=None,
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

    def create_case(
        self,
        *,
        session_id: str,
        employee_id: int,
        shipment_id: int,
        failure_reason: str,
    ) -> dict[str, Any]:
        return self._call(
            session_id=session_id,
            employee_id=employee_id,
            tool_name="create_delivery_recovery_case",
            request={
                "shipment_id": shipment_id,
                "failure_reason": failure_reason,
            },
        )["recovery_case"]

    def apply_reroute(
        self,
        *,
        session_id: str,
        employee_id: int,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._call(
            session_id=session_id,
            employee_id=employee_id,
            tool_name="apply_shipment_reroute",
            request=request,
        )
