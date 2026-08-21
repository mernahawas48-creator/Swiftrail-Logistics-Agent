from __future__ import annotations

import asyncio
import threading
from typing import Any, Protocol

from agent.mcp_graph_client import GraphMCPClient


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


def _run_async(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: list[Any] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


class LiveDeliveryRecoveryTools:
    """Graph 1 adapter for the existing live MCP server."""

    def __init__(self, url: str = "http://127.0.0.1:8000/mcp") -> None:
        self.url = url

    def _call(
        self,
        *,
        session_id: str,
        employee_id: int,
        tool_name: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            client = GraphMCPClient(self.url)
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

        return _run_async(operation())

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
