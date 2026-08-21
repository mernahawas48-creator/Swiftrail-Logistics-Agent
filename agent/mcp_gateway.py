"""Synchronous gateway from AgentLoop to the real Swiftrail MCP server."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol

ID_PATTERN = re.compile(r"(?:#|\b)(\d+)\b")


class MCPGateway(Protocol):
    """Small injectable contract used by the synchronous AgentLoop."""

    def call(
        self,
        destination: str,
        query: str,
        *,
        session_id: str,
        customer_id: int | None,
    ) -> dict[str, Any]:
        ...


class MCPGatewayError(RuntimeError):
    """Raised when a request cannot be mapped to or completed by MCP."""


@dataclass(slots=True)
class StdioMCPGateway:
    """Authenticate and execute one real MCP read tool over stdio.

    Each call owns one short-lived connection. This keeps the original
    synchronous AgentLoop API usable while still exercising the real MCP
    protocol and server-side authorization. The future async web platform can
    reuse ``acall`` and keep a longer-lived client per user session.
    """

    employee_id: int
    transport: str = "stdio"
    http_url: str | None = None

    def __post_init__(self) -> None:
        if self.employee_id < 1:
            raise ValueError("employee_id must be positive.")

    def call(
        self,
        destination: str,
        query: str,
        *,
        session_id: str,
        customer_id: int | None,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acall(
                    destination,
                    query,
                    session_id=session_id,
                    customer_id=customer_id,
                )
            )
        raise MCPGatewayError(
            "StdioMCPGateway.call cannot run inside an active event loop; "
            "use await StdioMCPGateway.acall from async code."
        )

    async def acall(
        self,
        destination: str,
        query: str,
        *,
        session_id: str,
        customer_id: int | None,
    ) -> dict[str, Any]:
        tool_name, request = self._build_request(
            destination,
            query,
            session_id=session_id,
            customer_id=customer_id,
        )
        # Lazy import keeps AgentLoop unit tests independent from the MCP
        # transport and Windows pywin32 runtime.
        from agent.client import SwiftrailAgent

        client = SwiftrailAgent(self.transport, self.http_url)
        try:
            await client.connect()
            auth_result = await client.call_tool(
                "authenticate",
                {
                    "request": {
                        "session_id": session_id,
                        "employee_id": self.employee_id,
                    }
                },
            )
            auth = self._normalize_payload(client.decode_tool_result(auth_result))
            if not isinstance(auth, dict) or auth.get("success") is not True:
                raise MCPGatewayError(f"MCP authentication failed: {auth}")

            result = await client.call_tool(tool_name, {"request": request})
            payload = self._normalize_payload(client.decode_tool_result(result))
            if not isinstance(payload, dict):
                raise MCPGatewayError(
                    f"MCP tool {tool_name} returned an invalid payload."
                )

            success = payload.get("success") is True
            return {
                "source": tool_name,
                "data": payload.get("data") if success else None,
                "code": payload.get("code", "UNKNOWN"),
                "message": payload.get("message", ""),
                "success": success,
            }
        finally:
            await client.close()

    @staticmethod
    def _build_request(
        destination: str,
        query: str,
        *,
        session_id: str,
        customer_id: int | None,
    ) -> tuple[str, dict[str, Any]]:
        if destination == "shipment":
            match = ID_PATTERN.search(query)
            if match is None:
                raise MCPGatewayError(
                    "A shipment request must include a numeric shipment ID."
                )
            return (
                "get_shipment_status",
                {"session_id": session_id, "shipment_id": int(match.group(1))},
            )

        if customer_id is None:
            raise MCPGatewayError(
                f"The {destination} route requires a customer-scoped session."
            )

        mapping = {
            "customer": ("search_customer", "customer_id"),
            "invoice": ("list_customer_invoices", "customer_id"),
            "credit": ("list_customer_credit_holds", "customer_id"),
        }
        if destination not in mapping:
            raise MCPGatewayError(f"Unsupported MCP destination: {destination}")

        tool_name, identifier_field = mapping[destination]
        return (
            tool_name,
            {"session_id": session_id, identifier_field: customer_id},
        )

    @staticmethod
    def _normalize_payload(payload: Any) -> Any:
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload
