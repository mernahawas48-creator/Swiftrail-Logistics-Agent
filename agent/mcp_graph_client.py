from __future__ import annotations

from typing import Any

from agent.client import SwiftrailAgent


class GraphMCPClient:
    """Small adapter used by state graphs to call the live MCP server."""

    def __init__(self, url: str = "http://127.0.0.1:8000/mcp"):
        self.client = SwiftrailAgent("http", url)
        self.connected = False

    async def connect(self) -> None:
        if not self.connected:
            await self.client.connect()
            self.connected = True

    async def close(self) -> None:
        if self.connected:
            await self.client.close()
            self.connected = False

    async def authenticate(self, session_id: str, employee_id: int) -> dict[str, Any]:
        await self.connect()
        result = await self.client.call_tool(
            "authenticate",
            {"request": {"session_id": session_id, "employee_id": employee_id}},
        )
        return self.client.decode_tool_result(result)

    async def call(self, tool_name: str, request: dict[str, Any]) -> dict[str, Any]:
        await self.connect()
        result = await self.client.call_tool(tool_name, {"request": request})
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError(f"MCP tool {tool_name} returned an error")
        decoded = self.client.decode_tool_result(result)
        if not isinstance(decoded, dict):
            raise RuntimeError(  # noqa: TRY004 - malformed external response
                f"MCP tool {tool_name} returned an invalid response"
            )
        return decoded

    async def list_tools(self) -> list[str]:
        await self.connect()
        return [tool.name for tool in await self.client.discover_tools()]
