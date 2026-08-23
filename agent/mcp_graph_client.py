from __future__ import annotations

from typing import Any

from agent.client import SwiftrailAgent


class GraphMCPClient:
    """Small adapter used by state graphs to call the live MCP server."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000/mcp",
        *,
        agent_id: str | None = None,
        permission_checker: Any | None = None,
    ):
        self.client = SwiftrailAgent("http", url)
        self.connected = False
        self.agent_id = agent_id
        self.permission_checker = permission_checker

    def _check_permission(self, tool_name: str) -> None:
        if self.agent_id is None or self.permission_checker is None:
            return
        checker = self.permission_checker
        allowed = checker(self.agent_id, tool_name)
        if not allowed:
            raise PermissionError(
                f"Agent '{self.agent_id}' is not permitted to call MCP tool '{tool_name}'."
            )

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
        self._check_permission(tool_name)
        await self.connect()
        result = await self.client.call_tool(tool_name, {"request": request})
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError(f"MCP tool {tool_name} returned an error")
        decoded = self.client.decode_tool_result(result)
        if not isinstance(decoded, dict):
            raise TypeError(f"MCP tool {tool_name} returned an invalid response")
        return decoded

    async def list_tools(self) -> list[str]:
        await self.connect()
        return [tool.name for tool in await self.client.discover_tools()]

