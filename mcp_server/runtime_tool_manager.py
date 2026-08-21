from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentToolRegistry:
    """In-memory runtime registry of agent-specific MCP tool permissions."""

    permissions: dict[str, set[str]] = field(default_factory=dict)

    def register_agent(self, agent_id: str, tools: set[str] | None = None) -> None:
        self.permissions.setdefault(agent_id, set()).update(tools or set())

    def add(self, agent_id: str, tool_name: str) -> None:
        self.permissions.setdefault(agent_id, set()).add(tool_name)

    def remove(self, agent_id: str, tool_name: str) -> None:
        self.permissions.setdefault(agent_id, set()).discard(tool_name)

    def tools_for(self, agent_id: str) -> set[str]:
        return set(self.permissions.get(agent_id, set()))

    def can_call(self, agent_id: str, tool_name: str) -> bool:
        """Return whether an agent currently has permission for a tool."""
        return tool_name in self.permissions.get(agent_id, set())

    def require_permission(self, agent_id: str, tool_name: str) -> None:
        """Fail closed when a caller tries to use an ungranted tool."""
        if not self.can_call(agent_id, tool_name):
            raise PermissionError(
                f"Agent '{agent_id}' is not permitted to call MCP tool '{tool_name}'."
            )



class RuntimeToolManager:
    """Bridge the admin panel's tool permissions to the live FastMCP server.

    FastMCP exposes public add_tool/remove_tool APIs. We retain the original
    Python function for every tool so a removed tool can be registered again
    without a redeploy. The manager only removes a server tool when no agent
    still has permission to use it.
    """

    PROTECTED_TOOLS = {"authenticate"}

    def __init__(self, server: Any, tool_functions: dict[str, Callable[..., Any]]):
        self.server = server
        self.tool_functions = dict(tool_functions)
        self.registry = AgentToolRegistry()
        self._lock = asyncio.Lock()

    def register_agent(self, agent_id: str, tools: set[str] | None = None) -> dict[str, Any]:
        self.registry.register_agent(agent_id, tools)
        return self.snapshot(agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent and clean up server tools no longer used."""
        tools = self.registry.permissions.pop(agent_id, set())
        for tool_name in tools:
            if tool_name in self.PROTECTED_TOOLS:
                continue
            still_used = any(tool_name in t for t in self.registry.permissions.values())
            if not still_used:
                self._remove_server_tool(tool_name)

    def set_agent_tools(self, agent_id: str, tools: set[str]) -> dict[str, Any]:
        """Replace an agent's permissions atomically at the registry level."""
        current = self.registry.tools_for(agent_id)
        for tool_name in current - tools:
            self.registry.remove(agent_id, tool_name)
        for tool_name in tools:
            if tool_name not in self.tool_functions:
                raise KeyError(f"Unknown tool: {tool_name}")
            self.registry.add(agent_id, tool_name)
            self._ensure_server_tool(tool_name, self.tool_functions[tool_name])
        return self.snapshot(agent_id)

    async def add_tool_to_agent(self, agent_id: str, tool_name: str) -> dict[str, Any]:
        async with self._lock:
            function = self.tool_functions.get(tool_name)
            if function is None:
                raise KeyError(f"Unknown tool: {tool_name}")
            self.registry.add(agent_id, tool_name)
            self._ensure_server_tool(tool_name, function)
            await self._notify_tools_changed()
            return self.snapshot(agent_id)

    async def remove_tool_from_agent(self, agent_id: str, tool_name: str) -> dict[str, Any]:
        if tool_name in self.PROTECTED_TOOLS:
            raise ValueError(f"{tool_name} is required for authenticated sessions and cannot be removed")
        async with self._lock:
            self.registry.remove(agent_id, tool_name)
            still_used = any(
                tool_name in tools for tools in self.registry.permissions.values()
            )
            if not still_used:
                self._remove_server_tool(tool_name)
            await self._notify_tools_changed()
            return self.snapshot(agent_id)

    def snapshot(self, agent_id: str) -> dict[str, Any]:
        return {
            "agent_id": agent_id,
            "tools": sorted(self.registry.tools_for(agent_id)),
            "server_tools": sorted(self.server_tools()),
        }

    def server_tools(self) -> set[str]:
        manager = getattr(self.server, "_tool_manager", None)
        if manager is None:
            return set()
        return {tool.name for tool in manager.list_tools()}

    def _ensure_server_tool(self, name: str, function: Callable[..., Any]) -> None:
        if name not in self.server_tools():
            self.server.add_tool(function, name=name)

    def _remove_server_tool(self, name: str) -> None:
        if name in self.server_tools():
            self.server.remove_tool(name)

    async def _notify_tools_changed(self) -> None:
        notifier = getattr(self.server, "notify_tools_changed", None)
        if notifier is None:
            return
        result = notifier()
        if asyncio.iscoroutine(result):
            await result

