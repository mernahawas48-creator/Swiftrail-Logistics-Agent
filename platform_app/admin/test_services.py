import asyncio

from platform_app.admin.services import AdminService
from mcp_server.agent_registry import AgentRegistry
from mcp_server.runtime_tool_manager import RuntimeToolManager


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeToolManager:
    def __init__(self):
        self.tools = {}

    def list_tools(self):
        return list(self.tools.values())


class FakeServer:
    def __init__(self):
        self._tool_manager = FakeToolManager()
        self.notifications = 0

    def add_tool(self, fn, name):
        self._tool_manager.tools[name] = FakeTool(name)

    def remove_tool(self, name):
        self._tool_manager.tools.pop(name, None)

    async def notify_tools_changed(self):
        self.notifications += 1


def test_admin_tool_changes_reach_live_server():
    server = FakeServer()
    manager = RuntimeToolManager(server, {"tool_a": lambda: None, "authenticate": lambda: None})
    agents = AgentRegistry()
    agents.register("state_graph_2", "Graph 2", "state_graph")
    service = AdminService(manager, agents)

    added = asyncio.run(service.add_tool("state_graph_2", "tool_a"))
    assert "tool_a" in added["tools"]
    assert "tool_a" in added["server_tools"]

    removed = asyncio.run(service.remove_tool("state_graph_2", "tool_a"))
    assert "tool_a" not in removed["tools"]
    assert "tool_a" not in removed["server_tools"]
    assert server.notifications == 2


def test_agent_registration_and_permission_enforcement():
    server = FakeServer()
    manager = RuntimeToolManager(server, {"tool_a": lambda: None, "tool_b": lambda: None})
    agents = AgentRegistry()
    service = AdminService(manager, agents)

    snapshot = service.register_agent("agent-1", "Agent 1", "state_graph", {"tool_a"})
    assert snapshot["tools"] == ["tool_a"]
    assert manager.registry.can_call("agent-1", "tool_a")
    assert not manager.registry.can_call("agent-1", "tool_b")

    import pytest
    with pytest.raises(PermissionError):
        manager.registry.require_permission("agent-1", "tool_b")

    service.unregister_agent("agent-1")
    assert agents.get("agent-1") is None
    assert not manager.registry.can_call("agent-1", "tool_a")


