from __future__ import annotations

from agent_registry import AgentRegistry
from mcp.server.fastmcp import FastMCP
from runtime_admin_api import RuntimeAdminAPI
from runtime_tool_manager import RuntimeToolManager
from starlette.testclient import TestClient


def authenticate():
    return {"success": True}


def tool_a():
    return {"success": True}


def build_api():
    server = FastMCP("runtime-admin-test")
    server.add_tool(authenticate, name="authenticate")
    server.add_tool(tool_a, name="tool_a")
    notifications = []

    async def notify():
        notifications.append("tools/list_changed")

    manager = RuntimeToolManager(
        server,
        {"authenticate": authenticate, "tool_a": tool_a},
        notification_callback=notify,
    )
    manager.registry.register_agent("agent-1", {"authenticate", "tool_a"})
    agents = AgentRegistry()
    agents.register("agent-1", "Agent 1", "state_graph")
    api = RuntimeAdminAPI(manager, agents, admin_token="local-test-token")
    api.register(server)
    return TestClient(server.streamable_http_app()), manager, notifications


def test_runtime_admin_api_requires_the_shared_token():
    client, _, _ = build_api()

    response = client.get("/admin/runtime/agents")

    assert response.status_code == 401


def test_runtime_admin_api_changes_the_live_server_and_notifies_sessions():
    client, manager, notifications = build_api()
    headers = {"X-Swiftrail-Admin-Token": "local-test-token"}

    listed = client.get("/admin/runtime/agents", headers=headers)
    assert listed.status_code == 200
    tools = {item["tool_name"]: item for item in listed.json()[0]["tools"]}
    assert tools["tool_a"]["enabled"] is True
    assert tools["tool_a"]["available_on_server"] is True

    removed = client.post(
        "/admin/runtime/agents/agent-1/tools",
        headers=headers,
        json={"tool_name": "tool_a", "enabled": False},
    )
    assert removed.status_code == 200
    assert manager.registry.can_call("agent-1", "tool_a") is False
    assert "tool_a" not in manager.server_tools()
    assert notifications == ["tools/list_changed"]

    restored = client.post(
        "/admin/runtime/agents/agent-1/tools",
        headers=headers,
        json={"tool_name": "tool_a", "enabled": True},
    )
    assert restored.status_code == 200
    assert manager.registry.can_call("agent-1", "tool_a") is True
    assert "tool_a" in manager.server_tools()
    assert notifications == ["tools/list_changed", "tools/list_changed"]


def test_runtime_admin_api_refuses_to_remove_authenticate():
    client, manager, _ = build_api()

    response = client.post(
        "/admin/runtime/agents/agent-1/tools",
        headers={"X-Swiftrail-Admin-Token": "local-test-token"},
        json={"tool_name": "authenticate", "enabled": False},
    )

    assert response.status_code == 400
    assert manager.registry.can_call("agent-1", "authenticate") is True
