from __future__ import annotations

from agent_registry import AgentRegistry
from runtime_tool_manager import RuntimeToolManager


def build_runtime_manager(app, modules):
    functions = {}
    server_names = {tool.name for tool in app._tool_manager.list_tools()}
    for module in modules:
        for name, value in vars(module).items():
            if name in server_names and callable(value):
                functions[name] = value

    manager = RuntimeToolManager(app, functions)
    agents = AgentRegistry()
    agents.register("state_graph_1", "State Graph 1", "state_graph")
    agents.register("state_graph_2", "Rate Exception Resolution", "state_graph")
    agents.register("memory_rag", "Memory / RAG Agent", "rag")
    agents.register("planning", "Decomposition / Planning Agent", "planning")

    # Safe defaults: each agent starts with read/authentication capabilities;
    # write tools can be granted from the admin surface at runtime.
    manager.registry.register_agent("state_graph_2", {"authenticate", "get_shipment_status", "get_shipment_rate_exception", "approve_rate_exception"})
    manager.registry.register_agent("memory_rag", {"authenticate"})
    manager.registry.register_agent("planning", {"authenticate", "get_shipment_status", "list_customer_credit_holds"})
    manager.registry.register_agent("state_graph_1", {"authenticate"})
    return manager, agents
