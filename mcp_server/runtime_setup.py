from __future__ import annotations

from agent_registry import AgentRegistry
from runtime_sessions import notify_tools_changed
from runtime_tool_manager import RuntimeToolManager


def build_runtime_manager(app, modules):
    functions = {}
    server_names = {tool.name for tool in app._tool_manager.list_tools()}
    for module in modules:
        for name, value in vars(module).items():
            if name in server_names and callable(value):
                functions[name] = value

    manager = RuntimeToolManager(
        app,
        functions,
        notification_callback=notify_tools_changed,
    )
    agents = AgentRegistry()
    agents.register(
        "graph1_delivery_exception", "Delivery Exception Recovery", "state_graph"
    )
    agents.register(
        "graph2_rate_exception", "Rate Exception Approval", "state_graph"
    )
    agents.register("memory_rag_agent", "Memory / RAG Agent", "memory_rag")
    agents.register("planning_agent", "Decomposition / Planning Agent", "planning")
    agents.register(
        "graph3_credit_hold_remediation",
        "Credit-Hold Remediation",
        "state_graph",
    )

    # Safe defaults: each agent starts with read/authentication capabilities;
    # write tools can be granted from the admin surface at runtime.
    manager.registry.register_agent(
        "graph2_rate_exception",
        {
            "authenticate",
            "get_shipment_status",
            "get_shipment_rate_exception",
            "approve_rate_exception",
        },
    )
    manager.registry.register_agent("memory_rag_agent", {"authenticate"})
    manager.registry.register_agent(
        "planning_agent",
        {
            "authenticate",
            "get_shipment_status",
            "list_customer_credit_holds",
        },
    )
    manager.registry.register_agent(
        "graph1_delivery_exception",
        {
            "authenticate",
            "get_shipment_status",
            "create_delivery_recovery_case",
            "apply_shipment_reroute",
        },
    )
    manager.registry.register_agent(
       "graph3_credit_hold_remediation",
       {
           "authenticate",
           "list_customer_invoices",
           "list_customer_credit_holds",
           "release_credit_hold",
        },
    )
    return manager, agents
