from __future__ import annotations

from agent.mcp_graph_client import GraphMCPClient


class LiveRateExceptionTools:
    """Live MCP implementation used by Graph 2 in production/demo mode."""

    def __init__(self, url: str = "http://127.0.0.1:8000/mcp", *, agent_id: str = "state_graph_2", permission_checker=None):
        self.client = GraphMCPClient(url, agent_id=agent_id, permission_checker=permission_checker)

    async def authenticate(self, session_id: str, employee_id: int):
        return await self.client.authenticate(session_id, employee_id)

    async def shipment_status(self, session_id: str, shipment_id: int):
        return await self.client.call("get_shipment_status", {"session_id": session_id, "shipment_id": shipment_id})

    async def rate_exception(self, session_id: str, shipment_id: int):
        return await self.client.call("get_shipment_rate_exception", {"session_id": session_id, "shipment_id": shipment_id})

    async def approve(self, session_id: str, exception_id: int, decision: bool, note: str):
        # The real MCP tool uses elicitation for the human decision. Graph 2's
        # platform already collected the decision, so the adapter provides the
        # MCP call and leaves the final human approval to the server policy.
        return await self.client.call(
            "approve_rate_exception",
            {"session_id": session_id, "exception_id": exception_id, "decision": {"approve": decision, "reviewer_note": note}},
        )

