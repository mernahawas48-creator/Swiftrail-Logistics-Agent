from __future__ import annotations

from typing import Any

from agent.mcp_graph_client import GraphMCPClient
from state_graph.core.async_utils import run_async


class LiveCreditHoldTools:
    def __init__(self, url="http://127.0.0.1:8000/mcp", *, permission_checker=None):
        self.url = url
        self.permission_checker = permission_checker

    def _call(self, tool_name, *, session_id, employee_id, request):
        async def operation() -> dict[str, Any]:
            client = GraphMCPClient(
                self.url,
                agent_id="graph3_credit_hold_remediation",
                permission_checker=self.permission_checker,
            )
            try:
                auth = await client.authenticate(session_id, employee_id)
                if auth.get("success") is not True:
                    raise RuntimeError(auth.get("message", "MCP authentication failed"))
                response = await client.call(
                    tool_name, {"session_id": session_id, **request}
                )
                if response.get("success") is not True:
                    raise RuntimeError(response.get("message", f"{tool_name} failed"))
                return response["data"]
            finally:
                await client.close()

        return run_async(operation())

    def load_account(self, *, session_id, employee_id, customer_id):
        invoices = self._call(
            "list_customer_invoices",
            session_id=session_id,
            employee_id=employee_id,
            request={"customer_id": customer_id},
        )["invoices"]
        holds = self._call(
            "list_customer_credit_holds",
            session_id=session_id,
            employee_id=employee_id,
            request={"customer_id": customer_id},
        )["active_holds"]
        return invoices, holds

    def release_hold(
        self, *, session_id, employee_id, hold_id, approved, note
    ):
        request: dict[str, Any] = {"hold_id": hold_id}
        if approved is not None:
            request["decision"] = {
                "confirm_release": approved,
                "authorization_note": note,
            }
        return self._call(
            "release_credit_hold",
            session_id=session_id,
            employee_id=employee_id,
            request=request,
        )
