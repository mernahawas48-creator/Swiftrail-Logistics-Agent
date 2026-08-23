from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from agent_registry import AgentRegistry
from dotenv import load_dotenv
from runtime_tool_manager import RuntimeToolManager
from starlette.requests import Request
from starlette.responses import JSONResponse


class RuntimeAdminAPI:
    """Token-protected HTTP bridge to the live MCP runtime manager."""

    def __init__(
        self,
        manager: RuntimeToolManager,
        agents: AgentRegistry,
        *,
        admin_token: str | None = None,
    ) -> None:
        if admin_token is None:
            project_root = Path(__file__).resolve().parents[1]
            load_dotenv(project_root / ".env")
            admin_token = os.getenv("SWIFTRAIL_ADMIN_TOKEN", "")
        self.manager = manager
        self.agents = agents
        self.admin_token = admin_token.strip()

    def register(self, app: Any) -> None:
        app.custom_route(
            "/admin/runtime/health",
            methods=["GET"],
            include_in_schema=False,
        )(self.health)
        app.custom_route(
            "/admin/runtime/agents",
            methods=["GET"],
            include_in_schema=False,
        )(self.list_agents)
        app.custom_route(
            "/admin/runtime/agents/{agent_id}/tools",
            methods=["POST"],
            include_in_schema=False,
        )(self.set_tool)

    async def health(self, request: Request) -> JSONResponse:
        error = self._authorize(request)
        if error is not None:
            return error
        return JSONResponse(
            {
                "status": "ok",
                "server_tools": sorted(self.manager.server_tools()),
                "agent_count": len(self.agents.list()),
            }
        )

    async def list_agents(self, request: Request) -> JSONResponse:
        error = self._authorize(request)
        if error is not None:
            return error
        return JSONResponse(
            [self._agent_payload(record) for record in self.agents.as_dicts()]
        )

    async def set_tool(self, request: Request) -> JSONResponse:
        error = self._authorize(request)
        if error is not None:
            return error

        agent_id = request.path_params["agent_id"]
        record = self.agents.get(agent_id)
        if record is None:
            return JSONResponse({"detail": "Unknown agent."}, status_code=404)

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

        tool_name = payload.get("tool_name") if isinstance(payload, dict) else None
        enabled = payload.get("enabled") if isinstance(payload, dict) else None
        if not isinstance(tool_name, str) or not isinstance(enabled, bool):
            return JSONResponse(
                {"detail": "tool_name must be a string and enabled must be boolean."},
                status_code=400,
            )

        try:
            if enabled:
                await self.manager.add_tool_to_agent(agent_id, tool_name)
            else:
                await self.manager.remove_tool_from_agent(agent_id, tool_name)
        except KeyError:
            return JSONResponse({"detail": "Unknown tool."}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

        return JSONResponse(
            self._agent_payload(
                {
                    "agent_id": record.agent_id,
                    "name": record.name,
                    "kind": record.kind,
                }
            )
        )

    def _agent_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(record.get("agent_id"))
        enabled = self.manager.registry.tools_for(agent_id)
        server_tools = self.manager.server_tools()
        return {
            "id": agent_id,
            "name": record.get("name", agent_id),
            "kind": record.get("kind", "agent"),
            "tools": [
                {
                    "tool_name": tool_name,
                    "enabled": tool_name in enabled,
                    "protected": tool_name in self.manager.PROTECTED_TOOLS,
                    "available_on_server": tool_name in server_tools,
                }
                for tool_name in sorted(self.manager.tool_functions)
            ],
        }

    def _authorize(self, request: Request) -> JSONResponse | None:
        if not self.admin_token:
            return JSONResponse(
                {"detail": "SWIFTRAIL_ADMIN_TOKEN is not configured."},
                status_code=503,
            )
        provided = request.headers.get("x-swiftrail-admin-token", "")
        if not secrets.compare_digest(provided, self.admin_token):
            return JSONResponse({"detail": "Unauthorized."}, status_code=401)
        return None


def register_runtime_admin_api(
    app: Any,
    manager: RuntimeToolManager,
    agents: AgentRegistry,
) -> RuntimeAdminAPI:
    api = RuntimeAdminAPI(manager, agents)
    api.register(app)
    return api
