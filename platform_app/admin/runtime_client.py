from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv


class RuntimeAdminClientError(RuntimeError):
    """Raised when the platform cannot update the live MCP runtime."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class RuntimeAdminClient:
    """Small authenticated client for the MCP server's internal admin API."""

    def __init__(
        self,
        base_url: str,
        admin_token: str,
        *,
        timeout: float = 10.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token.strip()
        self.timeout = timeout
        self.opener = opener

    @classmethod
    def from_env(cls) -> RuntimeAdminClient:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        return cls(
            os.getenv(
                "SWIFTRAIL_MCP_ADMIN_URL",
                "http://127.0.0.1:8000/admin/runtime",
            ),
            os.getenv("SWIFTRAIL_ADMIN_TOKEN", ""),
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_agents(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/agents")
        if not isinstance(payload, list):
            raise RuntimeAdminClientError("The MCP admin API returned invalid agents.")
        return payload

    def set_tool(
        self,
        agent_id: str,
        tool_name: str,
        enabled: bool,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/agents/{quote(agent_id, safe='')}/tools",
            {"tool_name": tool_name, "enabled": enabled},
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if not self.admin_token:
            raise RuntimeAdminClientError(
                "SWIFTRAIL_ADMIN_TOKEN is missing from the root .env file.",
                status_code=503,
            )

        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Swiftrail-Admin-Token": self.admin_token,
            },
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = self._error_detail(exc)
            raise RuntimeAdminClientError(detail, status_code=exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeAdminClientError(
                "The live MCP admin API is unavailable. Start mcp_server/server.py "
                "with --http and verify port 8000."
            ) from exc

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return f"The MCP admin API returned HTTP {error.code}."
        return str(payload.get("detail", f"The MCP admin API returned HTTP {error.code}."))
