from __future__ import annotations

import json

import pytest

from platform_app.admin.runtime_client import (
    RuntimeAdminClient,
    RuntimeAdminClientError,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_runtime_client_sends_token_and_tool_change_to_mcp_process():
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse({"id": "agent-1", "tools": []})

    client = RuntimeAdminClient(
        "http://127.0.0.1:8000/admin/runtime",
        "local-test-token",
        opener=opener,
    )
    result = client.set_tool("agent-1", "tool_a", False)

    request, timeout = requests[0]
    assert result["id"] == "agent-1"
    assert request.method == "POST"
    assert request.full_url.endswith("/agents/agent-1/tools")
    assert request.get_header("X-swiftrail-admin-token") == "local-test-token"
    assert json.loads(request.data) == {"tool_name": "tool_a", "enabled": False}
    assert timeout == 10.0


def test_runtime_client_fails_closed_without_admin_token():
    client = RuntimeAdminClient(
        "http://127.0.0.1:8000/admin/runtime",
        "",
    )

    with pytest.raises(RuntimeAdminClientError, match="ADMIN_TOKEN") as error:
        client.list_agents()

    assert error.value.status_code == 503
