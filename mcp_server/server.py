import asyncio
import sys

import tools.auth
import tools.credit_hold
import tools.delivery_recovery
import tools.portfolio
import tools.rate_exception
import tools.read_tools
import tools.resources_prompts
from app_instance import app
from runtime_admin_api import register_runtime_admin_api
from runtime_setup import build_runtime_manager

# Build runtime agent/tool management AFTER all MCP tools are registered.
runtime_tool_manager, agent_registry = build_runtime_manager(
    app,
    [
        tools.read_tools,
        tools.rate_exception,
        tools.credit_hold,
        tools.delivery_recovery,
        tools.auth,
        tools.portfolio,
        tools.resources_prompts,
    ],
)
runtime_admin_api = register_runtime_admin_api(
    app,
    runtime_tool_manager,
    agent_registry,
)


if __name__ == "__main__":
    # Transport:
    #   python server.py            -> stdio
    #   python server.py --http     -> Streamable HTTP
    if "--http" in sys.argv:
        print(
            "Starting Swiftrail MCP server on "
            "Streamable HTTP: http://127.0.0.1:8000/mcp"
        )
        asyncio.run(app.run_streamable_http_async())
    else:
        asyncio.run(app.run_stdio_async())
