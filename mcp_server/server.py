import asyncio
import sys

from app_instance import app

import tools.read_tools
import tools.rate_exception
import tools.credit_hold
import tools.auth
import tools.portfolio
import tools.resources_prompts

from runtime_setup import build_runtime_manager


# Build runtime agent/tool management AFTER all MCP tools are registered.
runtime_tool_manager, agent_registry = build_runtime_manager(
    app,
    [
        tools.read_tools,
        tools.rate_exception,
        tools.credit_hold,
        tools.auth,
        tools.portfolio,
        tools.resources_prompts,
    ],
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
