import asyncio
import sys

from app_instance import app

import tools.read_tools
import tools.rate_exception
import tools.credit_hold
import tools.auth
import tools.portfolio
import tools.resources_prompts


if __name__ == "__main__":
    # Transport: stdio for local development (default), Streamable HTTP for
    # a real remote deployment (matches the transport concern -- see README).
    #   python server.py            -> stdio
    #   python server.py --http     -> Streamable HTTP on 127.0.0.1:8000
    if "--http" in sys.argv:
        print("Starting Swiftrail MCP server on Streamable HTTP: http://127.0.0.1:8000/mcp")
        asyncio.run(app.run_streamable_http_async())
    else:
        asyncio.run(app.run_stdio_async())
