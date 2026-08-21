from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from mcp.server.lowlevel.server import NotificationOptions
from pydantic import ConfigDict


def _enforce_strict_generated_tool_arguments() -> None:
    """Make FastMCP's generated top-level tool argument models strict.

    The MCP Python SDK v1 generates an internal ``*Arguments`` Pydantic model
    from each tool function signature. Its default configuration ignores
    unknown fields, even when the nested request model uses ``extra="forbid"``.
    Applying the strict configuration before any tools are registered makes
    every published top-level inputSchema include:

        "additionalProperties": false

    The handlers still validate their explicit request models independently.
    """

    updated_config = dict(ArgModelBase.model_config)
    updated_config["extra"] = "forbid"
    ArgModelBase.model_config = ConfigDict(**updated_config)
    ArgModelBase.model_rebuild(force=True)


# This must run before tool modules are imported and their decorators execute.
_enforce_strict_generated_tool_arguments()


class CapabilityAwareFastMCP(FastMCP):
    """FastMCP server that truthfully declares dynamic tool-list changes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        low_level_server = self._mcp_server
        original_factory = low_level_server.create_initialization_options

        def create_initialization_options(
            notification_options=None,
            experimental_capabilities=None,
        ):
            requested = notification_options or NotificationOptions()
            accurate_options = NotificationOptions(
                prompts_changed=bool(
                    getattr(requested, "prompts_changed", False)
                ),
                resources_changed=bool(
                    getattr(requested, "resources_changed", False)
                ),
                tools_changed=True,
            )
            return original_factory(
                notification_options=accurate_options,
                experimental_capabilities=experimental_capabilities or {},
            )

        low_level_server.create_initialization_options = (
            create_initialization_options
        )


app = CapabilityAwareFastMCP(
    "swiftrail-mcp-server",
    host="127.0.0.1",
    port=8000,
    instructions=(
        "Use scoped Swiftrail tools only. Financial writes require server-side "
        "authorization, and severe risk actions require human elicitation."
    ),
)