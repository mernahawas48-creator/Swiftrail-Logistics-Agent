from __future__ import annotations

from typing import Any


def collect_registered_tool_functions(modules: list[Any]) -> dict[str, Any]:
    """Collect decorated tool functions from the existing tool modules."""
    result: dict[str, Any] = {}
    for module in modules:
        for name, value in vars(module).items():
            if callable(value) and not name.startswith("_"):
                # The module only exposes its actual @app.tool functions plus
                # imported helpers. Prefer names present in the server tool list.
                result.setdefault(name, value)
    return result
