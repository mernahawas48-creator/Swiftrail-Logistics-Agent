from __future__ import annotations

from typing import Any

_sessions: set[Any] = set()


def track_session(session: Any) -> None:
    """Track an authenticated MCP session for runtime list-change notices."""

    _sessions.add(session)


async def notify_tools_changed() -> None:
    """Broadcast a real tools/list_changed notification to live sessions."""

    stale: list[Any] = []
    for session in tuple(_sessions):
        sender = getattr(session, "send_tool_list_changed", None)
        if sender is None:
            stale.append(session)
            continue
        try:
            await sender()
        except Exception:
            stale.append(session)

    for session in stale:
        _sessions.discard(session)


def clear_sessions() -> None:
    """Reset tracked sessions for process shutdown and isolated tests."""

    _sessions.clear()
