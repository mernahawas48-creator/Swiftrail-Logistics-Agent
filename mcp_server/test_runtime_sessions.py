from __future__ import annotations

import asyncio

from runtime_sessions import clear_sessions, notify_tools_changed, track_session


class FakeSession:
    def __init__(self) -> None:
        self.notifications = 0

    async def send_tool_list_changed(self) -> None:
        self.notifications += 1


def test_runtime_notifications_reach_authenticated_mcp_sessions():
    clear_sessions()
    first = FakeSession()
    second = FakeSession()
    track_session(first)
    track_session(second)

    asyncio.run(notify_tools_changed())

    assert first.notifications == 1
    assert second.notifications == 1
    clear_sessions()
