from __future__ import annotations

import asyncio
import threading
from typing import Any


def run_async(awaitable):
    """Run an MCP coroutine from synchronous graph and web worker code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: list[Any] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]
