"""Base class for orchestrator input sources."""

from __future__ import annotations

import asyncio

from app.events import Event


class InputSource:
    """Abstract input source.

    Subclasses override :meth:`run` and use :meth:`emit` to push events
    onto the shared queue. ``run()`` is started as an asyncio task by
    ``app/main.py`` and is expected to live for the process lifetime.
    """

    def __init__(self, queue: asyncio.Queue[Event]) -> None:
        self._queue = queue

    @property
    def name(self) -> str:
        return type(self).__name__

    async def emit(self, event: Event) -> None:
        await self._queue.put(event)

    async def run(self) -> None:
        raise NotImplementedError
