"""Persistence client.

Phase 2 stub: in-memory dict that mirrors the eventual aiosqlite
key-value surface (``get``, ``set``, ``delete``) so call sites don't
need to change.
"""

from __future__ import annotations

import asyncio
from typing import Any


class Database:
    """Tiny async key-value store used by the orchestrator for state."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str, default: Any | None = None) -> Any | None:
        async with self._lock:
            return self._data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def close(self) -> None:
        """No-op for the stub; real client will close the sqlite handle."""

        return None
