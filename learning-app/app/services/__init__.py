"""Async clients for external resources (LLM, TTS, persistence).

Phase 2 ships stub implementations whose method signatures match what
the real clients (httpx for Gemma, ``asyncio.create_subprocess_exec``
for Piper, ``aiosqlite`` for storage) will look like, so later phases
swap implementations without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.services.clips import ClipPlayer
from app.services.db import Database
from app.services.gemma import GemmaClient
from app.services.piper import PiperClient


@dataclass
class Services:
    """Bundle of async clients passed into the orchestrator."""

    gemma: GemmaClient
    piper: PiperClient
    clips: ClipPlayer
    db: Database

    @classmethod
    def stubs(cls) -> Services:
        """Build the all-stub Services bundle used in Phase 2."""

        return cls(
            gemma=GemmaClient(stub=True),
            piper=PiperClient(stub=True),
            clips=ClipPlayer(stub=True),
            db=Database(),
        )

    @classmethod
    def live(cls) -> Services:
        """Build runtime services backed by local external resources."""

        return cls(
            gemma=GemmaClient(url=config.GEMMA_URL, timeout_s=config.GEMMA_TIMEOUT_S),
            piper=PiperClient(),
            clips=ClipPlayer(),
            db=Database(),
        )


__all__ = ["ClipPlayer", "Database", "GemmaClient", "PiperClient", "Services"]
