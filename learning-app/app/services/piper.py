"""Audio playback facade for Piper-generated WAV bytes.

Gemma owns synthesis through its Piper-backed HTTP API. The learning app
keeps this service slot for the SPEAKING transition, where it plays the
already-rendered WAV bytes and returns only after playback finishes.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

import pygame

from app.hardware.audio import init_pygame_mixer

log = logging.getLogger(__name__)


class PiperClient:
    """Async audio playback facade used by the SPEAKING transition."""

    def __init__(self, *, latency_s: float = 0.2, stub: bool = False) -> None:
        self._latency_s = latency_s
        self._stub = stub
        self._active_channel: Any | None = None

    async def play(self, wav_bytes: bytes, *, duration_ms: float | None = None) -> None:
        """Play WAV bytes and return when playback completes."""

        log.debug("piper.play bytes=%d duration_ms=%s", len(wav_bytes), duration_ms)
        if self._stub:
            await asyncio.sleep(self._stub_sleep_s(duration_ms))
            return

        try:
            self._ensure_mixer()
            sound = pygame.mixer.Sound(file=io.BytesIO(wav_bytes))
            channel = sound.play()
            self._active_channel = channel
            if channel is None:
                raise RuntimeError("pygame mixer could not allocate an audio channel")
            await asyncio.to_thread(_wait_for_channel, channel)
        except asyncio.CancelledError:
            self.stop()
            raise
        finally:
            self._active_channel = None

    def stop(self) -> None:
        """Stop currently playing audio, if any."""

        if self._active_channel is not None:
            self._active_channel.stop()
            self._active_channel = None

    async def aclose(self) -> None:
        """Stop playback during application shutdown."""

        self.stop()

    def _ensure_mixer(self) -> None:
        if not pygame.mixer.get_init():
            init_pygame_mixer()

    def _stub_sleep_s(self, duration_ms: float | None) -> float:
        if duration_ms is None:
            return self._latency_s
        return min(max(duration_ms / 1000.0, self._latency_s), 1.0)


def _wait_for_channel(channel: Any) -> None:
    while channel.get_busy():
        time.sleep(0.02)
