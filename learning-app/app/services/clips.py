"""Pre-generated WAV clip playback.

Plays the short acknowledgement / status clips produced by the sibling
``pre-generated-clips`` project (e.g. ``listen_start``, ``wait_thinking``,
``wait_checking``). Shares pygame's process-global mixer with
:class:`app.services.piper.PiperClient` so the same audio device is used
for canned clips and Gemma TTS replies.

The class is intentionally tolerant: a missing clip file logs a warning
and returns instead of raising, so a half-shipped clip set never breaks
a turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import pygame

from app import config

log = logging.getLogger(__name__)


class ClipPlayer:
    """Async player for pre-rendered ``.wav`` clips."""

    def __init__(
        self,
        *,
        clips_dir: Path = config.CLIPS_DIR,
        lang: str = config.CLIPS_LANG,
        latency_s: float = 0.2,
        stub: bool = False,
    ) -> None:
        self._clips_dir = clips_dir
        self._lang = lang
        self._latency_s = latency_s
        self._stub = stub
        self._active_channel: Any | None = None
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}

    @property
    def lang(self) -> str:
        return self._lang

    def clip_path(self, clip_id: str) -> Path:
        return self._clips_dir / self._lang / f"{clip_id}.wav"

    async def play(self, clip_id: str) -> None:
        """Play ``<clips_dir>/<lang>/<clip_id>.wav`` and return when done."""

        path = self.clip_path(clip_id)
        log.debug("clips.play id=%s path=%s", clip_id, path)
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return

        if not path.exists():
            log.warning("clip not found: %s; skipping playback", path)
            return

        try:
            self._ensure_mixer()
            sound = self._load(clip_id, path)
            channel = sound.play()
            self._active_channel = channel
            if channel is None:
                raise RuntimeError("pygame mixer could not allocate an audio channel")
            await asyncio.to_thread(_wait_for_channel, channel)
        except asyncio.CancelledError:
            self.stop()
            raise
        except Exception:
            log.exception("clip playback failed: %s", path)
        finally:
            self._active_channel = None

    def stop(self) -> None:
        if self._active_channel is not None:
            self._active_channel.stop()
            self._active_channel = None

    async def aclose(self) -> None:
        self.stop()
        self._sound_cache.clear()

    def _load(self, clip_id: str, path: Path) -> pygame.mixer.Sound:
        cached = self._sound_cache.get(clip_id)
        if cached is not None:
            return cached
        sound = pygame.mixer.Sound(file=str(path))
        self._sound_cache[clip_id] = sound
        return sound

    def _ensure_mixer(self) -> None:
        if not pygame.mixer.get_init():
            pygame.mixer.init()


def _wait_for_channel(channel: Any) -> None:
    while channel.get_busy():
        time.sleep(0.02)
