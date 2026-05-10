"""Router-side glue that keeps each route's TTS opt-in to ~3 lines.

Each helper returns a dict you can ``**`` into the response model. The dicts
include both the success-shape fields (``audio_url`` / ``audio_base64`` /
``audio_duration_ms`` / ``voice``) and an optional ``audio_error`` so the
caller never has to branch on engine availability.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .engine import PiperEngine
from .integration import (
    attach_file_tts,
    attach_inline_tts,
    wrap_stream_with_tts,
)
from .storage import TTSStorage


async def maybe_attach_inline(
    text: str,
    *,
    enabled: bool,
    voice: str | None,
    engine: PiperEngine | None,
) -> dict[str, object]:
    """Run inline (base64) TTS attach when enabled; else return ``{}``.

    Routers can always splat the result into their response constructor:
    when ``enabled=False`` (or no engine), the returned dict is empty so
    the response keeps its default ``audio_*=None`` fields.
    """
    if not enabled:
        return {}
    if engine is None or not engine.ready:
        return {
            "audio_base64": None,
            "audio_mime": "audio/wav",
            "voice": voice,
            "audio_error": "TTS engine not ready",
        }
    return await attach_inline_tts(text, engine=engine, voice=voice)


async def maybe_attach_file(
    text: str,
    *,
    enabled: bool,
    voice: str | None,
    engine: PiperEngine | None,
    storage: TTSStorage | None,
) -> dict[str, object]:
    """Run file (URL) TTS attach when enabled; else return ``{}``."""
    if not enabled:
        return {}
    if engine is None or not engine.ready or storage is None:
        return {
            "audio_url": None,
            "voice": voice,
            "audio_error": "TTS engine not ready",
        }
    return await attach_file_tts(text, engine=engine, storage=storage, voice=voice)


def maybe_wrap_stream(
    upstream: AsyncIterator[str],
    *,
    enabled: bool,
    voice: str | None,
    engine: PiperEngine | None,
    max_chars: int = 400,
) -> AsyncIterator[str]:
    """Wrap an upstream NDJSON iterator with TTS audio lines when enabled."""
    if not enabled or engine is None or not engine.ready:
        return upstream
    return wrap_stream_with_tts(
        upstream, engine=engine, voice=voice, max_chars=max_chars
    )
