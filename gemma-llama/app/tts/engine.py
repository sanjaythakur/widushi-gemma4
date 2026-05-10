"""Async wrapper around the Piper TTS subprocess.

We deliberately do **not** use the ``piper-tts`` Python wheel: the
prebuilt aarch64 / x86_64 binaries from ``rhasspy/piper`` releases run on
the Pi 5 with no extra Python deps, and a per-call subprocess keeps the
control flow trivially correct (no shared state across requests). Cold
spawn is ~80-200 ms on Pi 5 which is dwarfed by the actual synth time, so
no pooling is needed for the v1.

Each ``synth()`` call invokes ::

    piper --model <voice>.onnx --config <voice>.onnx.json --output_file -

with the input text on stdin and reads a complete WAV (header + PCM) from
stdout. A bounded asyncio semaphore caps concurrency so a burst of
streaming requests cannot fork-bomb the Pi.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .voices import (
    DEFAULT_PERSONALITY,
    PERSONALITIES,
    PiperVoice,
    read_sample_rate,
    resolve_personality,
    voice_paths,
)

logger = logging.getLogger(__name__)


class PiperError(RuntimeError):
    """Raised when the ``piper`` subprocess exits non-zero or is missing."""


@dataclass
class _VoiceCacheEntry:
    voice: PiperVoice
    model_path: Path
    config_path: Path
    sample_rate: int


class PiperEngine:
    """Process-wide TTS engine.

    Lifecycle:
        * :meth:`start` validates the binary + voices directory and warms a
          per-voice metadata cache (sample rates from the ``.onnx.json``).
        * :meth:`synth` renders a single chunk of text to WAV bytes.
        * :meth:`stop` is a no-op today; kept for symmetry with the llama
          adapter so the lifespan-managed shutdown path stays uniform.

    The engine is safe to call from many concurrent requests; concurrency is
    capped by ``max_concurrency`` (default = number of CPU cores - 1, min 1)
    so streaming routes do not collectively pin every core away from
    llama.cpp on Pi 5.
    """

    def __init__(
        self,
        *,
        binary_path: str,
        voices_dir: Path,
        default_voice: str = DEFAULT_PERSONALITY,
        max_concurrency: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.binary_path = binary_path
        self.voices_dir = Path(voices_dir)
        self.default_voice = default_voice
        self.enabled = enabled
        self._voice_cache: dict[str, _VoiceCacheEntry] = {}
        self._sema = asyncio.Semaphore(
            max_concurrency if max_concurrency and max_concurrency > 0 else 2
        )
        self._ready = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Validate the binary + voices dir and warm metadata.

        Never raises -- a failure simply leaves the engine in
        ``ready=False`` so the rest of the API can keep serving requests
        without TTS. Routers honour this by short-circuiting the TTS attach
        with a clear ``audio_error`` shape.
        """
        if not self.enabled:
            logger.info("TTS engine disabled via TTS_ENABLED=false")
            return
        if not shutil.which(self.binary_path) and not Path(self.binary_path).exists():
            logger.warning(
                "piper binary not found at %s; TTS will be unavailable",
                self.binary_path,
            )
            return
        if not self.voices_dir.exists():
            logger.warning(
                "TTS voices dir %s does not exist; TTS will be unavailable",
                self.voices_dir,
            )
            return

        for key, voice in PERSONALITIES.items():
            model_path, config_path = voice_paths(voice, self.voices_dir)
            if not model_path.exists():
                logger.info(
                    "TTS voice %s not present at %s; will 422 if requested",
                    key, model_path,
                )
                continue
            self._voice_cache[key] = _VoiceCacheEntry(
                voice=voice,
                model_path=model_path,
                config_path=config_path,
                sample_rate=read_sample_rate(config_path),
            )
            logger.info(
                "TTS voice loaded: %s (%s, %d Hz)",
                key, voice.voice_id, self._voice_cache[key].sample_rate,
            )

        self._ready = bool(self._voice_cache)
        if not self._ready:
            logger.warning(
                "TTS engine started but no voices were cached; check %s "
                "and the docker/download_voices.sh logs.",
                self.voices_dir,
            )

    async def stop(self) -> None:
        # No long-lived subprocess today. Hook left in place so the lifespan
        # shutdown path stays symmetric with LlamaAdapter.
        return None

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    def available_voices(self) -> list[dict[str, object]]:
        """Return the registry as JSON-serialisable dicts for ``/tts/voices``."""
        out: list[dict[str, object]] = []
        for key, voice in PERSONALITIES.items():
            cached = self._voice_cache.get(key)
            out.append({
                "id": key,
                "description": voice.description,
                "language": voice.language,
                "voice_id": voice.voice_id,
                "downloaded": cached is not None,
                "sample_rate": cached.sample_rate if cached else None,
                "is_default": key == self.default_voice,
            })
        return out

    def sample_rate_for(self, key: str | None) -> int:
        resolved, _ = resolve_personality(key)
        cached = self._voice_cache.get(resolved)
        return cached.sample_rate if cached else 22050

    # ------------------------------------------------------------------
    # synth
    # ------------------------------------------------------------------

    async def synth(self, text: str, voice: str | None = None) -> bytes:
        """Render ``text`` to a complete WAV blob (header + PCM) with ``voice``.

        Empty / whitespace-only input returns ``b""`` so callers can feed the
        flushed remainder of a sentence buffer without a guard.
        """
        if not self._ready:
            raise PiperError(
                "TTS engine is not ready. Check that piper is installed and "
                "that voices have been downloaded into TTS_VOICES_DIR."
            )
        cleaned = (text or "").strip()
        if not cleaned:
            return b""

        key, _ = resolve_personality(voice)
        cached = self._voice_cache.get(key)
        if cached is None:
            raise PiperError(
                f"voice {key!r} is not cached on disk; rerun "
                "docker/download_voices.sh or add the id to TTS_VOICES_ENABLED."
            )

        cmd = [
            self.binary_path,
            "--model", str(cached.model_path),
            "--config", str(cached.config_path),
            "--output_file", "-",
            # --quiet keeps piper from writing progress lines to stderr that we
            # otherwise have to filter on every call.
            "--quiet",
        ]

        async with self._sema:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await proc.communicate(cleaned.encode("utf-8"))
            except asyncio.CancelledError:
                proc.kill()
                await proc.wait()
                raise

        if proc.returncode != 0 or not stdout:
            err = stderr.decode("utf-8", errors="replace").strip() or "(no stderr)"
            raise PiperError(
                f"piper exited with code {proc.returncode}: {err[:400]}"
            )
        return stdout
