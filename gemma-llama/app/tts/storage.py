"""Disk-backed cache for ``audio_url``-style TTS outputs.

Every mode endpoint that returns JSON (``/free-convo/turn``,
``/voice-mirror/suggest``, ``/voice-mirror/score``, ``/vision/teach-object``,
``/audio/listen`` non-streaming) writes the full WAV to
``${TTS_OUTPUT_DIR}/<uuid>.wav`` and returns a URL pointing at
``GET /tts/output/{id}.wav``. A background janitor evicts files older than
``ttl_seconds`` so the directory does not grow without bound under 24/7
operation on Pi 5.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


_FILENAME_RE_HINT = "<uuid4>.wav"


class TTSStorage:
    """Tiny TTL store on local disk."""

    def __init__(self, output_dir: Path, ttl_seconds: int = 600) -> None:
        self.output_dir = Path(output_dir)
        self.ttl_seconds = max(30, int(ttl_seconds))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._janitor_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # janitor
    # ------------------------------------------------------------------

    def start_janitor(self, *, interval_seconds: int = 60) -> None:
        if self._janitor_task is not None:
            return
        self._janitor_task = asyncio.create_task(
            self._janitor_loop(interval_seconds), name="tts-janitor"
        )

    async def stop_janitor(self) -> None:
        task = self._janitor_task
        self._janitor_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _janitor_loop(self, interval_seconds: int) -> None:
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                self.evict_expired()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("TTS janitor iteration failed; continuing")

    def evict_expired(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        evicted = 0
        for p in self.output_dir.glob("*.wav"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    evicted += 1
            except OSError:
                continue
        if evicted:
            logger.info("TTS janitor evicted %d expired files", evicted)
        return evicted

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def store(self, wav_bytes: bytes) -> tuple[str, Path]:
        """Persist ``wav_bytes`` and return ``(audio_id, full_path)``."""
        if not wav_bytes:
            raise ValueError("refusing to store empty WAV blob")
        audio_id = uuid.uuid4().hex
        path = self.output_dir / f"{audio_id}.wav"
        path.write_bytes(wav_bytes)
        return audio_id, path

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def resolve(self, audio_id: str) -> Path | None:
        # Defensive id check: filename hint is ``<uuid4>.wav`` so anything
        # beyond hex chars is an obvious traversal attempt.
        if not audio_id or not all(c in "0123456789abcdef" for c in audio_id.lower()):
            return None
        path = self.output_dir / f"{audio_id}.wav"
        if not path.exists():
            return None
        # Honour the TTL on read too so a slow client cannot fish files out
        # past expiry.
        if (time.time() - path.stat().st_mtime) > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return path


# ---------------------------------------------------------------------------
# WAV concatenation helpers (used by attach_file_tts to glue per-sentence
# WAVs into a single file).
# ---------------------------------------------------------------------------


def concat_wavs(blobs: list[bytes]) -> bytes:
    """Join multiple WAV blobs into a single WAV with one shared header.

    All inputs are expected to share the same channels / sample width /
    framerate (Piper guarantees this for a fixed voice). If a blob is empty
    or unreadable we skip it -- a single bad sentence should not poison the
    whole concatenation.
    """
    blobs = [b for b in blobs if b]
    if not blobs:
        return b""

    out = io.BytesIO()
    writer: wave.Wave_write | None = None
    try:
        for blob in blobs:
            try:
                reader = wave.open(io.BytesIO(blob), "rb")
            except (wave.Error, EOFError):
                logger.warning("skipping unreadable WAV blob in concat")
                continue
            try:
                if writer is None:
                    writer = wave.open(out, "wb")
                    writer.setnchannels(reader.getnchannels())
                    writer.setsampwidth(reader.getsampwidth())
                    writer.setframerate(reader.getframerate())
                writer.writeframes(reader.readframes(reader.getnframes()))
            finally:
                reader.close()
    finally:
        if writer is not None:
            writer.close()

    return out.getvalue()


def wav_duration_ms(wav_bytes: bytes) -> float:
    """Return playback duration in milliseconds, or ``0.0`` if unparseable."""
    if not wav_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            frames = r.getnframes()
            rate = r.getframerate() or 1
            return round((frames / rate) * 1000.0, 2)
    except (wave.Error, EOFError):
        return 0.0
