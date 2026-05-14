"""OpenWakeWord input source."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Protocol

from app import config
from app.events import Event, EventType
from app.hardware.mic import MicSource
from app.input.sources import InputSource
from app.input.utterance import (
    FollowUpListenSignal,
    UtteranceStopSignal,
    pcm16_mono_to_wav,
)
from app.services.clips import ClipPlayer
from app.state import AppState

log = logging.getLogger(__name__)

OPENWAKEWORD_SAMPLE_RATE = 16000
OPENWAKEWORD_FRAME_SAMPLES = 1280
OPENWAKEWORD_SAMPLE_WIDTH_BYTES = 2
OPENWAKEWORD_FRAME_BYTES = OPENWAKEWORD_FRAME_SAMPLES * OPENWAKEWORD_SAMPLE_WIDTH_BYTES


class WakeWordDetector(Protocol):
    def predict(self, pcm16: bytes) -> Mapping[str, float]:
        """Return wake-word scores for one 80 ms, 16 kHz, mono int16 frame."""


DetectorFactory = Callable[[Path], WakeWordDetector]


class OpenWakeWordDetector:
    """Thin wrapper around openWakeWord's synchronous model API."""

    def __init__(self, model_path: Path) -> None:
        try:
            from openwakeword.model import Model  # noqa: PLC0415 - optional runtime dependency
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword is required for wake-word detection; "
                "install dependencies or set WAKE_WORD_ENABLED=0"
            ) from exc

        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
        )

    def predict(self, pcm16: bytes) -> Mapping[str, float]:
        import numpy as np  # noqa: PLC0415 - optional runtime dependency

        frame = np.frombuffer(pcm16, dtype=np.int16)
        return self._model.predict(frame)


class WakeWordSource(InputSource):
    def __init__(
        self,
        queue: asyncio.Queue[Event],
        *,
        mic: MicSource | None = None,
        model_path: Path = config.WAKE_WORD_MODEL,
        enabled: bool = config.WAKE_WORD_ENABLED,
        threshold: float = config.WAKE_WORD_THRESHOLD,
        debounce_s: float = config.WAKE_WORD_DEBOUNCE_S,
        stop_signal: UtteranceStopSignal | None = None,
        followup_signal: FollowUpListenSignal | None = None,
        clips: ClipPlayer | None = None,
        detector_factory: DetectorFactory = OpenWakeWordDetector,
        clock: Callable[[], float] = time.monotonic,
        state_getter: Callable[[], AppState] | None = None,
    ) -> None:
        super().__init__(queue)
        self._mic = mic or MicSource(
            sample_rate=OPENWAKEWORD_SAMPLE_RATE,
            chunk_ms=config.WAKE_WORD_FRAME_MS,
            stub=not enabled,
        )
        self._model_path = model_path
        self._enabled = enabled
        self._threshold = threshold
        self._debounce_s = debounce_s
        self._stop_signal = stop_signal
        self._followup_signal = followup_signal
        self._clips = clips
        self._detector_factory = detector_factory
        self._clock = clock
        # Optional read-only view of the orchestrator's current state.
        # When provided, wake-word prediction is gated on the FSM being
        # in IDLE so background noise (or the user's continued talking)
        # cannot start a phantom recording while a turn is mid-flight in
        # THINKING/SPEAKING. None means "always-on" (used by the unit
        # tests, which exercise the source in isolation).
        self._state_getter = state_getter
        self._last_detection_at = 0.0
        self._gated_state_logged: AppState | None = None

    async def run(self) -> None:
        if not self._enabled:
            log.info("wake-word source disabled (set WAKE_WORD_ENABLED=1 to enable)")
            await asyncio.Event().wait()
            return

        if not self._model_path.exists():
            log.warning(
                "wake-word model not found at %s; detection disabled",
                self._model_path,
            )
            await asyncio.Event().wait()
            return

        detector = self._detector_factory(self._model_path)
        log.info(
            "wake-word source enabled: model=%s threshold=%.2f debounce=%.1fs",
            self._model_path,
            self._threshold,
            self._debounce_s,
        )

        buffer = bytearray()
        frames = self._mic.frames()
        try:
            async for chunk in frames:
                prepared = self._prepare_chunk(chunk)

                # The FSM raises this when SPEAKING -> LISTENING fires
                # (i.e. PLAYBACK_DONE). Skip wake-word detection and
                # drop the wake-word frame buffer; the user is mid-
                # conversation and we want their next sentence captured
                # without re-saying "Widushi". The current chunk is
                # prepended back into the mic stream so the recorder's
                # normal prepare + RMS path observes it (otherwise its
                # leading-edge voice would never be analyzed for
                # voiced_seen).
                if (
                    self._followup_signal is not None
                    and self._followup_signal.consume()
                ):
                    buffer.clear()
                    self._gated_state_logged = None
                    log.info("follow-up listen requested; recording without wake word")
                    payload = await self._record_utterance(
                        _prepend_chunk(chunk, frames),
                        bytearray(),
                    )
                    payload["followup"] = True
                    event_type = (
                        EventType.CANCEL
                        if payload.get("cancel_reason")
                        else EventType.UTTERANCE_END
                    )
                    await self.emit(Event(event_type, payload=payload))
                    continue

                # Skip wake-word detection while a turn is in flight.
                # Without this, the user's continued talking (or pure
                # background noise above threshold) keeps re-firing
                # WAKE_DETECTED during THINKING/SPEAKING. The FSM
                # silently ignores those events (no entry in the
                # dispatch table for that state), but the source still
                # plays ``listen_start`` and starts a 15 s recording
                # whose UTTERANCE_END is also dropped — wasting Gemma
                # latency and confusing the user, who sees ``listening
                # for spoken question`` lines while the UI still shows
                # the THINKING face.
                current_state = (
                    self._state_getter() if self._state_getter is not None else None
                )
                if current_state is not None and current_state is not AppState.IDLE:
                    if buffer:
                        buffer.clear()
                    if self._gated_state_logged is not current_state:
                        log.debug(
                            "wake-word detection paused while FSM in %s",
                            current_state.name,
                        )
                        self._gated_state_logged = current_state
                    continue
                self._gated_state_logged = None

                buffer.extend(prepared)
                while len(buffer) >= OPENWAKEWORD_FRAME_BYTES:
                    frame = bytes(buffer[:OPENWAKEWORD_FRAME_BYTES])
                    del buffer[:OPENWAKEWORD_FRAME_BYTES]
                    if await self._predict_frame(detector, frame):
                        # Drop the wake-word pre-roll: by the time the
                        # listen_start clip finishes, those samples are
                        # stale and would otherwise prefix the recording.
                        buffer.clear()
                        await self._play_listen_start()
                        payload = await self._record_utterance(frames, buffer)
                        buffer.clear()
                        payload["followup"] = False
                        event_type = (
                            EventType.CANCEL
                            if payload.get("cancel_reason")
                            else EventType.UTTERANCE_END
                        )
                        await self.emit(Event(event_type, payload=payload))
                        break
        except Exception:
            log.exception("wake-word source failed; detection disabled")
            await asyncio.Event().wait()

    def _prepare_chunk(self, chunk: bytes) -> bytes:
        if self._mic.sample_rate == OPENWAKEWORD_SAMPLE_RATE:
            return chunk

        return _resample_pcm16(chunk, self._mic.sample_rate, OPENWAKEWORD_SAMPLE_RATE)

    async def _predict_frame(self, detector: WakeWordDetector, frame: bytes) -> bool:
        scores = await asyncio.to_thread(detector.predict, frame)
        label, score = _best_score(scores)
        if label is None or score < self._threshold:
            return False

        now = self._clock()
        if now - self._last_detection_at < self._debounce_s:
            return False

        self._last_detection_at = now
        log.info("wake word detected: label=%s score=%.3f", label, score)
        await self.emit(
            Event(
                EventType.WAKE_DETECTED,
                payload={
                    "source": "openwakeword",
                    "label": label,
                    "score": score,
                },
            )
        )
        return True

    async def _play_listen_start(self) -> None:
        """Play the ``listen_start`` clip and then flush any mic frames
        captured during playback so the recording doesn't start with our
        own clip audio echoing back.

        The sounddevice mic queue is bounded (~8 chunks * 80 ms ≈ 640 ms);
        without a post-play drain that buffered self-capture would prefix
        the user's question. ``MicSource.drain`` is a no-op when the
        queue isn't backed yet (e.g. ``FakeMic`` in tests).
        """

        if self._clips is None:
            return

        try:
            await self._clips.play("listen_start")
        finally:
            drain = getattr(self._mic, "drain", None)
            if callable(drain):
                dropped = drain()
                if dropped:
                    log.debug(
                        "drained %d mic chunks captured during listen_start",
                        dropped,
                    )

    async def _record_utterance(
        self,
        frames: AsyncIterator[bytes],
        buffered: bytearray,
    ) -> dict[str, object]:
        """Record one spoken question after a wake-word hit."""

        stop_signal = self._stop_signal or UtteranceStopSignal()
        stop_signal.activate()
        pcm = bytearray(buffered)
        buffered.clear()

        started_at = self._clock()
        # voiced_seen only flips True after LISTENING_VOICE_ONSET_FRAMES
        # *consecutive* voiced chunks. voiced_run tracks the current run
        # of voiced chunks and resets on any silent chunk so a single
        # blip (fan, chair creak, listen_start clip echo) can't disarm
        # the no-voice silence-timeout fallback.
        voiced_seen = False
        voiced_run = 0
        onset_required = max(1, int(config.LISTENING_VOICE_ONSET_FRAMES))
        silence_started_at: float | None = None
        rms_debug = config.LISTENING_RMS_DEBUG
        chunk_idx = 0
        log.info("listening for spoken question")
        try:
            async for chunk in frames:
                prepared = self._prepare_chunk(chunk)
                pcm.extend(prepared)

                now = self._clock()
                elapsed = now - started_at
                rms = _pcm16_rms(prepared)
                is_silent = rms < config.LISTENING_SILENCE_RMS_THRESHOLD
                if not is_silent:
                    voiced_run += 1
                    if voiced_run >= onset_required:
                        voiced_seen = True
                        silence_started_at = None
                else:
                    voiced_run = 0
                    if silence_started_at is None:
                        silence_started_at = now

                if rms_debug:
                    log.info(
                        "rms[%03d] t=%.2fs rms=%.0f thr=%.0f %s run=%d seen=%s",
                        chunk_idx,
                        elapsed,
                        rms,
                        config.LISTENING_SILENCE_RMS_THRESHOLD,
                        "silent" if is_silent else "voiced",
                        voiced_run,
                        voiced_seen,
                    )
                chunk_idx += 1

                trailing_silence = (
                    silence_started_at is not None
                    and now - silence_started_at >= config.LISTENING_TRAILING_SILENCE_S
                )
                if (
                    stop_signal.requested
                    or elapsed >= config.LISTENING_MAX_RECORDING_S
                    or (
                        elapsed >= config.LISTENING_MIN_RECORDING_S
                        and voiced_seen
                        and trailing_silence
                    )
                    or (
                        elapsed >= config.LISTENING_SILENCE_TIMEOUT_S
                        and not voiced_seen
                    )
                ):
                    break
        finally:
            stop_signal.deactivate()

        if not voiced_seen:
            log.info("no spoken question detected; returning to idle")
            return {
                "cancel_reason": "no_voice_detected",
                "sample_rate": OPENWAKEWORD_SAMPLE_RATE,
            }

        wav = pcm16_mono_to_wav(bytes(pcm), sample_rate=OPENWAKEWORD_SAMPLE_RATE)
        log.info(
            "spoken question captured: %.2fs pcm=%d wav=%d",
            len(pcm) / (OPENWAKEWORD_SAMPLE_RATE * OPENWAKEWORD_SAMPLE_WIDTH_BYTES),
            len(pcm),
            len(wav),
        )
        return {
            "audio_bytes": wav,
            "filename": "question.wav",
            "content_type": "audio/wav",
            "sample_rate": OPENWAKEWORD_SAMPLE_RATE,
        }


def _best_score(scores: Mapping[str, float]) -> tuple[str | None, float]:
    if not scores:
        return None, 0.0
    label, score = max(scores.items(), key=lambda item: float(item[1]))
    return label, float(score)


def _pcm16_rms(pcm16: bytes) -> float:
    if len(pcm16) < 2:
        return 0.0
    sample_count = len(pcm16) // 2
    samples = memoryview(pcm16[: sample_count * 2]).cast("h")
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


async def _prepend_chunk(
    first: bytes, rest: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    """Yield ``first`` and then drain ``rest``.

    Used by the follow-up listen path so the chunk that triggered the
    bypass flows through ``_record_utterance``'s normal prepare + RMS
    loop instead of being dropped into the un-analyzed buffered prefix.
    """

    yield first
    async for chunk in rest:
        yield chunk


def _resample_pcm16(chunk: bytes, source_rate: int, target_rate: int) -> bytes:
    import numpy as np  # noqa: PLC0415 - optional runtime dependency

    samples = np.frombuffer(chunk, dtype=np.int16)
    if len(samples) == 0:
        return b""

    duration_s = len(samples) / source_rate
    target_count = max(1, round(duration_s * target_rate))
    source_positions = np.linspace(0, len(samples) - 1, num=len(samples))
    target_positions = np.linspace(0, len(samples) - 1, num=target_count)
    resampled = np.interp(target_positions, source_positions, samples).astype(np.int16)
    return resampled.tobytes()
