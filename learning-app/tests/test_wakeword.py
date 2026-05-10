from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from app.events import Event, EventType
from app.input.utterance import FollowUpListenSignal, UtteranceStopSignal
from app.input.wakeword import OPENWAKEWORD_FRAME_BYTES, WakeWordSource


class FakeMic:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        sample_rate: int = 16000,
        delay_s: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._chunks = chunks
        self._delay_s = delay_s

    async def frames(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_s)
            yield chunk


class FakeDetector:
    def __init__(self, scores: list[Mapping[str, float]]) -> None:
        self._scores = scores
        self.frames_seen = 0

    def predict(self, pcm16: bytes) -> Mapping[str, float]:
        self.frames_seen += 1
        assert len(pcm16) == OPENWAKEWORD_FRAME_BYTES
        return self._scores.pop(0) if self._scores else {"vDu_shee": 0.0}


async def _run_source(source: WakeWordSource) -> None:
    task = asyncio.create_task(source.run())
    try:
        await asyncio.wait_for(task, timeout=1)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _model_file(tmp_path: Path) -> Path:
    model = tmp_path / "vDu_shee.onnx"
    model.write_bytes(b"fake model")
    return model


@pytest.mark.asyncio
async def test_wakeword_source_emits_wake_detected_above_threshold(
    tmp_path: Path,
) -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector([{"vDu_shee": 0.1}, {"vDu_shee": 0.8}])
    source = WakeWordSource(
        queue,
        mic=FakeMic([b"\x00" * OPENWAKEWORD_FRAME_BYTES] * 2),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        detector_factory=lambda _: detector,
    )

    await _run_source(source)

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event.type is EventType.WAKE_DETECTED
    assert event.payload["source"] == "openwakeword"
    assert event.payload["label"] == "vDu_shee"
    assert event.payload["score"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_wakeword_source_debounces_repeated_detections(
    tmp_path: Path,
) -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector(
        [
            {"vDu_shee": 0.9},
            {"vDu_shee": 0.91},
            {"vDu_shee": 0.92},
        ]
    )
    clock_values = iter([10.0, 10.5, 11.0, 11.5])
    source = WakeWordSource(
        queue,
        mic=FakeMic([b"\x00" * OPENWAKEWORD_FRAME_BYTES] * 3),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=2.0,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event.type is EventType.WAKE_DETECTED
    cancel = await asyncio.wait_for(queue.get(), timeout=1)
    assert cancel.type is EventType.CANCEL
    assert cancel.payload["cancel_reason"] == "no_voice_detected"


@pytest.mark.asyncio
async def test_wakeword_source_records_question_after_wake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.LISTENING_MIN_RECORDING_S", 0.0)
    monkeypatch.setattr("app.config.LISTENING_TRAILING_SILENCE_S", 0.05)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector([{"vDu_shee": 0.9}])
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    source = WakeWordSource(
        queue,
        mic=FakeMic([silence, voice, silence, silence]),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    wake = await asyncio.wait_for(queue.get(), timeout=1)
    utterance = await asyncio.wait_for(queue.get(), timeout=1)
    assert wake.type is EventType.WAKE_DETECTED
    assert utterance.type is EventType.UTTERANCE_END
    assert utterance.payload["filename"] == "question.wav"
    assert utterance.payload["content_type"] == "audio/wav"
    audio_bytes = utterance.payload["audio_bytes"]
    assert isinstance(audio_bytes, bytes)
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000


@pytest.mark.asyncio
async def test_initial_silence_after_wake_cancels_to_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 0.2)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector([{"vDu_shee": 0.9}])
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3])
    source = WakeWordSource(
        queue,
        mic=FakeMic([silence, silence, silence]),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    wake = await asyncio.wait_for(queue.get(), timeout=1)
    cancel = await asyncio.wait_for(queue.get(), timeout=1)
    assert wake.type is EventType.WAKE_DETECTED
    assert cancel.type is EventType.CANCEL
    assert cancel.payload["cancel_reason"] == "no_voice_detected"


@pytest.mark.asyncio
async def test_followup_signal_records_without_wake_word(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the FSM raises the follow-up signal (PLAYBACK_DONE side
    effect), the wake-word source must skip wake-word detection on the
    next mic chunk and start recording immediately so the user can ask
    a follow-up without re-saying "Widushi"."""

    monkeypatch.setattr("app.config.LISTENING_MIN_RECORDING_S", 0.0)
    monkeypatch.setattr("app.config.LISTENING_TRAILING_SILENCE_S", 0.05)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 5.0)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    followup = FollowUpListenSignal()
    followup.request()  # FSM has just emitted PLAYBACK_DONE

    detector = FakeDetector([{"vDu_shee": 0.0}] * 4)
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    # Mic supplies exactly what the recorder will consume so no extra
    # chunk leaks into the detector after the follow-up listen ends.
    source = WakeWordSource(
        queue,
        mic=FakeMic([voice, silence, silence]),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        followup_signal=followup,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    utterance = await asyncio.wait_for(queue.get(), timeout=1)
    assert utterance.type is EventType.UTTERANCE_END, (
        "follow-up listen should bypass wake-word detection and record immediately"
    )
    assert utterance.payload["filename"] == "question.wav"
    assert isinstance(utterance.payload["audio_bytes"], bytes)
    assert detector.frames_seen == 0, (
        "wake-word detector must not be invoked on the chunk that opens a follow-up listen"
    )


@pytest.mark.asyncio
async def test_followup_signal_silence_emits_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the follow-up listen never hears speech, the wake-word source
    must emit CANCEL with `cancel_reason='no_voice_detected'` so the
    orchestrator's wildcard CANCEL handler drops the FSM back to IDLE."""

    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 0.2)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    followup = FollowUpListenSignal()
    followup.request()

    detector = FakeDetector([{"vDu_shee": 0.0}] * 4)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    source = WakeWordSource(
        queue,
        mic=FakeMic([silence, silence, silence, silence]),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        followup_signal=followup,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    cancel = await asyncio.wait_for(queue.get(), timeout=1)
    assert cancel.type is EventType.CANCEL
    assert cancel.payload["cancel_reason"] == "no_voice_detected"


@pytest.mark.asyncio
async def test_manual_stop_finishes_active_recording(tmp_path: Path) -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    stop_signal = UtteranceStopSignal()
    detector = FakeDetector([{"vDu_shee": 0.9}])
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    source = WakeWordSource(
        queue,
        mic=FakeMic([voice] * 100, delay_s=0.01),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        stop_signal=stop_signal,
        detector_factory=lambda _: detector,
    )

    task = asyncio.create_task(source.run())
    try:
        wake = await asyncio.wait_for(queue.get(), timeout=1)
        assert wake.type is EventType.WAKE_DETECTED
        await asyncio.sleep(0.02)
        assert stop_signal.request_stop()
        utterance = await asyncio.wait_for(queue.get(), timeout=1)
        assert utterance.type is EventType.UTTERANCE_END
        assert utterance.payload["audio_bytes"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

