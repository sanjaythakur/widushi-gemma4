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
from app.orchestrator import Orchestrator
from app.services import Services
from app.services.clips import ClipPlayer
from app.state import AppState


class RecordingClips(ClipPlayer):
    """ClipPlayer test double recording playback order."""

    def __init__(self, *, latency_s: float = 0.0) -> None:
        super().__init__(stub=True, latency_s=latency_s)
        self.played: list[str] = []

    async def play(self, clip_id: str) -> None:
        self.played.append(clip_id)
        await asyncio.sleep(self._latency_s)


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
    monkeypatch.setattr("app.config.LISTENING_VOICE_ONSET_FRAMES", 2)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector([{"vDu_shee": 0.9}])
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    source = WakeWordSource(
        queue,
        # Two consecutive voice frames are required to trip voiced_seen
        # under the default LISTENING_VOICE_ONSET_FRAMES so a single
        # ambient-noise blip can't be mistaken for speech.
        mic=FakeMic([silence, voice, voice, silence, silence]),  # type: ignore[arg-type]
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
async def test_listen_start_clip_plays_before_recording_on_wake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a clip player is wired and the wake word fires, the source
    must play ``listen_start`` before starting to record, and tag the
    UTTERANCE_END payload with ``followup=False``.
    """

    monkeypatch.setattr("app.config.LISTENING_MIN_RECORDING_S", 0.0)
    monkeypatch.setattr("app.config.LISTENING_TRAILING_SILENCE_S", 0.05)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)
    monkeypatch.setattr("app.config.LISTENING_VOICE_ONSET_FRAMES", 2)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector([{"vDu_shee": 0.9}])
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    clips = RecordingClips(latency_s=0.0)
    source = WakeWordSource(
        queue,
        # Chunk 0 drives detection; clip plays (FakeMic.drain is a no-op
        # because there's no underlying queue); chunks 1+ feed the
        # recorder. Two consecutive voice frames are required for the
        # onset filter to declare voiced_seen.
        mic=FakeMic([silence, voice, voice, silence, silence]),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        clips=clips,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    wake = await asyncio.wait_for(queue.get(), timeout=1)
    utterance = await asyncio.wait_for(queue.get(), timeout=1)
    assert wake.type is EventType.WAKE_DETECTED
    assert utterance.type is EventType.UTTERANCE_END
    assert clips.played == ["listen_start"]
    assert utterance.payload["followup"] is False


@pytest.mark.asyncio
async def test_followup_path_skips_listen_start_and_tags_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The follow-up loopback path must NOT play ``listen_start`` (the
    user is mid-conversation) and must tag UTTERANCE_END with
    ``followup=True`` so the FSM picks ``wait_checking``.
    """

    monkeypatch.setattr("app.config.LISTENING_MIN_RECORDING_S", 0.0)
    monkeypatch.setattr("app.config.LISTENING_TRAILING_SILENCE_S", 0.05)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 5.0)
    monkeypatch.setattr("app.config.LISTENING_VOICE_ONSET_FRAMES", 2)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    followup = FollowUpListenSignal()
    followup.request()

    detector = FakeDetector([{"vDu_shee": 0.0}] * 4)
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    clips = RecordingClips(latency_s=0.0)
    source = WakeWordSource(
        queue,
        mic=FakeMic([voice, voice, silence, silence]),  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        followup_signal=followup,
        clips=clips,
        detector_factory=lambda _: detector,
        clock=lambda: next(clock_values),
    )

    await _run_source(source)

    utterance = await asyncio.wait_for(queue.get(), timeout=1)
    assert utterance.type is EventType.UTTERANCE_END
    assert utterance.payload["followup"] is True
    assert clips.played == [], "follow-up path must not play listen_start"


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
    monkeypatch.setattr("app.config.LISTENING_VOICE_ONSET_FRAMES", 2)

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
        mic=FakeMic([voice, voice, silence, silence]),  # type: ignore[arg-type]
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
async def test_listening_silence_drops_fsm_to_idle_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end proof that the LISTENING -> IDLE silence-timeout path
    works through a real Orchestrator: WAKE_DETECTED moves the FSM to
    LISTENING, the recorder hears only silence for
    LISTENING_SILENCE_TIMEOUT_S, the wake-word source emits CANCEL, and
    the orchestrator's wildcard CANCEL handler lands the FSM back in
    IDLE so the user has to re-say "Widushi".
    """

    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 0.3)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    orch = Orchestrator(queue, Services.stubs())
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )

    detector = FakeDetector([{"vDu_shee": 0.9}])
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    # 1 wake-word frame + ~600 ms of silence (> 300 ms timeout).
    mic = FakeMic([silence] * 31, delay_s=0.02)
    source = WakeWordSource(
        queue,
        mic=mic,  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        detector_factory=lambda _: detector,
    )

    fsm_task = asyncio.create_task(orch.run())
    source_task = asyncio.create_task(source.run())

    try:
        deadline = asyncio.get_event_loop().time() + 3.0
        while asyncio.get_event_loop().time() < deadline:
            if (
                len(transitions) >= 2
                and transitions[-1][1] is AppState.IDLE
                and transitions[-1][2] is EventType.CANCEL
            ):
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.gather(fsm_task, source_task, return_exceptions=True)

    states = [(prev.name, curr.name, ev.name) for prev, curr, ev in transitions]
    assert states == [
        ("IDLE", "LISTENING", "WAKE_DETECTED"),
        ("LISTENING", "IDLE", "CANCEL"),
    ], f"unexpected transitions: {states}"
    assert orch.state is AppState.IDLE


@pytest.mark.asyncio
async def test_followup_listening_silence_drops_fsm_to_idle_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end proof for the SPEAKING -> LISTENING follow-up loopback:
    after PLAYBACK_DONE puts the FSM in LISTENING and raises the
    follow-up signal, silence in the recorder must emit CANCEL and drop
    the FSM back to IDLE so the wake word is required again.
    """

    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 0.3)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]

    followup_listen = FollowUpListenSignal()
    orch = Orchestrator(queue, services, followup_listen=followup_listen)
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )

    # Detector always returns 0 -- the only way into LISTENING in this
    # test is via the FSM raising the follow-up signal after PLAYBACK_DONE.
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    detector = FakeDetector([{"vDu_shee": 0.0}] * 500)
    mic = FakeMic([silence] * 500, delay_s=0.01)
    source = WakeWordSource(
        queue,
        mic=mic,  # type: ignore[arg-type]
        model_path=_model_file(tmp_path),
        enabled=True,
        threshold=0.5,
        debounce_s=0,
        followup_signal=followup_listen,
        detector_factory=lambda _: detector,
    )

    fsm_task = asyncio.create_task(orch.run())
    source_task = asyncio.create_task(source.run())

    try:
        # Drive a full turn through the FSM: WAKE_DETECTED + UTTERANCE_END
        # walks IDLE -> LISTENING -> THINKING -> SPEAKING -> LISTENING
        # (the SPEAKING -> LISTENING follow-up loopback). The loopback
        # raises the follow-up signal, which the wake-word source picks
        # up on its next mic chunk and starts a silent recording that
        # times out -> CANCEL -> IDLE.
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))

        deadline = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < deadline:
            if (
                transitions
                and transitions[-1][1] is AppState.IDLE
                and transitions[-1][2] is EventType.CANCEL
            ):
                break
            await asyncio.sleep(0.02)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        source_task.cancel()
        await asyncio.gather(fsm_task, source_task, return_exceptions=True)

    states = [(prev.name, curr.name, ev.name) for prev, curr, ev in transitions]
    assert states == [
        ("IDLE", "LISTENING", "WAKE_DETECTED"),
        ("LISTENING", "THINKING", "UTTERANCE_END"),
        ("THINKING", "SPEAKING", "REPLY_READY"),
        ("SPEAKING", "LISTENING", "PLAYBACK_DONE"),
        ("LISTENING", "IDLE", "CANCEL"),
    ], f"unexpected transitions: {states}"
    assert orch.state is AppState.IDLE


@pytest.mark.asyncio
async def test_single_noise_blip_does_not_disarm_silence_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single isolated chunk above the RMS threshold (e.g. a fan blip,
    a chair creak, the tail of the ``listen_start`` clip echoing back
    through the mic) must NOT count as "voice detected" — otherwise the
    LISTENING_SILENCE_TIMEOUT_S no-voice -> CANCEL fallback never fires
    and the FSM marches straight to THINKING even though the user said
    nothing. The voice-onset filter requires
    ``LISTENING_VOICE_ONSET_FRAMES`` consecutive voiced chunks.
    """

    monkeypatch.setattr("app.config.LISTENING_SILENCE_TIMEOUT_S", 0.3)
    monkeypatch.setattr("app.config.LISTENING_TRAILING_SILENCE_S", 0.05)
    monkeypatch.setattr("app.config.LISTENING_SILENCE_RMS_THRESHOLD", 500.0)
    monkeypatch.setattr("app.config.LISTENING_VOICE_ONSET_FRAMES", 2)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    detector = FakeDetector([{"vDu_shee": 0.9}])
    voice = b"\xff\x7f" * (OPENWAKEWORD_FRAME_BYTES // 2)
    silence = b"\x00" * OPENWAKEWORD_FRAME_BYTES
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    source = WakeWordSource(
        queue,
        # Wake-word frame, then a single noise blip surrounded by
        # silence. Without the onset filter that one blip would flip
        # voiced_seen=True and the recorder would wait for trailing
        # silence -> UTTERANCE_END instead of CANCEL.
        mic=FakeMic([silence, silence, voice, silence, silence, silence]),  # type: ignore[arg-type]
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
    assert cancel.type is EventType.CANCEL, (
        "single noise blip must not be treated as user speech"
    )
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

