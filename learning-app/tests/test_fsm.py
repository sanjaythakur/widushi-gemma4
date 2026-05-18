"""Smoke test for the orchestrator FSM.

Walks ``IDLE -> LISTENING -> THINKING -> SPEAKING -> LISTENING`` end
to end with stub services. After ``PLAYBACK_DONE`` the FSM stays in
``LISTENING`` so the user can ask a follow-up without re-saying the
wake word; the wake-word / utterance source is responsible for the
``LISTENING -> IDLE`` silence-timeout fallback (covered separately in
``test_wakeword.py``).

Doesn't require pygame, uvicorn, or any network.
"""

from __future__ import annotations

import asyncio

import pytest

from app.events import Event, EventType
from app.input.utterance import FollowUpListenSignal
from app.modes import IntentRouter, ModeRegistry, TutorMode
from app.orchestrator import Orchestrator
from app.services import Services
from app.services.clips import ClipPlayer
from app.services.gemma import GemmaClient, GemmaReply
from app.services.piper import PiperClient
from app.state import AppState


def _tutor_only_orchestrator(queue, services, *, followup_listen=None) -> Orchestrator:
    """Build an Orchestrator pinned to TutorMode for legacy FSM tests.

    The post-wake default is now ``FreeConvoMode`` which calls a
    different Gemma endpoint; tests that exercise the orchestrator's
    audio/text plumbing (and only that) pin to TutorMode explicitly so
    they remain agnostic to mode-selection changes.
    """

    registry = ModeRegistry()
    registry.register(TutorMode.name, TutorMode)
    router = IntentRouter(default=TutorMode.name, known_modes=registry.names())
    return Orchestrator(
        queue,
        services,
        followup_listen=followup_listen,
        mode_registry=registry,
        intent_router=router,
    )


class RecordingClips(ClipPlayer):
    """ClipPlayer test double recording playback order and timing."""

    def __init__(self, *, latency_s: float = 0.0) -> None:
        super().__init__(stub=True, latency_s=latency_s)
        self.played: list[str] = []
        self.finished: list[str] = []

    async def play(self, clip_id: str) -> None:
        self.played.append(clip_id)
        try:
            await asyncio.sleep(self._latency_s)
        except asyncio.CancelledError:
            raise
        self.finished.append(clip_id)


@pytest.mark.asyncio
async def test_full_turn_round_trip() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    # Tighten the stub latencies so the test is fast.
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    services.clips._latency_s = 0.0  # type: ignore[attr-defined]

    followup_listen = FollowUpListenSignal()
    orch = _tutor_only_orchestrator(queue, services, followup_listen=followup_listen)
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )

    runner = asyncio.create_task(orch.run())

    await queue.put(Event(EventType.WAKE_DETECTED))
    await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hello"}))

    async def _wait_until_followup_listening() -> None:
        # We expect the sequence to land in LISTENING (the SPEAKING
        # -> LISTENING follow-up loopback) within ~2s given the
        # trimmed stub latencies above.
        deadline = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < deadline:
            if (
                len(transitions) >= 4
                and transitions[-1][1] is AppState.LISTENING
                and transitions[-1][2] is EventType.PLAYBACK_DONE
                and orch.state is AppState.LISTENING
            ):
                return
            await asyncio.sleep(0.01)
        raise AssertionError(
            f"FSM did not loop SPEAKING -> LISTENING; saw transitions={transitions}, "
            f"current={orch.state}"
        )

    try:
        await _wait_until_followup_listening()
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    states = [(prev.name, curr.name, ev.name) for prev, curr, ev in transitions]
    assert states == [
        ("IDLE", "LISTENING", "WAKE_DETECTED"),
        ("LISTENING", "THINKING", "UTTERANCE_END"),
        ("THINKING", "SPEAKING", "REPLY_READY"),
        ("SPEAKING", "LISTENING", "PLAYBACK_DONE"),
    ]
    # The PLAYBACK_DONE side effect must have asked the wake-word
    # source to open a follow-up listen.
    assert followup_listen.pending is True


@pytest.mark.asyncio
async def test_playback_done_without_followup_signal_stays_in_listening() -> None:
    """Without a follow-up signal wired in, the FSM still loops to
    LISTENING on PLAYBACK_DONE; it just won't trigger any source."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    services.clips._latency_s = 0.0  # type: ignore[attr-defined]

    orch = _tutor_only_orchestrator(queue, services)  # no followup_listen
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if (
                transitions
                and transitions[-1][1] is AppState.LISTENING
                and transitions[-1][2] is EventType.PLAYBACK_DONE
            ):
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    assert orch.state is AppState.LISTENING
    assert transitions[-1] == (AppState.SPEAKING, AppState.LISTENING, EventType.PLAYBACK_DONE)


@pytest.mark.asyncio
async def test_audio_utterance_uses_gemma_listen_audio() -> None:
    class RecordingGemma(GemmaClient):
        def __init__(self) -> None:
            super().__init__(stub=True)
            self.audio_seen: bytes | None = None

        async def listen_audio_with_tts(
            self,
            audio_bytes: bytes,
            *,
            filename: str = "question.wav",
            content_type: str = "audio/wav",
        ) -> GemmaReply:
            self.audio_seen = audio_bytes
            assert filename == "question.wav"
            assert content_type == "audio/wav"
            return GemmaReply(
                text="audio answer",
                audio_bytes=b"answer wav",
                audio_duration_ms=10.0,
                voice="warm-academic",
            )

    class RecordingPiper(PiperClient):
        def __init__(self) -> None:
            super().__init__(stub=True)
            self.wav_seen: bytes | None = None
            self.duration_seen: float | None = None

        async def play(
            self,
            wav_bytes: bytes,
            *,
            duration_ms: float | None = None,
        ) -> None:
            self.wav_seen = wav_bytes
            self.duration_seen = duration_ms

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    gemma = RecordingGemma()
    piper = RecordingPiper()
    services.gemma = gemma
    services.piper = piper
    orch = _tutor_only_orchestrator(queue, services)
    replies: list[str] = []
    orch.add_listener(
        lambda _prev, _curr, ev: replies.append(ev.payload["text"])
        if ev.type is EventType.REPLY_READY
        else None
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(
            Event(
                EventType.UTTERANCE_END,
                payload={
                    "audio_bytes": b"wav bytes",
                    "filename": "question.wav",
                    "content_type": "audio/wav",
                },
            )
        )
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if replies and piper.wav_seen is not None:
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    assert gemma.audio_seen == b"wav bytes"
    assert piper.wav_seen == b"answer wav"
    assert piper.duration_seen == 10.0
    assert replies == ["audio answer"]


@pytest.mark.asyncio
async def test_tts_failure_cancels_turn_before_speaking() -> None:
    class FailingGemma(GemmaClient):
        def __init__(self) -> None:
            super().__init__(stub=True)

        async def complete_with_tts(self, prompt: str) -> GemmaReply:
            assert prompt == "hello"
            raise RuntimeError("TTS engine not ready")

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma = FailingGemma()
    orch = _tutor_only_orchestrator(queue, services)
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hello"}))
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if transitions and transitions[-1][1] is AppState.IDLE:
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    states = [(prev.name, curr.name, ev.name) for prev, curr, ev in transitions]
    assert states == [
        ("IDLE", "LISTENING", "WAKE_DETECTED"),
        ("LISTENING", "THINKING", "UTTERANCE_END"),
        ("THINKING", "IDLE", "CANCEL"),
    ]


@pytest.mark.asyncio
async def test_cancel_returns_to_idle_from_anywhere() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma._latency_s = 0.5  # type: ignore[attr-defined]

    orch = _tutor_only_orchestrator(queue, services)
    runner = asyncio.create_task(orch.run())

    await queue.put(Event(EventType.WAKE_DETECTED))
    # Give the FSM a beat to land in LISTENING.
    await asyncio.sleep(0.05)
    assert orch.state is AppState.LISTENING

    await queue.put(Event(EventType.UTTERANCE_END))
    await asyncio.sleep(0.05)
    assert orch.state is AppState.THINKING

    await queue.put(Event(EventType.CANCEL))
    await asyncio.sleep(0.05)
    assert orch.state is AppState.IDLE

    await queue.put(Event(EventType.SHUTDOWN))
    await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_unknown_transitions_are_ignored() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    orch = _tutor_only_orchestrator(queue, Services.stubs())
    runner = asyncio.create_task(orch.run())

    # PLAYBACK_DONE in IDLE has no mapping; must be ignored, state stays.
    await queue.put(Event(EventType.PLAYBACK_DONE))
    await asyncio.sleep(0.05)
    assert orch.state is AppState.IDLE

    await queue.put(Event(EventType.SHUTDOWN))
    await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_thinking_plays_wait_thinking_for_fresh_turn() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    clips = RecordingClips(latency_s=0.0)
    services.clips = clips

    orch = _tutor_only_orchestrator(queue, services)
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(
            Event(
                EventType.UTTERANCE_END,
                payload={"prompt": "hi", "followup": False},
            )
        )
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.state is AppState.LISTENING and "wait_thinking" in clips.played:
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    assert clips.played == ["wait_thinking"]
    assert "wait_checking" not in clips.played


@pytest.mark.asyncio
async def test_thinking_plays_wait_checking_for_followup_turn() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    clips = RecordingClips(latency_s=0.0)
    services.clips = clips

    orch = _tutor_only_orchestrator(queue, services)
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(
            Event(
                EventType.UTTERANCE_END,
                payload={"prompt": "hi", "followup": True},
            )
        )
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.state is AppState.LISTENING and "wait_checking" in clips.played:
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    assert clips.played == ["wait_checking"]
    assert "wait_thinking" not in clips.played


@pytest.mark.asyncio
async def test_reply_ready_waits_for_clip_to_finish() -> None:
    """When the ack clip outlasts Gemma, REPLY_READY must not fire until
    the clip is fully played -- otherwise SPEAKING audio talks over the
    canned 'let me think' line.
    """

    class FastGemma(GemmaClient):
        def __init__(self) -> None:
            super().__init__(stub=True)

        async def complete_with_tts(self, prompt: str) -> GemmaReply:
            await asyncio.sleep(0.01)
            return GemmaReply(
                text="ok",
                audio_bytes=b"answer wav",
                audio_duration_ms=10.0,
                voice="warm-academic",
            )

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma = FastGemma()
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    clips = RecordingClips(latency_s=0.2)
    services.clips = clips

    orch = _tutor_only_orchestrator(queue, services)

    reply_ready_at: list[float] = []
    orch.add_listener(
        lambda _prev, _curr, ev: reply_ready_at.append(asyncio.get_event_loop().time())
        if ev.type is EventType.REPLY_READY
        else None
    )

    runner = asyncio.create_task(orch.run())
    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(
            Event(
                EventType.UTTERANCE_END,
                payload={"prompt": "hi", "followup": False},
            )
        )
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if reply_ready_at and clips.finished == ["wait_thinking"]:
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    assert clips.finished == ["wait_thinking"], (
        "ack clip must finish before REPLY_READY"
    )
    assert reply_ready_at, "REPLY_READY never observed"


@pytest.mark.asyncio
async def test_gemma_failure_cancels_in_flight_clip() -> None:
    """If Gemma raises, the in-flight ack clip must be cancelled and
    the turn must CANCEL back to IDLE without REPLY_READY firing.
    """

    class FailingGemma(GemmaClient):
        def __init__(self) -> None:
            super().__init__(stub=True)

        async def complete_with_tts(self, prompt: str) -> GemmaReply:
            await asyncio.sleep(0.01)
            raise RuntimeError("gemma boom")

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = Services.stubs()
    services.gemma = FailingGemma()
    clips = RecordingClips(latency_s=2.0)  # would outlast the test if not cancelled
    services.clips = clips

    orch = _tutor_only_orchestrator(queue, services)
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )

    runner = asyncio.create_task(orch.run())
    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if transitions and transitions[-1][1] is AppState.IDLE:
                break
            await asyncio.sleep(0.01)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)

    assert clips.played == ["wait_thinking"]
    assert "wait_thinking" not in clips.finished, (
        "Gemma failure must cancel the in-flight clip"
    )
    states = [(prev.name, curr.name, ev.name) for prev, curr, ev in transitions]
    assert states == [
        ("IDLE", "LISTENING", "WAKE_DETECTED"),
        ("LISTENING", "THINKING", "UTTERANCE_END"),
        ("THINKING", "IDLE", "CANCEL"),
    ]
