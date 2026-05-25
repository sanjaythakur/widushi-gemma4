"""End-to-end tests for the new mid-session mode-change machinery.

Covers:

* Orchestrator handles ``MODE_CHANGE`` while in ``SESSION`` and swaps
  the active mode without dropping to ``SystemState.IDLE``.
* ``FreeConvoMode.run_thinking`` schedules a swap to VoiceMirror when
  Gemma reports ``start_learning=true`` and does not when it is false.
* ``VoiceMirrorMode`` increments its turn counter only on praise /
  correct, and emits ``MODE_CHANGE`` to ``vision`` after
  ``target_turns`` successful turns.
* ``VisionMode.run_thinking`` calls ``camera.snapshot`` and
  ``vision_teach_object`` and emits ``MODE_CHANGE`` to ``roleplay``
  after ``target_turns`` taught turns.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.events import Event, EventType
from app.modes import (
    FreeConvoMode,
    IntentRouter,
    Mode,
    ModeRegistry,
    RolePlayMode,
    VisionMode,
    VoiceMirrorMode,
)
from app.orchestrator import Orchestrator
from app.services import Services
from app.services.gemma import (
    FreeConvoReply,
    GemmaReply,
    ScoreResult,
    TeachReply,
    WordSuggestion,
)
from app.state import AppState
from app.system_state import SystemState

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _OrchStub:
    """Bare-bones orchestrator stand-in for Mode.run_thinking unit tests.

    Only exposes the attributes the modes touch (``services`` plus a
    ``queue`` accepted for forward-compat).
    """

    def __init__(self, gemma: Any, camera: Any | None = None) -> None:
        ns_services = type(
            "S", (), {"gemma": gemma, "camera": camera}
        )()
        self.services = ns_services
        self.queue: asyncio.Queue[Event] = asyncio.Queue()


class _SilentGemma:
    """Concrete-enough Gemma client for unit tests.

    Each method records its calls so tests can assert on them, and
    returns deterministic stub payloads.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.free_convo_reply = FreeConvoReply(
            text="ok",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            start_learning=False,
            transcript="hello",
        )
        self.suggestions: list[WordSuggestion] = []
        self.score_result = ScoreResult(
            text="praise",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            target_word="apple",
            transcript="apple",
            verdict="praise",
        )
        self.teach_reply = TeachReply(
            text="Yes, this is milk. Say: I drink milk.",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            object="milk",
            transcript="milk",
        )
        self.complete_reply = GemmaReply(
            text="hi there",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
        )

    async def free_convo_turn(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "turn.wav",
        content_type: str = "audio/wav",
    ) -> FreeConvoReply:
        self.calls.append({"kind": "free_convo_turn", "bytes": audio_bytes})
        return self.free_convo_reply

    async def voice_mirror_suggest(
        self,
        *,
        history: list[str] | None = None,
        level: str | None = None,
    ) -> WordSuggestion:
        self.calls.append({"kind": "suggest", "history": list(history or [])})
        if self.suggestions:
            return self.suggestions.pop(0)
        return WordSuggestion(
            text="Try saying: apple.",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            word="apple",
            example_sentence="I like apples.",
        )

    async def voice_mirror_score(
        self,
        audio_bytes: bytes,
        *,
        target_word: str,
        filename: str = "attempt.wav",
        content_type: str = "audio/wav",
    ) -> ScoreResult:
        self.calls.append(
            {"kind": "score", "target_word": target_word, "bytes": audio_bytes}
        )
        return self.score_result

    async def vision_teach_object(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        *,
        image_filename: str = "frame.jpg",
        image_content_type: str = "image/jpeg",
        audio_filename: str = "guess.wav",
        audio_content_type: str = "audio/wav",
    ) -> TeachReply:
        self.calls.append(
            {
                "kind": "teach",
                "image_bytes": image_bytes,
                "audio_bytes": audio_bytes,
            }
        )
        return self.teach_reply

    async def complete_with_tts(self, prompt: str) -> GemmaReply:
        self.calls.append({"kind": "complete", "prompt": prompt})
        return self.complete_reply

    async def listen_audio_with_tts(self, audio_bytes: bytes, **_: Any) -> GemmaReply:
        self.calls.append({"kind": "listen_audio", "bytes": audio_bytes})
        return self.complete_reply


class _FakeCamera:
    def __init__(self, payload: bytes = b"\xff\xd8\xff\xe0jpeg") -> None:
        self.payload = payload
        self.snap_count = 0

    async def snapshot(self) -> bytes:
        self.snap_count += 1
        return self.payload


# ---------------------------------------------------------------------------
# MODE_CHANGE orchestrator handling
# ---------------------------------------------------------------------------


class _GreetingMode(Mode):
    """Mode whose on_enter returns a fixed GemmaReply (an intro line)."""

    name = "greeting"

    async def on_enter(self, orch):  # type: ignore[override]
        await super().on_enter(orch)
        return GemmaReply(
            text="hello from greeting",
            audio_bytes=b"intro wav",
            audio_duration_ms=5.0,
            voice="warm-academic",
        )


class _SilentModeA(Mode):
    """Mode whose on_enter returns None (no intro)."""

    name = "silent_a"


class _SilentModeB(Mode):
    name = "silent_b"


def _stub_services_fast() -> Services:
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    services.clips._latency_s = 0.0  # type: ignore[attr-defined]
    return services


@pytest.mark.asyncio
async def test_mode_change_swaps_active_mode_and_stays_in_session() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    registry = ModeRegistry()
    registry.register("silent_a", _SilentModeA)
    registry.register("silent_b", _SilentModeB)
    router = IntentRouter(default="silent_a", known_modes=registry.names())

    orch = Orchestrator(
        queue, services, mode_registry=registry, intent_router=router
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.active_mode is not None:
                break
            await asyncio.sleep(0.01)
        first = orch.active_mode
        assert first is not None and first.name == "silent_a"
        assert orch.system_state is SystemState.SESSION

        await queue.put(
            Event(
                EventType.MODE_CHANGE,
                payload={"target_mode": "silent_b", "auto_listen": True},
            )
        )

        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.active_mode is not first:
                break
            await asyncio.sleep(0.01)
        assert orch.active_mode is not None
        assert orch.active_mode.name == "silent_b"
        # Session must persist across the swap.
        assert orch.system_state is SystemState.SESSION
        assert orch.state is AppState.LISTENING
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_mode_change_with_intro_plays_greeting_then_listens() -> None:
    """An ``on_enter`` returning a GemmaReply must be spoken before listening."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    registry = ModeRegistry()
    registry.register("silent_a", _SilentModeA)
    registry.register(_GreetingMode.name, _GreetingMode)
    router = IntentRouter(default="silent_a", known_modes=registry.names())

    orch = Orchestrator(
        queue, services, mode_registry=registry, intent_router=router
    )
    transitions: list[tuple[AppState, AppState, EventType]] = []
    orch.add_listener(
        lambda prev, curr, ev: transitions.append((prev, curr, ev.type))
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(
            Event(
                EventType.MODE_CHANGE,
                payload={"target_mode": "greeting"},
            )
        )

        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            # After the intro REPLY_READY is processed we move to SPEAKING.
            if any(curr is AppState.SPEAKING for _, curr, _ in transitions):
                break
            await asyncio.sleep(0.01)
        assert any(curr is AppState.SPEAKING for _, curr, _ in transitions), (
            f"never spoke greeting; transitions={transitions}"
        )
        assert isinstance(orch.active_mode, _GreetingMode)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


# ---------------------------------------------------------------------------
# FreeConvoMode -> start_learning gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_free_convo_schedules_mode_change_when_start_learning_true() -> None:
    mode = FreeConvoMode()
    await mode.on_enter(_OrchStub(_SilentGemma()))

    gemma = _SilentGemma()
    gemma.free_convo_reply = FreeConvoReply(
        text="Great, let's practice!",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        start_learning=True,
        transcript="i want to learn english",
    )
    orch = _OrchStub(gemma)

    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"abc"}),
    )

    assert getattr(reply, "start_learning", False) is True
    pending = mode.pop_pending_mode_change()
    assert pending is not None
    assert pending["target_mode"] == "voice_mirror"
    assert pending["auto_listen"] is False


@pytest.mark.asyncio
async def test_free_convo_does_not_schedule_mode_change_when_start_learning_false() -> None:
    mode = FreeConvoMode()
    await mode.on_enter(_OrchStub(_SilentGemma()))

    gemma = _SilentGemma()
    gemma.free_convo_reply = FreeConvoReply(
        text="Sure, what do you want to talk about?",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        start_learning=False,
        transcript="how are you",
    )
    orch = _OrchStub(gemma)

    await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"abc"}),
    )

    assert mode.pop_pending_mode_change() is None


# ---------------------------------------------------------------------------
# VoiceMirrorMode -> turn counting + handoff to vision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voice_mirror_on_enter_suggests_a_word() -> None:
    mode = VoiceMirrorMode()
    gemma = _SilentGemma()
    orch = _OrchStub(gemma)

    intro = await mode.on_enter(orch)  # type: ignore[arg-type]
    assert isinstance(intro, WordSuggestion)
    assert intro.word == "apple"
    assert mode.current_word == "apple"
    assert mode.turns_done == 0
    assert any(c["kind"] == "suggest" for c in gemma.calls)


@pytest.mark.asyncio
async def test_voice_mirror_retry_does_not_increment_turns_or_swap() -> None:
    mode = VoiceMirrorMode()
    gemma = _SilentGemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]

    gemma.score_result = ScoreResult(
        text="Let's try 'apple' again.",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="apple",
        transcript="ah",
        verdict="retry",
    )

    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"attempt"}),
    )
    assert getattr(reply, "verdict", None) == "retry"
    assert mode.turns_done == 0
    assert mode.current_word == "apple"
    assert mode.pop_pending_mode_change() is None


@pytest.mark.asyncio
async def test_voice_mirror_swaps_to_vision_after_target_turns() -> None:
    mode = VoiceMirrorMode()
    assert mode.target_turns == 2
    gemma = _SilentGemma()
    # Pre-seed the second word the mode will ask for after the first praise.
    gemma.suggestions = [
        WordSuggestion(
            text="Try saying: apple.",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            word="apple",
            example_sentence="I like apples.",
        ),
        WordSuggestion(
            text="Now try saying: banana.",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            word="banana",
            example_sentence="I like bananas.",
        ),
    ]
    orch = _OrchStub(gemma)

    await mode.on_enter(orch)  # type: ignore[arg-type]
    assert mode.current_word == "apple"

    gemma.score_result = ScoreResult(
        text="Nice apple!",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="apple",
        transcript="apple",
        verdict="praise",
    )
    await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"a1"}),
    )
    assert mode.turns_done == 1
    assert mode.current_word == "banana"
    assert mode.pop_pending_mode_change() is None

    gemma.score_result = ScoreResult(
        text="Great banana!",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="banana",
        transcript="banana",
        verdict="praise",
    )
    await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"a2"}),
    )
    assert mode.turns_done == 2
    pending = mode.pop_pending_mode_change()
    assert pending is not None
    assert pending["target_mode"] == "vision"


# ---------------------------------------------------------------------------
# VisionMode -> camera + teach + handoff to roleplay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vision_run_thinking_calls_camera_and_teach_object() -> None:
    mode = VisionMode()
    gemma = _SilentGemma()
    camera = _FakeCamera(payload=b"my-jpeg")
    orch = _OrchStub(gemma, camera=camera)
    await mode.on_enter(orch)  # type: ignore[arg-type]

    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"guess"}),
    )

    assert camera.snap_count == 1
    teach_calls = [c for c in gemma.calls if c["kind"] == "teach"]
    assert len(teach_calls) == 1
    assert teach_calls[0]["image_bytes"] == b"my-jpeg"
    assert teach_calls[0]["audio_bytes"] == b"guess"
    assert isinstance(reply, TeachReply)
    assert mode.turns_done == 1
    assert mode.pop_pending_mode_change() is None


@pytest.mark.asyncio
async def test_vision_swaps_to_roleplay_after_target_turns() -> None:
    mode = VisionMode()
    assert mode.target_turns == 2
    gemma = _SilentGemma()
    camera = _FakeCamera()
    orch = _OrchStub(gemma, camera=camera)
    await mode.on_enter(orch)  # type: ignore[arg-type]

    for _ in range(mode.target_turns):
        await mode.run_thinking(
            orch,  # type: ignore[arg-type]
            Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"guess"}),
        )

    pending = mode.pop_pending_mode_change()
    assert pending is not None
    assert pending["target_mode"] == "roleplay"


# ---------------------------------------------------------------------------
# RolePlayMode smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_roleplay_on_enter_speaks_scenario_intro() -> None:
    mode = RolePlayMode()
    gemma = _SilentGemma()
    orch = _OrchStub(gemma)

    intro = await mode.on_enter(orch)  # type: ignore[arg-type]
    assert isinstance(intro, GemmaReply)
    assert any(c["kind"] == "complete" for c in gemma.calls)
    assert mode.scenario in {"shopkeeper", "doctor"}
    assert mode.transcript  # at least the NPC's opening line
