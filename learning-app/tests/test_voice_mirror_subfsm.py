"""Sub-FSM unit tests for :class:`VoiceMirrorMode`.

Each test exercises one branch of the diagram in ``phases/todo/1.png``:

* praise -> turns bumped, NextPhrase substate observed, new word in
  ``current_word``.
* correct -> same word, turns unchanged, CorrectSub observed.
* retry -> same word, turns unchanged, CalmAndRetrySub observed.
* silence -> ``CANCEL`` with ``reason="silence"`` routes through the
  orchestrator into a CalmAndRetry side effect, ``SystemState`` stays
  ``SESSION``.
* target_turns reached -> DrillCompleteSub then pending ``MODE_CHANGE``
  to ``vision``.

Plus a small smoke test that :func:`build_face_registry` exposes the
new ``happy`` and ``worried`` face IDs.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.events import Event, EventType
from app.modes import IntentRouter, ModeRegistry, VoiceMirrorMode
from app.modes.voice_mirror import (
    CalmAndRetrySub,
    CorrectSub,
    DrillCompleteSub,
    ListenSub,
    NextPhraseSub,
    PraiseSub,
    PromptSub,
    ScoreSub,
)
from app.orchestrator import Orchestrator
from app.services import Services
from app.services.gemma import GemmaReply, ScoreResult, WordSuggestion
from app.state import AppState
from app.system_state import SystemState
from app.ui.faces import build_face_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Gemma:
    """Hand-rolled Gemma double with verdict-controlled scoring."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.suggestions: list[WordSuggestion] = [
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
            WordSuggestion(
                text="Now try saying: cherry.",
                audio_bytes=b"wav",
                audio_duration_ms=10.0,
                word="cherry",
                example_sentence="I like cherries.",
            ),
        ]
        self.score_result: ScoreResult = ScoreResult(
            text="Nice!",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            target_word="apple",
            transcript="apple",
            verdict="praise",
        )
        self.complete_reply = GemmaReply(
            text="Take a breath. Let's try again. Try saying: apple.",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
        )

    async def voice_mirror_suggest(
        self, *, history: list[str] | None = None, level: str | None = None
    ) -> WordSuggestion:
        self.calls.append({"kind": "suggest", "history": list(history or [])})
        if self.suggestions:
            return self.suggestions.pop(0)
        return WordSuggestion(
            text="Try saying: apple.",
            audio_bytes=b"wav",
            audio_duration_ms=10.0,
            word="apple",
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

    async def complete_with_tts(self, prompt: str) -> GemmaReply:
        self.calls.append({"kind": "complete", "prompt": prompt})
        return self.complete_reply


class _OrchStub:
    """Minimal orchestrator double for direct ``run_thinking`` calls."""

    def __init__(self, gemma: _Gemma) -> None:
        ns_services = type("S", (), {"gemma": gemma, "camera": None})()
        self.services = ns_services
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        self.state = AppState.THINKING


def _stub_services_fast() -> Services:
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    services.clips._latency_s = 0.0  # type: ignore[attr-defined]
    return services


def _record_substate_listener(mode: VoiceMirrorMode) -> list[str]:
    """Subscribe a listener to mode substate swaps and return the log."""

    seen: list[str] = []
    machine = mode.substate_machine
    assert machine is not None
    machine.add_listener(lambda _prev, curr: seen.append(curr.name))
    return seen


# ---------------------------------------------------------------------------
# Substate happy-path coverage (verdict matrix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_enter_starts_in_prompt_substate() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)

    seen = _record_substate_listener(mode)
    intro = await mode.on_enter(orch)  # type: ignore[arg-type]

    assert isinstance(intro, WordSuggestion)
    assert intro.word == "apple"
    assert mode.current_word == "apple"
    assert isinstance(mode.current_substate, PromptSub)
    assert seen == ["voice_mirror.prompt"]


@pytest.mark.asyncio
async def test_praise_increments_turns_and_moves_to_prompt_on_next_word() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]
    seen = _record_substate_listener(mode)

    gemma.score_result = ScoreResult(
        text="Nice apple!",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="apple",
        transcript="apple",
        verdict="praise",
    )
    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"attempt"}),
    )

    assert reply.text.startswith("Nice apple!")
    assert mode.turns_done == 1
    assert mode.current_word == "banana"
    # Praise -> NextPhrase -> Prompt (transient hops all observed).
    assert seen == [
        "voice_mirror.praise",
        "voice_mirror.next_phrase",
        "voice_mirror.prompt",
    ]
    assert isinstance(mode.current_substate, PromptSub)
    assert mode.pop_pending_mode_change() is None


@pytest.mark.asyncio
async def test_correct_keeps_same_word_and_does_not_bump_turns() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]
    seen = _record_substate_listener(mode)

    gemma.score_result = ScoreResult(
        text="Almost! Try the 'a' shorter.",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="apple",
        transcript="aple",
        verdict="correct",
    )
    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"attempt"}),
    )

    assert reply.text == "Almost! Try the 'a' shorter."
    assert mode.turns_done == 0
    assert mode.current_word == "apple"
    assert isinstance(mode.current_substate, CorrectSub)
    assert seen == ["voice_mirror.correct"]
    assert mode.pop_pending_mode_change() is None


@pytest.mark.asyncio
async def test_retry_keeps_same_word_via_calm_and_retry() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]
    seen = _record_substate_listener(mode)

    gemma.score_result = ScoreResult(
        text="Let's try again, gently.",
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

    assert reply.text == "Let's try again, gently."
    assert mode.turns_done == 0
    assert mode.current_word == "apple"
    assert isinstance(mode.current_substate, CalmAndRetrySub)
    assert seen == ["voice_mirror.calm_and_retry"]


@pytest.mark.asyncio
async def test_drill_complete_after_target_turns_requests_vision() -> None:
    mode = VoiceMirrorMode()
    assert mode.target_turns == 2
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]
    seen = _record_substate_listener(mode)

    # Turn 1: praise -> NextPhrase -> Prompt (banana).
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

    # Turn 2: praise -> drill complete (no new word fetched).
    gemma.score_result = ScoreResult(
        text="Great banana!",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="banana",
        transcript="banana",
        verdict="praise",
    )
    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"a2"}),
    )
    assert reply.text == "Great banana!"
    assert mode.turns_done == 2
    assert isinstance(mode.current_substate, DrillCompleteSub)
    assert "voice_mirror.drill_complete" in seen

    pending = mode.pop_pending_mode_change()
    assert pending is not None
    assert pending["target_mode"] == "vision"
    assert pending["auto_listen"] is False


# ---------------------------------------------------------------------------
# Sub-FSM event interception via handle_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_event_moves_to_score_on_utterance_end() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]
    orch.state = AppState.LISTENING

    transition = mode.handle_event(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"x"}),
    )
    assert transition is None  # defer to default table
    assert isinstance(mode.current_substate, ScoreSub)


@pytest.mark.asyncio
async def test_handle_event_returns_to_listen_after_correct_playback() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]

    gemma.score_result = ScoreResult(
        text="Almost!",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="apple",
        transcript="aple",
        verdict="correct",
    )
    await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"attempt"}),
    )
    assert isinstance(mode.current_substate, CorrectSub)

    orch.state = AppState.SPEAKING
    mode.handle_event(
        orch,  # type: ignore[arg-type]
        Event(EventType.PLAYBACK_DONE, payload={"bytes": 1}),
    )
    assert isinstance(mode.current_substate, ListenSub)


@pytest.mark.asyncio
async def test_handle_event_returns_to_listen_after_calm_and_retry_playback() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]

    gemma.score_result = ScoreResult(
        text="Take a breath.",
        audio_bytes=b"wav",
        audio_duration_ms=10.0,
        target_word="apple",
        transcript="ah",
        verdict="retry",
    )
    await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"audio_bytes": b"attempt"}),
    )
    assert isinstance(mode.current_substate, CalmAndRetrySub)

    orch.state = AppState.SPEAKING
    mode.handle_event(
        orch,  # type: ignore[arg-type]
        Event(EventType.PLAYBACK_DONE, payload={"bytes": 1}),
    )
    assert isinstance(mode.current_substate, ListenSub)


@pytest.mark.asyncio
async def test_handle_event_returns_to_listen_after_initial_prompt_playback() -> None:
    mode = VoiceMirrorMode()
    gemma = _Gemma()
    orch = _OrchStub(gemma)
    await mode.on_enter(orch)  # type: ignore[arg-type]
    assert isinstance(mode.current_substate, PromptSub)

    orch.state = AppState.SPEAKING
    mode.handle_event(
        orch,  # type: ignore[arg-type]
        Event(EventType.PLAYBACK_DONE, payload={"bytes": 1}),
    )
    assert isinstance(mode.current_substate, ListenSub)


# ---------------------------------------------------------------------------
# Silence CANCEL routing -- integration with the real orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silence_cancel_routes_to_calm_and_retry_and_keeps_session() -> None:
    """A CANCEL with ``reason="silence"`` while LISTENING must NOT drop
    the session. It must run the CalmAndRetry side effect (which posts
    REPLY_READY) and end up back in SPEAKING / LISTENING under the same
    active mode.
    """

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    # Single-mode registry so the orchestrator goes straight to
    # VoiceMirrorMode on wake -- skips the FreeConvo gate that's
    # irrelevant here.
    registry = ModeRegistry()
    registry.register(VoiceMirrorMode.name, VoiceMirrorMode)
    router = IntentRouter(
        default=VoiceMirrorMode.name, known_modes=registry.names()
    )
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

        # Wait until the mode is active and we're past initial
        # speaking (the on_enter cue is playing/has played).
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if isinstance(orch.active_mode, VoiceMirrorMode):
                break
            await asyncio.sleep(0.01)
        assert isinstance(orch.active_mode, VoiceMirrorMode)

        # Fire the silence CANCEL while LISTENING -- ensure we're in
        # LISTENING first.
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.state is AppState.LISTENING:
                break
            await asyncio.sleep(0.01)
        assert orch.state is AppState.LISTENING

        await queue.put(
            Event(EventType.CANCEL, payload={"reason": "silence"})
        )

        # The CalmAndRetry side effect posts REPLY_READY, which moves
        # to SPEAKING. Wait for that to confirm we stayed in session.
        deadline = asyncio.get_event_loop().time() + 2.0
        saw_speaking_after_cancel = False
        while asyncio.get_event_loop().time() < deadline:
            for _prev, curr, ev_type in transitions:
                if (
                    ev_type is EventType.REPLY_READY
                    and curr is AppState.SPEAKING
                ):
                    saw_speaking_after_cancel = True
                    break
            if saw_speaking_after_cancel:
                break
            await asyncio.sleep(0.01)

        assert saw_speaking_after_cancel, (
            f"no SPEAKING after silence CANCEL; transitions={transitions}"
        )
        assert orch.system_state is SystemState.SESSION
        assert isinstance(orch.active_mode, VoiceMirrorMode)
        assert isinstance(
            orch.active_mode.current_substate, (CalmAndRetrySub, ListenSub, PromptSub)
        )
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_non_silence_cancel_still_drops_to_idle() -> None:
    """CANCEL without ``reason="silence"`` (e.g. Gemma failure) still
    tears down the session per the original wildcard behavior."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()
    registry = ModeRegistry()
    registry.register(VoiceMirrorMode.name, VoiceMirrorMode)
    router = IntentRouter(
        default=VoiceMirrorMode.name, known_modes=registry.names()
    )
    orch = Orchestrator(
        queue, services, mode_registry=registry, intent_router=router
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if isinstance(orch.active_mode, VoiceMirrorMode):
                break
            await asyncio.sleep(0.01)
        assert isinstance(orch.active_mode, VoiceMirrorMode)

        await queue.put(Event(EventType.CANCEL))

        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.system_state is SystemState.IDLE:
                break
            await asyncio.sleep(0.01)
        assert orch.system_state is SystemState.IDLE
        assert orch.active_mode is None
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


# ---------------------------------------------------------------------------
# Face registry smoke test
# ---------------------------------------------------------------------------


def test_face_registry_exposes_new_substate_faces() -> None:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.display.init()
    try:
        registry = build_face_registry()
        for face_id in ("idle", "listening", "thinking", "speaking", "happy", "worried"):
            assert face_id in registry, f"missing face: {face_id}"
    finally:
        pygame.display.quit()


# Silence the "imported but unused" lint warnings for substates the
# subtests reference selectively.
_ = (NextPhraseSub, PraiseSub, DrillCompleteSub)
