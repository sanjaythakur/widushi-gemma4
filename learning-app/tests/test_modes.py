"""Unit tests for the Mode contract, ModeRegistry, and IntentRouter."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.events import Event, EventType
from app.modes import (
    SYSTEM_PROMPT_HINGLISH,
    IntentRouter,
    Mode,
    ModeRegistry,
    TutorMode,
    UnknownModeError,
    build_default_registry,
    build_layered_prompt,
)
from app.modes.stubs import RoleplayMode, VisionMode, VoiceMirrorMode
from app.services.gemma import GemmaReply

# ---------------------------------------------------------------------------
# TutorMode + prompt architecture
# ---------------------------------------------------------------------------


def test_tutor_mode_name_and_prompts() -> None:
    tutor = TutorMode()
    assert tutor.name == "tutor"
    # Hinglish system prompt is wired in.
    assert tutor.system_prompt() == SYSTEM_PROMPT_HINGLISH
    assert "Hindi" in tutor.system_prompt()
    assert "English" in tutor.system_prompt()
    # Mode-level prompt is non-empty and self-describing.
    assert "TUTOR" in tutor.mode_prompt()


def test_build_prompt_layers_system_mode_and_user() -> None:
    tutor = TutorMode()
    prompt = tutor.build_prompt("How do I order tea?")
    assert "[SYSTEM]" in prompt
    assert "[MODE: tutor]" in prompt
    assert "User: How do I order tea?" in prompt
    # SYSTEM block contains Hinglish guidance.
    assert "Hindi" in prompt


def test_build_prompt_includes_history_when_present() -> None:
    tutor = TutorMode()
    tutor.record_turn(role="user", text="Hello")
    tutor.record_turn(role="assistant", text="Hi there.")
    prompt = tutor.build_prompt("How are you?")
    assert "[HISTORY]" in prompt
    assert "User: Hello" in prompt
    assert "Assistant: Hi there." in prompt


def test_build_layered_prompt_helper_matches_mode_output() -> None:
    """The free helper used by ad-hoc callers behaves identically."""

    out = build_layered_prompt(
        system_prompt="SYS",
        mode_prompt="MD",
        history_lines=["User: a", "Assistant: b"],
        user_text="c",
    )
    assert "[SYSTEM]\nSYS" in out
    assert "[MODE]\nMD" in out
    assert "[HISTORY]\nUser: a\nAssistant: b" in out
    assert out.endswith("User: c")


# ---------------------------------------------------------------------------
# Episode history
# ---------------------------------------------------------------------------


def test_record_turn_appends_to_episode_history() -> None:
    mode = TutorMode()
    assert mode.episode_history == []
    mode.record_turn(role="user", text="hi", extras={"intent": "greeting"})
    mode.record_turn(role="assistant", text="hello back")
    history = mode.episode_history
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].text == "hi"
    assert history[0].extras == {"intent": "greeting"}
    assert history[1].role == "assistant"


def test_reset_history_clears() -> None:
    mode = TutorMode()
    mode.record_turn(role="user", text="hi")
    mode.reset_history()
    assert mode.episode_history == []


# ---------------------------------------------------------------------------
# Default Mode.run_thinking (text + audio paths)
# ---------------------------------------------------------------------------


class _DummyGemma:
    """Minimal stand-in for GemmaClient.

    Doesn't need to inherit because run_thinking only calls two methods.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete_with_tts(self, prompt: str) -> GemmaReply:
        self.calls.append({"kind": "text", "prompt": prompt})
        return GemmaReply(
            text=f"echo:{prompt}", audio_bytes=b"wav", audio_duration_ms=10.0
        )

    async def listen_audio_with_tts(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "question.wav",
        content_type: str = "audio/wav",
    ) -> GemmaReply:
        self.calls.append(
            {
                "kind": "audio",
                "bytes": audio_bytes,
                "filename": filename,
                "content_type": content_type,
            }
        )
        return GemmaReply(
            text="heard you", audio_bytes=b"wav", audio_duration_ms=12.0
        )


class _OrchStub:
    """Bare-bones orchestrator stand-in for Mode.run_thinking tests."""

    def __init__(self, gemma: _DummyGemma) -> None:
        self.services = type("S", (), {"gemma": gemma})()


@pytest.mark.asyncio
async def test_default_run_thinking_text_path_appends_history() -> None:
    mode = TutorMode()
    gemma = _DummyGemma()
    orch = _OrchStub(gemma)

    reply = await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(EventType.UTTERANCE_END, payload={"prompt": "hello"}),
    )

    assert reply.text == "echo:hello"
    # Default Mode.run_thinking must pass the raw prompt (the SYSTEM
    # injection lives in build_prompt, NOT here) so existing Gemma
    # contracts keep working.
    assert gemma.calls == [{"kind": "text", "prompt": "hello"}]
    history = mode.episode_history
    assert [(t.role, t.text) for t in history] == [
        ("user", "hello"),
        ("assistant", "echo:hello"),
    ]


@pytest.mark.asyncio
async def test_default_run_thinking_audio_path_uses_listen_endpoint() -> None:
    mode = TutorMode()
    gemma = _DummyGemma()
    orch = _OrchStub(gemma)

    await mode.run_thinking(
        orch,  # type: ignore[arg-type]
        Event(
            EventType.UTTERANCE_END,
            payload={
                "audio_bytes": b"abc",
                "filename": "q.wav",
                "content_type": "audio/wav",
            },
        ),
    )

    assert len(gemma.calls) == 1
    call = gemma.calls[0]
    assert call["kind"] == "audio"
    assert call["bytes"] == b"abc"
    assert call["filename"] == "q.wav"
    history = mode.episode_history
    assert history[0].role == "user" and history[0].text == "<audio>"
    assert history[1].role == "assistant"


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class _LifecycleMode(Mode):
    name = "lifecycle"

    def __init__(self) -> None:
        super().__init__()
        self.enters = 0
        self.exits = 0

    async def on_enter(self, orch: Any) -> None:  # noqa: D401
        await super().on_enter(orch)
        self.enters += 1

    async def on_exit(self, orch: Any) -> None:
        self.exits += 1


@pytest.mark.asyncio
async def test_on_enter_resets_history_and_increments_counter() -> None:
    mode = _LifecycleMode()
    mode.record_turn(role="user", text="leftover")
    await mode.on_enter(None)
    assert mode.episode_history == []
    assert mode.enters == 1


def test_handle_event_default_returns_none() -> None:
    """Default contract: handle_event is a no-op; orchestrator falls
    back to its default table."""

    mode = TutorMode()
    fake_event = Event(EventType.UTTERANCE_END)
    assert mode.handle_event(orch=None, event=fake_event) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModeRegistry
# ---------------------------------------------------------------------------


def test_default_registry_has_all_interactive_modes() -> None:
    registry = build_default_registry()
    assert sorted(registry.names()) == [
        "free_convo",
        "roleplay",
        "tutor",
        "vision",
        "voice_mirror",
    ]
    for name in ("free_convo", "voice_mirror", "vision", "roleplay", "tutor"):
        assert name in registry


def test_registry_get_returns_new_instances_each_time() -> None:
    registry = build_default_registry()
    a = registry.get("tutor")
    b = registry.get("tutor")
    assert isinstance(a, TutorMode) and isinstance(b, TutorMode)
    assert a is not b


def test_registry_get_unknown_raises() -> None:
    registry = build_default_registry()
    with pytest.raises(UnknownModeError):
        registry.get("does-not-exist")


def test_registry_register_can_add_stub_modes() -> None:
    registry = ModeRegistry()
    registry.register(VisionMode.name, VisionMode)
    registry.register(RoleplayMode.name, RoleplayMode)
    registry.register(VoiceMirrorMode.name, VoiceMirrorMode)
    assert sorted(registry.names()) == ["roleplay", "vision", "voice_mirror"]
    assert isinstance(registry.get("vision"), VisionMode)
    assert isinstance(registry.get("roleplay"), RoleplayMode)
    assert isinstance(registry.get("voice_mirror"), VoiceMirrorMode)


# ---------------------------------------------------------------------------
# IntentRouter (stub)
# ---------------------------------------------------------------------------


def test_intent_router_stub_always_returns_free_convo() -> None:
    """Post-wake we always land in FreeConvoMode; mode swaps happen mid-session."""

    router = IntentRouter()
    assert router.default == "free_convo"
    assert router.classify(Event(EventType.WAKE_DETECTED)) == "free_convo"
    assert (
        router.classify(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))
        == "free_convo"
    )


def test_intent_router_default_override() -> None:
    router = IntentRouter(default="vision")
    assert router.classify(Event(EventType.WAKE_DETECTED)) == "vision"


# ---------------------------------------------------------------------------
# Stub modes still satisfy the contract (importable + buildable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [VisionMode, RoleplayMode, VoiceMirrorMode])
def test_stub_modes_satisfy_contract(cls: type[Mode]) -> None:
    mode = cls()
    assert isinstance(mode, Mode)
    assert isinstance(mode.name, str) and mode.name
    # All stub modes share the Hinglish system prompt for now.
    assert mode.system_prompt() == SYSTEM_PROMPT_HINGLISH
    assert mode.mode_prompt() != ""
    # Lifecycle awaitables.
    asyncio.get_event_loop()  # ensure pytest-asyncio plugin loaded
