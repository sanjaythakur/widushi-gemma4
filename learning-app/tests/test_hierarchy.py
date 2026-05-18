"""Tests for the outer SystemState + Mode lifecycle on the Orchestrator.

These tests exercise the hierarchical layer added on top of the flat
turn FSM:

* ``SystemState.IDLE -> SESSION`` on the first ``WAKE_DETECTED``, with
  the :class:`IntentRouter`'s choice installed as ``active_mode``.
* ``Mode.on_enter`` / ``on_exit`` are awaited at the boundary.
* ``CANCEL`` drops back to ``SystemState.IDLE`` and clears
  ``active_mode``.
* Custom intent routers + mode registries plug in.
"""

from __future__ import annotations

import asyncio

import pytest

from app.events import Event, EventType
from app.modes import FreeConvoMode, IntentRouter, Mode, ModeRegistry, TutorMode
from app.orchestrator import Orchestrator
from app.services import Services
from app.services.gemma import GemmaReply
from app.state import AppState
from app.system_state import SystemState


class _TrackingMode(Mode):
    """Mode that records lifecycle calls so tests can assert on them."""

    name = "tracking"

    def __init__(self) -> None:
        super().__init__()
        self.enter_count = 0
        self.exit_count = 0
        self.handle_calls: list[EventType] = []

    async def on_enter(self, orch) -> None:  # type: ignore[override]
        await super().on_enter(orch)
        self.enter_count += 1

    async def on_exit(self, orch) -> None:  # type: ignore[override]
        self.exit_count += 1

    def handle_event(self, orch, event):  # type: ignore[override]
        self.handle_calls.append(event.type)
        return None


def _stub_services_fast() -> Services:
    services = Services.stubs()
    services.gemma._latency_s = 0.01  # type: ignore[attr-defined]
    services.piper._latency_s = 0.01  # type: ignore[attr-defined]
    services.clips._latency_s = 0.0  # type: ignore[attr-defined]
    return services


@pytest.mark.asyncio
async def test_initial_system_state_is_idle() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    orch = Orchestrator(queue, Services.stubs())
    assert orch.system_state is SystemState.IDLE
    assert orch.active_mode is None


@pytest.mark.asyncio
async def test_wake_enters_session_with_free_convo_by_default() -> None:
    """Post-wake the device lands in FreeConvoMode (per spec-new.md)."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    orch = Orchestrator(queue, _stub_services_fast())
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.system_state is SystemState.SESSION:
                break
            await asyncio.sleep(0.01)
        assert orch.system_state is SystemState.SESSION
        assert isinstance(orch.active_mode, FreeConvoMode)
        assert orch.state is AppState.LISTENING
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_cancel_returns_system_state_to_idle_and_exits_mode() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()
    tracker = _TrackingMode()

    registry = ModeRegistry()
    registry.register(tracker.name, lambda: tracker)
    router = IntentRouter(default=tracker.name, known_modes=registry.names())

    orch = Orchestrator(
        queue,
        services,
        mode_registry=registry,
        intent_router=router,
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.system_state is SystemState.SESSION:
                break
            await asyncio.sleep(0.01)
        assert orch.system_state is SystemState.SESSION
        assert tracker.enter_count == 1
        assert tracker.exit_count == 0

        await queue.put(Event(EventType.CANCEL))
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.system_state is SystemState.IDLE:
                break
            await asyncio.sleep(0.01)
        assert orch.system_state is SystemState.IDLE
        assert orch.active_mode is None
        assert orch.state is AppState.IDLE
        assert tracker.exit_count == 1
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_mode_handle_event_is_consulted_before_default_table() -> None:
    """Every dispatched event passes through Mode.handle_event first."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()
    tracker = _TrackingMode()

    registry = ModeRegistry()
    registry.register(tracker.name, lambda: tracker)
    router = IntentRouter(default=tracker.name, known_modes=registry.names())

    orch = Orchestrator(
        queue,
        services,
        mode_registry=registry,
        intent_router=router,
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))
        deadline = asyncio.get_event_loop().time() + 2.0
        # Wait until we've seen at least the wake + utterance events
        # pass through the mode's handle_event.
        while asyncio.get_event_loop().time() < deadline:
            if EventType.UTTERANCE_END in tracker.handle_calls:
                break
            await asyncio.sleep(0.01)
        assert EventType.WAKE_DETECTED in tracker.handle_calls
        assert EventType.UTTERANCE_END in tracker.handle_calls
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_run_thinking_dispatches_to_active_mode() -> None:
    """The THINKING side-effect must call active_mode.run_thinking."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    class _CountingMode(Mode):
        name = "counting"

        def __init__(self) -> None:
            super().__init__()
            self.run_count = 0

        async def run_thinking(self, orch, event):  # type: ignore[override]
            self.run_count += 1
            return GemmaReply(
                text="ok", audio_bytes=b"wav", audio_duration_ms=10.0
            )

    mode = _CountingMode()
    registry = ModeRegistry()
    registry.register(mode.name, lambda: mode)
    router = IntentRouter(default=mode.name, known_modes=registry.names())

    orch = Orchestrator(
        queue,
        services,
        mode_registry=registry,
        intent_router=router,
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if mode.run_count >= 1:
                break
            await asyncio.sleep(0.01)
        assert mode.run_count == 1
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_gemma_failure_drops_system_state_to_idle_via_cancel() -> None:
    """Gemma errors must CANCEL the turn AND tear down the session."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    class _FailingMode(Mode):
        name = "failing"

        async def run_thinking(self, orch, event):  # type: ignore[override]
            raise RuntimeError("simulated gemma failure")

    failing = _FailingMode()
    registry = ModeRegistry()
    registry.register(failing.name, lambda: failing)
    router = IntentRouter(default=failing.name, known_modes=registry.names())

    orch = Orchestrator(
        queue,
        services,
        mode_registry=registry,
        intent_router=router,
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        await queue.put(Event(EventType.UTTERANCE_END, payload={"prompt": "hi"}))
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if (
                orch.system_state is SystemState.IDLE
                and orch.state is AppState.IDLE
            ):
                break
            await asyncio.sleep(0.01)
        assert orch.system_state is SystemState.IDLE
        assert orch.state is AppState.IDLE
        assert orch.active_mode is None
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_intent_router_picks_registered_mode() -> None:
    """A custom router that picks a non-default mode installs that mode."""

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    class _ModeA(Mode):
        name = "a"

    class _ModeB(Mode):
        name = "b"

    registry = ModeRegistry()
    registry.register("a", _ModeA)
    registry.register("b", _ModeB)
    router = IntentRouter(default="b", known_modes=registry.names())

    orch = Orchestrator(
        queue,
        services,
        mode_registry=registry,
        intent_router=router,
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.active_mode is not None:
                break
            await asyncio.sleep(0.01)
        assert isinstance(orch.active_mode, _ModeB)
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)


@pytest.mark.asyncio
async def test_unknown_router_choice_falls_back_to_default() -> None:
    """If the router returns an unregistered name, we fall back to the
    router's configured default; if even that is missing, SystemState
    stays IDLE.
    """

    class _BadRouter(IntentRouter):
        def classify(self, event):  # type: ignore[override]
            return "does-not-exist"

    queue: asyncio.Queue[Event] = asyncio.Queue()
    services = _stub_services_fast()

    registry = ModeRegistry()
    registry.register(TutorMode.name, TutorMode)
    router = _BadRouter(default=TutorMode.name, known_modes=registry.names())

    orch = Orchestrator(
        queue,
        services,
        mode_registry=registry,
        intent_router=router,
    )
    runner = asyncio.create_task(orch.run())

    try:
        await queue.put(Event(EventType.WAKE_DETECTED))
        deadline = asyncio.get_event_loop().time() + 1.0
        while asyncio.get_event_loop().time() < deadline:
            if orch.active_mode is not None:
                break
            await asyncio.sleep(0.01)
        # We fell back to the default (tutor) rather than IDLE-staying.
        assert isinstance(orch.active_mode, TutorMode)
        assert orch.system_state is SystemState.SESSION
    finally:
        await queue.put(Event(EventType.SHUTDOWN))
        await asyncio.wait_for(runner, timeout=2.0)
