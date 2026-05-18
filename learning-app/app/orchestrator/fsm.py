"""Central state machine for the Widushi app.

This is now a *hierarchical* state machine:

* ``SystemState`` (outer): ``IDLE`` / ``SESSION`` / ``ERROR``. The
  orchestrator switches into ``SESSION`` on the first
  ``WAKE_DETECTED`` and picks an active :class:`Mode` via the
  :class:`IntentRouter`. ``CANCEL`` drops back to ``IDLE``.
* ``AppState`` (inner): the original per-turn flow (``IDLE`` ->
  ``LISTENING`` -> ``THINKING`` -> ``SPEAKING`` -> ``LISTENING``). The
  active :class:`Mode` may intercept transitions via
  :meth:`Mode.handle_event`; otherwise the orchestrator falls back to
  its default transition table — which is the same one Phase 2 shipped,
  so existing tests in ``tests/test_fsm.py`` keep passing.

The orchestrator owns the ``state`` variable and is the *only* thing
allowed to mutate it. All input sources push :class:`Event` objects onto
a single ``asyncio.Queue``; the orchestrator runs a
``while True: await queue.get()`` loop and dispatches.

Each transition optionally triggers a side-effect coroutine — for
example, ``THINKING`` calls :meth:`Mode.run_thinking` (which by default
hits ``services.gemma``) and then self-posts ``REPLY_READY`` so the FSM
keeps walking.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.events import Event, EventType
from app.input.utterance import FollowUpListenSignal
from app.modes.base import Mode
from app.modes.intent_router import IntentRouter
from app.modes.registry import ModeRegistry, UnknownModeError, build_default_registry
from app.services import Services
from app.state import AppState
from app.system_state import SystemState

log = logging.getLogger(__name__)

SideEffect = Callable[["Orchestrator", Event], Awaitable[None]]


@dataclass(frozen=True)
class Transition:
    target: AppState
    side_effect: SideEffect | None = None


class Orchestrator:
    """The single FSM consumer of the event queue."""

    def __init__(
        self,
        queue: asyncio.Queue[Event],
        services: Services,
        *,
        initial_state: AppState = AppState.IDLE,
        followup_listen: FollowUpListenSignal | None = None,
        mode_registry: ModeRegistry | None = None,
        intent_router: IntentRouter | None = None,
    ) -> None:
        self._queue = queue
        self._services = services
        self._state = initial_state
        self._followup_listen = followup_listen
        self._listeners: list[Callable[[AppState, AppState, Event], None]] = []
        self._side_effect_tasks: set[asyncio.Task[None]] = set()
        self._table: dict[tuple[AppState, EventType], Transition] = self._build_table()

        self._mode_registry = mode_registry or build_default_registry()
        self._intent_router = intent_router or IntentRouter(
            known_modes=self._mode_registry.names()
        )
        self._system_state = SystemState.IDLE
        self._active_mode: Mode | None = None

    @property
    def state(self) -> AppState:
        """Inner per-turn state. Preserved for UI / legacy callers."""

        return self._state

    @property
    def system_state(self) -> SystemState:
        """Outer hierarchical state."""

        return self._system_state

    @property
    def active_mode(self) -> Mode | None:
        """The mode currently owning the session, or ``None`` in IDLE."""

        return self._active_mode

    @property
    def mode_registry(self) -> ModeRegistry:
        return self._mode_registry

    @property
    def intent_router(self) -> IntentRouter:
        return self._intent_router

    @property
    def services(self) -> Services:
        return self._services

    @property
    def queue(self) -> asyncio.Queue[Event]:
        return self._queue

    @property
    def followup_listen(self) -> FollowUpListenSignal | None:
        return self._followup_listen

    def add_listener(self, fn: Callable[[AppState, AppState, Event], None]) -> None:
        """Register a callback fired after every accepted transition."""

        self._listeners.append(fn)

    async def run(self) -> None:
        """Main consume loop. Runs until a ``SHUTDOWN`` event is dequeued."""

        log.info(
            "orchestrator started in system=%s app=%s",
            self._system_state.name,
            self._state.name,
        )
        try:
            while True:
                event = await self._queue.get()
                if event.type is EventType.SHUTDOWN:
                    log.info("orchestrator: SHUTDOWN received, exiting")
                    return
                await self._dispatch(event)
        finally:
            await self._cancel_side_effects()
            if self._active_mode is not None:
                await self._exit_active_mode()

    async def _dispatch(self, event: Event) -> None:
        # CANCEL is a wildcard: from any state, drop back to IDLE, cancel
        # the active mode (if any), and abandon any pending side effect.
        if event.type is EventType.CANCEL:
            await self._cancel_side_effects()
            if self._active_mode is not None:
                await self._exit_active_mode()
            self._system_state = SystemState.IDLE
            await self._transition(AppState.IDLE, event, side_effect=None)
            return

        # Mid-session mode swap: exit the current mode and install the
        # next one without leaving SESSION. The new mode's ``on_enter``
        # may return a GemmaReply (e.g. VoiceMirror's "Try saying: apple")
        # which we play before listening for the learner's next turn.
        if event.type is EventType.MODE_CHANGE:
            await self._handle_mode_change(event)
            return

        # Hierarchical step: a fresh WAKE_DETECTED while at rest enters
        # SESSION and installs the mode the IntentRouter picks. The
        # downstream IDLE+WAKE_DETECTED table entry then runs as before.
        if (
            event.type is EventType.WAKE_DETECTED
            and self._system_state is SystemState.IDLE
        ):
            await self._enter_session(event)

        # Modes get first refusal on every event. If the mode returns a
        # Transition we use that; otherwise we fall back to the default
        # table.
        transition: Transition | None = None
        if self._active_mode is not None:
            try:
                transition = self._active_mode.handle_event(self, event)
            except Exception:  # mode misbehavior must never break the FSM
                log.exception(
                    "mode %s.handle_event raised; deferring to default table",
                    self._active_mode.name,
                )
                transition = None

        if transition is None:
            transition = self._table.get((self._state, event.type))

        if transition is None:
            log.debug(
                "orchestrator: ignored %s in %s", event.type.name, self._state.name
            )
            return
        await self._transition(transition.target, event, transition.side_effect)

    async def _enter_session(self, event: Event) -> None:
        """Pick a mode via IntentRouter and enter SESSION.

        Falls back to the IntentRouter's ``default`` mode name if the
        classifier returns an unregistered name. If even the default is
        missing, logs an error and stays in SystemState.IDLE so the
        legacy code path (no mode) keeps working.
        """

        name = self._intent_router.classify(event)
        try:
            mode = self._mode_registry.get(name)
        except UnknownModeError:
            log.warning(
                "intent router picked unregistered mode %r; falling back to %r",
                name,
                self._intent_router.default,
            )
            try:
                mode = self._mode_registry.get(self._intent_router.default)
            except UnknownModeError:
                log.error(
                    "no mode registered (not even default %r); staying in IDLE",
                    self._intent_router.default,
                )
                return

        self._active_mode = mode
        self._system_state = SystemState.SESSION
        log.info(
            "system %s -> SESSION (mode=%s)", SystemState.IDLE.name, mode.name
        )
        try:
            await mode.on_enter(self)
        except Exception:
            log.exception("mode %s.on_enter raised; continuing", mode.name)
        # Note: any GemmaReply returned by on_enter is intentionally
        # ignored at session entry. The default ``IDLE+WAKE_DETECTED``
        # table transition that immediately follows takes us into
        # LISTENING, which is the natural place for the first user turn
        # (the wake-word source has already played ``listen_start``).
        # Modes that want a greeting on initial wake can either:
        # (a) request it via the next ``run_thinking`` turn, or
        # (b) request a MODE_CHANGE to themselves from another mode.

    async def _handle_mode_change(self, event: Event) -> None:
        """Swap the active Mode without leaving ``SystemState.SESSION``.

        Steps:

        1. Cancel any in-flight side effects (e.g. a pending follow-up
           listen task spawned by ``PLAYBACK_DONE``).
        2. ``on_exit`` the current mode.
        3. Resolve the new mode from the registry; fall back to the
           router's default if unknown.
        4. ``on_enter`` the new mode. If it returns a :class:`GemmaReply`
           we move the inner FSM to ``THINKING`` and enqueue a
           ``REPLY_READY`` so the SPEAKING side effect plays the
           greeting before listening. Otherwise we transition straight
           to ``LISTENING`` and (optionally) arm the follow-up signal so
           the wake source begins recording without a new wake word.
        """

        target = str(event.payload.get("target_mode") or "").strip()
        auto_listen = bool(event.payload.get("auto_listen", True))

        if not target:
            log.warning("MODE_CHANGE with no target_mode; ignoring")
            return

        await self._cancel_side_effects()
        if self._active_mode is not None:
            await self._exit_active_mode()

        try:
            new_mode = self._mode_registry.get(target)
        except UnknownModeError:
            log.warning(
                "MODE_CHANGE: unknown mode %r; falling back to router default %r",
                target,
                self._intent_router.default,
            )
            try:
                new_mode = self._mode_registry.get(self._intent_router.default)
            except UnknownModeError:
                log.error(
                    "MODE_CHANGE: no fallback mode registered; dropping session"
                )
                self._system_state = SystemState.IDLE
                await self._transition(AppState.IDLE, event, side_effect=None)
                return

        self._active_mode = new_mode
        self._system_state = SystemState.SESSION
        log.info("MODE_CHANGE -> %s (auto_listen=%s)", new_mode.name, auto_listen)

        try:
            intro = await new_mode.on_enter(self)
        except Exception:
            log.exception(
                "mode %s.on_enter raised during MODE_CHANGE; cancelling session",
                new_mode.name,
            )
            await self._queue.put(Event(EventType.CANCEL))
            return

        if intro is not None and isinstance(intro.audio_bytes, bytes):
            # Speak the intro: move to THINKING and enqueue REPLY_READY
            # so the default ``THINKING+REPLY_READY -> SPEAKING`` table
            # entry kicks the standard playback path.
            await self._transition(AppState.THINKING, event, side_effect=None)
            await self._queue.put(
                Event(
                    EventType.REPLY_READY,
                    payload={
                        "text": intro.text,
                        "audio_bytes": intro.audio_bytes,
                        "audio_duration_ms": intro.audio_duration_ms,
                        "voice": intro.voice,
                    },
                )
            )
            return

        # No intro to speak: move straight to LISTENING.
        await self._transition(AppState.LISTENING, event, side_effect=None)
        if auto_listen and self._followup_listen is not None:
            self._followup_listen.request()

    async def _exit_active_mode(self) -> None:
        mode = self._active_mode
        if mode is None:
            return
        self._active_mode = None
        try:
            await mode.on_exit(self)
        except Exception:
            log.exception("mode %s.on_exit raised; continuing", mode.name)

    async def _transition(
        self,
        target: AppState,
        event: Event,
        side_effect: SideEffect | None,
    ) -> None:
        previous = self._state
        self._state = target
        log.info(
            "transition %s -> %s (on %s)", previous.name, target.name, event.type.name
        )
        for listener in self._listeners:
            try:
                listener(previous, target, event)
            except Exception:  # listeners must never break the FSM
                log.exception("listener raised; ignoring")
        if side_effect is not None:
            self._spawn_side_effect(side_effect, event)

    def _spawn_side_effect(self, side_effect: SideEffect, event: Event) -> None:
        task = asyncio.create_task(side_effect(self, event))
        self._side_effect_tasks.add(task)
        task.add_done_callback(self._side_effect_tasks.discard)

    async def _cancel_side_effects(self) -> None:
        if not self._side_effect_tasks:
            return
        for task in list(self._side_effect_tasks):
            task.cancel()
        await asyncio.gather(*self._side_effect_tasks, return_exceptions=True)
        self._side_effect_tasks.clear()

    def _build_table(self) -> dict[tuple[AppState, EventType], Transition]:
        return {
            (AppState.IDLE, EventType.WAKE_DETECTED): Transition(
                AppState.LISTENING,
            ),
            (AppState.LISTENING, EventType.UTTERANCE_END): Transition(
                AppState.THINKING,
                side_effect=_thinking_side_effect,
            ),
            (AppState.THINKING, EventType.REPLY_READY): Transition(
                AppState.SPEAKING,
                side_effect=_speaking_side_effect,
            ),
            (AppState.SPEAKING, EventType.PLAYBACK_DONE): Transition(
                AppState.LISTENING,
                side_effect=_followup_listen_side_effect,
            ),
        }


async def _call_active_mode_thinking(orch: Orchestrator, event: Event):
    """Delegate to the active mode's ``run_thinking``.

    Falls back to a transient default :class:`Mode` so the THINKING
    side-effect still works in pathological setups where no mode was
    installed (e.g. UTTERANCE_END arriving without a prior
    WAKE_DETECTED). Tests don't exercise this branch but it keeps the
    code defensive.
    """

    mode = orch.active_mode
    if mode is None:
        log.debug("no active mode; using transient default Mode for run_thinking")
        mode = Mode()
    return await mode.run_thinking(orch, event)


async def _thinking_side_effect(orch: Orchestrator, event: Event) -> None:
    """Call Gemma via the active mode and post REPLY_READY when ready.

    Plays the ack clip (``wait_thinking`` for fresh turns,
    ``wait_checking`` for follow-up loopbacks) concurrently with the
    Gemma call so the clip masks Gemma's latency. REPLY_READY is only
    posted after both finish so SPEAKING audio doesn't trample the clip.
    """

    is_followup = bool(event.payload.get("followup", False))
    clip_id = "wait_checking" if is_followup else "wait_thinking"
    clip_task = asyncio.create_task(orch.services.clips.play(clip_id))

    try:
        try:
            reply = await _call_active_mode_thinking(orch, event)
            if not isinstance(reply.audio_bytes, bytes):
                raise RuntimeError("Gemma reply did not include audio bytes")
        except asyncio.CancelledError:
            clip_task.cancel()
            await asyncio.gather(clip_task, return_exceptions=True)
            raise
        except Exception:
            clip_task.cancel()
            await asyncio.gather(clip_task, return_exceptions=True)
            log.exception("gemma reply/TTS failed; cancelling turn")
            await orch.queue.put(Event(EventType.CANCEL))
            return

        await asyncio.gather(clip_task, return_exceptions=True)
        await orch.queue.put(
            Event(
                EventType.REPLY_READY,
                payload={
                    "text": reply.text,
                    "audio_bytes": reply.audio_bytes,
                    "audio_duration_ms": reply.audio_duration_ms,
                    "voice": reply.voice,
                },
            )
        )
    finally:
        if not clip_task.done():
            clip_task.cancel()
            await asyncio.gather(clip_task, return_exceptions=True)


async def _followup_listen_side_effect(orch: Orchestrator, event: Event) -> None:
    """Ask the wake-word source to start a follow-up listen so the user
    can ask another question without re-saying "Widushi". The wake-word
    source is responsible for emitting CANCEL after
    LISTENING_SILENCE_TIMEOUT_S of silence so the FSM falls back to IDLE.
    """

    signal = orch.followup_listen
    if signal is None:
        log.debug("PLAYBACK_DONE: no follow-up signal wired; staying in LISTENING")
        return
    signal.request()


async def _speaking_side_effect(orch: Orchestrator, event: Event) -> None:
    """Play the prepared reply audio and post the next event.

    Normally that next event is ``PLAYBACK_DONE`` which loops back to
    ``LISTENING`` with a follow-up. If the active mode has scheduled a
    mid-session swap via :meth:`Mode.request_mode_change`, we instead
    post ``MODE_CHANGE`` so the new mode's greeting plays uninterrupted
    by a wake-source recording started by the follow-up listen.
    """

    try:
        wav = event.payload.get("audio_bytes")
        if not isinstance(wav, bytes):
            raise RuntimeError("REPLY_READY did not include audio bytes")
        duration_ms = event.payload.get("audio_duration_ms")
        await orch.services.piper.play(
            wav,
            duration_ms=float(duration_ms) if isinstance(duration_ms, (int, float)) else None,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("piper playback failed; cancelling turn")
        await orch.queue.put(Event(EventType.CANCEL))
        return

    pending: dict | None = None
    mode = orch.active_mode
    if mode is not None:
        pending = mode.pop_pending_mode_change()

    if pending:
        await orch.queue.put(
            Event(
                EventType.MODE_CHANGE,
                payload={
                    "target_mode": str(pending.get("target_mode") or ""),
                    "auto_listen": bool(pending.get("auto_listen", True)),
                },
            )
        )
        return

    await orch.queue.put(
        Event(EventType.PLAYBACK_DONE, payload={"bytes": len(wav)})
    )
