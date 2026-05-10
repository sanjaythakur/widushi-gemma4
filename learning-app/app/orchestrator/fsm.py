"""Central finite-state machine for the Widushi app.

The orchestrator owns the ``state`` variable and is the *only* thing
allowed to mutate it (per ``phases/2-system-scaffolding.md``). All
input sources push :class:`Event` objects onto a single
``asyncio.Queue``; the orchestrator runs a ``while True: await
queue.get()`` loop and dispatches based on
``(current_state, event_type)``.

Each transition optionally triggers a side-effect coroutine — for
example, ``THINKING`` calls ``services.gemma.complete(...)`` and then
self-posts ``REPLY_READY`` so the FSM keeps walking. In Phase 2 those
coroutines run against stub services; the call sites are unchanged
when the real implementations land.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.events import Event, EventType
from app.input.utterance import FollowUpListenSignal
from app.services import Services
from app.services.gemma import GemmaReply
from app.state import AppState

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
    ) -> None:
        self._queue = queue
        self._services = services
        self._state = initial_state
        self._followup_listen = followup_listen
        self._listeners: list[Callable[[AppState, AppState, Event], None]] = []
        self._side_effect_tasks: set[asyncio.Task[None]] = set()
        self._table: dict[tuple[AppState, EventType], Transition] = self._build_table()

    @property
    def state(self) -> AppState:
        return self._state

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

        log.info("orchestrator started in %s", self._state.name)
        try:
            while True:
                event = await self._queue.get()
                if event.type is EventType.SHUTDOWN:
                    log.info("orchestrator: SHUTDOWN received, exiting")
                    return
                await self._dispatch(event)
        finally:
            await self._cancel_side_effects()

    async def _dispatch(self, event: Event) -> None:
        # CANCEL is a wildcard: from any state, drop back to IDLE and
        # abandon any pending side effect.
        if event.type is EventType.CANCEL:
            await self._cancel_side_effects()
            await self._transition(AppState.IDLE, event, side_effect=None)
            return

        transition = self._table.get((self._state, event.type))
        if transition is None:
            log.debug(
                "orchestrator: ignored %s in %s", event.type.name, self._state.name
            )
            return
        await self._transition(transition.target, event, transition.side_effect)

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


async def _call_gemma(orch: Orchestrator, event: Event) -> GemmaReply:
    audio_bytes = event.payload.get("audio_bytes")
    if isinstance(audio_bytes, bytes):
        return await orch.services.gemma.listen_audio_with_tts(
            audio_bytes,
            filename=str(event.payload.get("filename", "question.wav")),
            content_type=str(event.payload.get("content_type", "audio/wav")),
        )
    prompt = event.payload.get("prompt", "")
    return await orch.services.gemma.complete_with_tts(str(prompt))


async def _thinking_side_effect(orch: Orchestrator, event: Event) -> None:
    """Call Gemma and post REPLY_READY after text and audio are ready.

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
            reply = await _call_gemma(orch, event)
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
    """Play the prepared reply audio and post PLAYBACK_DONE when it finishes."""

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
    await orch.queue.put(
        Event(EventType.PLAYBACK_DONE, payload={"bytes": len(wav)})
    )
