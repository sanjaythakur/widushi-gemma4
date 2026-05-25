"""Generic sub-FSM contract for :class:`Mode` implementations.

A :class:`Mode` (the outer per-activity controller) may opt into a
sub-FSM by constructing a :class:`SubStateMachine` and transitioning
between :class:`ModeSubState` instances as the activity progresses. The
machinery is mode-agnostic: it owns the "which substate is current"
bookkeeping, fires lifecycle hooks (``on_enter`` / ``on_exit``), and
notifies listeners so the UI (or tests) can react.

Vision and Roleplay don't ship with a sub-FSM yet; they simply leave
``Mode.substate_machine`` unset and behave exactly as before.
VoiceMirrorMode is the first consumer and models the
Prompt/Listen/Score/Praise/Correct/CalmAndRetry/NextPhrase/DrillComplete
diagram in ``phases/todo/1.png``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular imports; runtime references only
    from app.modes.base import Mode
    from app.orchestrator.fsm import Orchestrator

log = logging.getLogger(__name__)


SubStateListener = Callable[
    ["ModeSubState | None", "ModeSubState"], None
]
"""Callback fired after each substate swap.

Signature: ``(previous_or_None, current)``. Raised exceptions are
swallowed so a misbehaving listener cannot break the FSM.
"""


class ModeSubState:
    """One named state inside a :class:`Mode`'s sub-FSM.

    Subclasses override ``name`` and optionally ``face_id``. When
    ``face_id`` is ``None`` the UI falls back to the orchestrator's
    outer :class:`AppState` face (idle/listening/thinking/speaking).
    """

    name: str = "base"
    face_id: str | None = None

    async def on_enter(self, mode: Mode, orch: Orchestrator | None) -> None:
        """Called after this substate becomes :attr:`SubStateMachine.current`."""

        del mode, orch

    async def on_exit(self, mode: Mode, orch: Orchestrator | None) -> None:
        """Called just before another substate replaces this one."""

        del mode, orch

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} name={self.name!r} face_id={self.face_id!r}>"


class SubStateMachine:
    """Tiny coordinator that owns the active :class:`ModeSubState`.

    Modes that want explicit substates instantiate one of these and call
    :meth:`transition` whenever they want to swap. Listeners (the UI
    loop, tests) attach via :meth:`add_listener` to observe changes.
    """

    def __init__(self) -> None:
        self._current: ModeSubState | None = None
        self._listeners: list[SubStateListener] = []

    @property
    def current(self) -> ModeSubState | None:
        return self._current

    def add_listener(self, listener: SubStateListener) -> None:
        self._listeners.append(listener)

    async def transition(
        self,
        mode: Mode,
        orch: Orchestrator | None,
        target: ModeSubState,
    ) -> None:
        """Swap to ``target``: exit the previous substate, install the new
        one, then fire each registered listener.

        Use this from async contexts (``on_enter``, ``run_thinking``,
        side-effect coroutines). For the synchronous fast path -- e.g.
        :meth:`Mode.handle_event`, which the orchestrator invokes from
        its event-dispatch loop -- use :meth:`set_current` instead.

        ``orch`` may be ``None`` in unit tests that drive a mode without
        a real orchestrator; the lifecycle hooks must tolerate that.
        """

        previous = self._current
        if previous is target:
            return

        if previous is not None:
            try:
                await previous.on_exit(mode, orch)
            except Exception:  # noqa: BLE001 - lifecycle hooks must not break the FSM
                log.exception(
                    "substate %s.on_exit raised; continuing", previous.name
                )

        self._set_and_notify(previous, target)

        try:
            await target.on_enter(mode, orch)
        except Exception:  # noqa: BLE001 - lifecycle hooks must not break the FSM
            log.exception("substate %s.on_enter raised; continuing", target.name)

    def set_current(self, target: ModeSubState) -> None:
        """Synchronous variant of :meth:`transition`.

        Bypasses async ``on_enter`` / ``on_exit`` so it can be called
        from sync hooks like :meth:`Mode.handle_event`. Listeners still
        fire so the UI updates immediately.
        """

        previous = self._current
        if previous is target:
            return
        self._set_and_notify(previous, target)

    def _set_and_notify(
        self,
        previous: ModeSubState | None,
        target: ModeSubState,
    ) -> None:
        self._current = target
        log.info(
            "substate %s -> %s",
            previous.name if previous is not None else "<none>",
            target.name,
        )
        for listener in self._listeners:
            try:
                listener(previous, target)
            except Exception:  # noqa: BLE001 - listeners must never break the FSM
                log.exception("substate listener raised; ignoring")

    def reset(self) -> None:
        """Drop the current substate without firing lifecycle hooks.

        Used by :meth:`Mode.on_enter` when a mode wants to start fresh.
        """

        self._current = None


SubStateAction = Callable[["Mode", "Orchestrator | None"], Awaitable[None]]
"""Optional async hook signature kept here for tests that build ad-hoc
substates inline (e.g. via lambdas wrapped in async helpers)."""


__all__ = [
    "ModeSubState",
    "SubStateAction",
    "SubStateListener",
    "SubStateMachine",
]
