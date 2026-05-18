"""Outer (system-level) state for the hierarchical state machine.

The flat FSM in :mod:`app.state` (``AppState``) describes a *single turn*
inside a session — LISTENING / THINKING / SPEAKING. ``SystemState`` is
the layer above: whether the device is at rest, mid-session inside some
:class:`app.modes.base.Mode`, or in a recoverable error.

Kept in its own module for the same reason as ``app.state``: UI, tests,
and services can import the enum without pulling in the orchestrator (or
pygame).
"""

from __future__ import annotations

from enum import Enum, auto


class SystemState(Enum):
    """Top-level state of the device.

    * ``IDLE`` — at rest, waiting for wake word.
    * ``SESSION`` — a :class:`Mode` is active and owns the inner-turn
      FSM (``AppState``). The orchestrator delegates per-event handling
      to the active mode before falling back to its default transition
      table.
    * ``ERROR`` — unrecoverable infrastructure failure inside the
      orchestrator itself. Gemma / Piper failures are *not* ERROR; they
      cancel the turn back to ``IDLE`` via the existing CANCEL path.
    """

    IDLE = auto()
    SESSION = auto()
    ERROR = auto()
