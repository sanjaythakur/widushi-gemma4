"""Events flowing through the orchestrator's input queue.

Every input source (keyboard, FastAPI, future mic / wake-word /
camera) produces ``Event`` objects and pushes them into the single
``asyncio.Queue`` consumed by the ``Orchestrator``.

The set of types is deliberately small in Phase 2 — just enough to
walk the four-state FSM end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    WAKE_DETECTED = auto()
    UTTERANCE_END = auto()
    REPLY_READY = auto()
    PLAYBACK_DONE = auto()
    CANCEL = auto()
    SHUTDOWN = auto()


@dataclass(frozen=True)
class Event:
    """A single message handed to the orchestrator."""

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
