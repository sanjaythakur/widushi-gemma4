"""Top-level FSM states for the orchestrator.

Kept in its own module so the UI, services, and tests can import the
enum without pulling in the orchestrator (and therefore pygame).
"""

from __future__ import annotations

from enum import Enum, auto


class AppState(Enum):
    """The four high-level conversation states from the spec."""

    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
