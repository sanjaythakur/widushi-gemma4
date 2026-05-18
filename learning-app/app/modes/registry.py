"""Mode Registry.

Maps mode names ("free_convo", "voice_mirror", "vision", "roleplay",
"tutor") to factories that construct a :class:`Mode` instance. The
orchestrator consults the registry when the :class:`IntentRouter` picks
a name and when a mid-session ``MODE_CHANGE`` event arrives.
"""

from __future__ import annotations

from collections.abc import Callable

from app.modes.base import Mode
from app.modes.free_convo import FreeConvoMode
from app.modes.roleplay import RolePlayMode
from app.modes.tutor import TutorMode
from app.modes.vision import VisionMode
from app.modes.voice_mirror import VoiceMirrorMode

ModeFactory = Callable[[], Mode]


class UnknownModeError(KeyError):
    """Raised when ``ModeRegistry.get`` is called with an unregistered name."""


class ModeRegistry:
    """Name -> Mode factory lookup."""

    def __init__(self) -> None:
        self._factories: dict[str, ModeFactory] = {}

    def register(self, name: str, factory: ModeFactory) -> None:
        """Register ``factory`` under ``name``. Overwrites any prior entry."""

        self._factories[name] = factory

    def get(self, name: str) -> Mode:
        """Return a *new* :class:`Mode` instance for ``name``.

        Raises :class:`UnknownModeError` if the name is unregistered so
        callers can fall back to a default (the orchestrator falls back
        to the :class:`IntentRouter`'s configured default).
        """

        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise UnknownModeError(name) from exc
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._factories


def build_default_registry() -> ModeRegistry:
    """The registry used by ``main.py``.

    Registers every interactive mode the device knows about. The
    post-wake default is :class:`FreeConvoMode` (see
    :class:`IntentRouter`); other modes are reached via in-session
    ``MODE_CHANGE`` events as the learner progresses.
    """

    registry = ModeRegistry()
    registry.register(FreeConvoMode.name, FreeConvoMode)
    registry.register(VoiceMirrorMode.name, VoiceMirrorMode)
    registry.register(VisionMode.name, VisionMode)
    registry.register(RolePlayMode.name, RolePlayMode)
    # TutorMode is kept registered for legacy callers / direct routing.
    registry.register(TutorMode.name, TutorMode)
    return registry


__all__ = [
    "ModeFactory",
    "ModeRegistry",
    "UnknownModeError",
    "build_default_registry",
]
