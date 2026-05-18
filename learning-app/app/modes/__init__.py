"""Hierarchical Mode layer.

A :class:`Mode` is a sub-FSM specialised for one teaching activity.
The orchestrator owns the outer ``SystemState``; when ``SystemState``
is ``SESSION`` the active Mode owns the inner ``AppState`` turn flow,
builds the prompts for Gemma, and may schedule mid-session swaps to
the next mode via :meth:`Mode.request_mode_change`.

Default flow today: wake -> :class:`FreeConvoMode` ->
:class:`VoiceMirrorMode` (on ``start_learning``) -> :class:`VisionMode`
(after 2 scored turns) -> :class:`RolePlayMode` (after 2 taught turns).
"""

from __future__ import annotations

from app.modes.base import EpisodeTurn, Mode
from app.modes.free_convo import FreeConvoMode
from app.modes.intent_router import IntentRouter
from app.modes.prompts import SYSTEM_PROMPT_HINGLISH, build_layered_prompt
from app.modes.registry import (
    ModeFactory,
    ModeRegistry,
    UnknownModeError,
    build_default_registry,
)
from app.modes.roleplay import RolePlayMode
from app.modes.tutor import TutorMode
from app.modes.vision import VisionMode
from app.modes.voice_mirror import VoiceMirrorMode

__all__ = [
    "EpisodeTurn",
    "FreeConvoMode",
    "IntentRouter",
    "Mode",
    "ModeFactory",
    "ModeRegistry",
    "RolePlayMode",
    "SYSTEM_PROMPT_HINGLISH",
    "TutorMode",
    "UnknownModeError",
    "VisionMode",
    "VoiceMirrorMode",
    "build_default_registry",
    "build_layered_prompt",
]
