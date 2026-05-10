"""Per-state face renderers."""

from __future__ import annotations

from app.state import AppState
from app.ui.faces.base import Face
from app.ui.faces.idle import IdleFace
from app.ui.faces.listening import ListeningFace
from app.ui.faces.speaking import SpeakingFace
from app.ui.faces.thinking import ThinkingFace


def build_face_registry() -> dict[AppState, Face]:
    """One face instance per state, instantiated up front."""

    return {
        AppState.IDLE: IdleFace(),
        AppState.LISTENING: ListeningFace(),
        AppState.THINKING: ThinkingFace(),
        AppState.SPEAKING: SpeakingFace(),
    }


__all__ = [
    "Face",
    "IdleFace",
    "ListeningFace",
    "SpeakingFace",
    "ThinkingFace",
    "build_face_registry",
]
