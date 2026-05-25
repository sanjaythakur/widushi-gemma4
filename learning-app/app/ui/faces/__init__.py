"""Per-state face renderers.

Faces are keyed by a string ``face_id`` so both the outer
:class:`app.state.AppState` (idle/listening/thinking/speaking) AND
mode-substate face IDs (``happy``, ``worried``, ...) share one
registry. The UI loop picks a face by:

1. asking the active mode for ``current_substate.face_id``, if any;
2. otherwise mapping ``orchestrator.state`` to a default face_id via
   :data:`APPSTATE_TO_FACE_ID`.
"""

from __future__ import annotations

from app.state import AppState
from app.ui.faces.base import Face
from app.ui.faces.happy import HappyFace
from app.ui.faces.idle import IdleFace
from app.ui.faces.listening import ListeningFace
from app.ui.faces.speaking import SpeakingFace
from app.ui.faces.thinking import ThinkingFace
from app.ui.faces.worried import WorriedFace

#: Maps :class:`AppState` to the default ``face_id`` used when the
#: active mode does not name a substate face explicitly.
APPSTATE_TO_FACE_ID: dict[AppState, str] = {
    AppState.IDLE: "idle",
    AppState.LISTENING: "listening",
    AppState.THINKING: "thinking",
    AppState.SPEAKING: "speaking",
}


def build_face_registry() -> dict[str, Face]:
    """Return one face instance per ``face_id``, instantiated up front."""

    return {
        "idle": IdleFace(),
        "listening": ListeningFace(),
        "thinking": ThinkingFace(),
        "speaking": SpeakingFace(),
        "happy": HappyFace(),
        "worried": WorriedFace(),
    }


__all__ = [
    "APPSTATE_TO_FACE_ID",
    "Face",
    "HappyFace",
    "IdleFace",
    "ListeningFace",
    "SpeakingFace",
    "ThinkingFace",
    "WorriedFace",
    "build_face_registry",
]
