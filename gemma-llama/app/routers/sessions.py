"""Ops endpoints for the session/episode/turn store (Phase 1B).

* ``GET /sessions/{id}`` -- pretty-printable session view (recent
  episodes + their tail turns). Used by the playground "Inspect"
  chip to verify the Phase 1B preset story (e.g. ``state_json``
  should hold ``["apple","river"]`` after the two drill presets).
* ``POST /sessions/{id}/end`` -- mark the session and any open
  episodes as closed. The playground's "Reset session" button hits
  this before clearing its local ``ACTIVE_SESSION_ID``.

These are *not* part of the public mode-endpoint surface; they're
debug / runtime-orchestration affordances. Both routes return JSON;
neither runs inference.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..deps import get_session_manager
from ..memory import EpisodeView, Session, SessionManager, Turn

logger = logging.getLogger(__name__)
router = APIRouter()


def _turn_to_dict(turn: Turn) -> dict[str, Any]:
    usage: Any = None
    if turn.usage_json:
        try:
            usage = json.loads(turn.usage_json)
        except json.JSONDecodeError:
            usage = turn.usage_json
    return {
        "id": turn.id,
        "ts": turn.ts,
        "role": turn.role,
        "text": turn.text,
        "media_kind": turn.media_kind,
        "inference_ms": turn.inference_ms,
        "usage": usage,
    }


def _episode_view_to_dict(view: EpisodeView) -> dict[str, Any]:
    state: Any = {}
    if view.episode.state_json:
        try:
            state = json.loads(view.episode.state_json)
        except json.JSONDecodeError:
            state = {}
    return {
        "id": view.episode.id,
        "mode": view.episode.mode,
        "started_at": view.episode.started_at,
        "ended_at": view.episode.ended_at,
        "summary": view.episode.summary,
        "state_json": state,
        "turn_count": view.turn_count,
        "recent_turns": [_turn_to_dict(t) for t in view.recent_turns],
    }


def _session_to_dict(session: Session, episodes: list[EpisodeView]) -> dict[str, Any]:
    meta: Any = None
    if session.meta_json:
        try:
            meta = json.loads(session.meta_json)
        except json.JSONDecodeError:
            meta = session.meta_json
    return {
        "session_id": session.id,
        "learner_id": session.learner_id,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "rolling_summary": session.rolling_summary,
        "meta": meta,
        "episodes": [_episode_view_to_dict(v) for v in episodes],
    }


@router.get("/sessions/{session_id}", summary="Inspect a session (ops)")
async def get_session(
    session_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Return the session + episode + recent-turn tail for ``session_id``.

    Shape (Phase 1B; ``summary`` and ``rolling_summary`` are reserved for
    Phase 2's LLM summariser):

    .. code-block:: json

        {
          "session_id": 7,
          "learner_id": "kalzy",
          "started_at": "...",
          "ended_at": null,
          "rolling_summary": null,
          "meta": null,
          "episodes": [
            {
              "id": 12,
              "mode": "voice_mirror",
              "started_at": "...",
              "ended_at": null,
              "summary": null,
              "state_json": {"words_drilled": ["apple", "river"]},
              "turn_count": 4,
              "recent_turns": [{"id": ..., "role": "user", ...}, ...]
            }
          ]
        }
    """
    result = await session_manager.get_session_view(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    session, episode_views = result
    return JSONResponse(_session_to_dict(session, episode_views))


@router.post(
    "/sessions/{session_id}/end",
    summary="Close a session and any in-flight episodes (ops)",
)
async def end_session(
    session_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Idempotent close. Returns ``{ended_at}``; 404s for unknown ids."""
    existing = await session_manager.get_session_view(session_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    ended_at = await session_manager.end_session(session_id)
    return JSONResponse({"session_id": session_id, "ended_at": ended_at})


__all__ = ["router"]
