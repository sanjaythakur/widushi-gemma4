"""Memory & context engine for the FastAPI gateway.

Phase 1A shipped the **learner-only** slice (file-backed Markdown
profiles, in-memory mtime cache, learner block rendered into every
mode's system prompt). Phase 1B layered on session / episode / turn
persistence backed by SQLite, an :class:`SessionManager` that owns all
writes, and two additional context blocks (episodic + working) which the
:class:`ContextEngine` renders into a user-message prelude. Phase 2 adds
explicit episode lifecycle (``X-Episode-Hint: close``) and an
LLM-driven :class:`EpisodeSummariser` that writes
``episode.summary`` + ``session.rolling_summary``; the episodic block
now prefers those compact summaries to raw turns when available.

* :class:`LearnerLoader` / :class:`LearnerProfile` -- Phase 1A.
* :class:`ContextEngine` -- learner block (1A) + episodic/working blocks
  (1B, now summary-aware in 2).
* :class:`Database` -- single-writer SQLite wrapper (1B).
* :class:`SessionManager`, :class:`Session`, :class:`Episode`, :class:`Turn`
  -- persistence layer (1B); gained ``close_episode`` /
  ``update_rolling_summary`` / counter helpers in 2.
* :class:`EpisodeSummariser`, :class:`SummariserConfig` -- LLM summariser
  (Phase 2).
* :func:`render_with_learner` -- thin Jinja shim, retained as a seam.
"""
from __future__ import annotations

from .context_engine import ContextEngine
from .db import Database
from .learner_loader import LearnerLoader, LearnerProfile
from .prompt_builder import render_with_learner
from .session_manager import Episode, EpisodeView, Session, SessionManager, Turn
from .summariser import EpisodeSummariser, SummariserConfig

__all__ = [
    "ContextEngine",
    "Database",
    "Episode",
    "EpisodeSummariser",
    "EpisodeView",
    "LearnerLoader",
    "LearnerProfile",
    "Session",
    "SessionManager",
    "SummariserConfig",
    "Turn",
    "render_with_learner",
]
