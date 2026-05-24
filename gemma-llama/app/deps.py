"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Request

from .llama_adapter import LlamaAdapter
from .memory import ContextEngine, EpisodeSummariser, SessionManager
from .model_config import ModelConfig
from .tts import PiperEngine
from .tts.storage import TTSStorage


def get_adapter(request: Request) -> LlamaAdapter:
    return request.app.state.adapter


def get_model_config(request: Request) -> ModelConfig:
    return request.app.state.model_config


def get_context_engine(request: Request) -> ContextEngine:
    """Return the process-wide :class:`ContextEngine`.

    Built once in :func:`app.main.lifespan` so the underlying
    :class:`LearnerLoader`'s mtime cache survives across requests.
    """
    return request.app.state.context_engine


def get_session_manager(request: Request) -> SessionManager:
    """Return the process-wide :class:`SessionManager` (Phase 1B).

    Built once in :func:`app.main.lifespan` against the open
    :class:`~app.memory.Database`. All mode routers funnel session /
    episode / turn writes through this single instance so the
    coarse asyncio lock in :class:`~app.memory.Database` actually
    serialises every writer in the process.
    """
    return request.app.state.session_manager


def get_episode_summariser(request: Request) -> EpisodeSummariser:
    """Return the process-wide :class:`EpisodeSummariser` (Phase 2).

    Built once in :func:`app.main.lifespan` so the LlamaAdapter +
    SessionManager wiring is shared with the mode routers. Routers
    call :meth:`EpisodeSummariser.maybe_close_episode` after recording
    their last turn; when the ``X-Episode-Hint`` header asked for a
    close, the summariser writes ``episode.summary`` (and, every Nth
    close, refreshes ``session.rolling_summary``).
    """
    return request.app.state.episode_summariser


def get_tts_engine(request: Request) -> PiperEngine | None:
    """Return the process-wide ``PiperEngine`` (or ``None`` if disabled)."""
    return getattr(request.app.state, "tts_engine", None)


def get_tts_storage(request: Request) -> TTSStorage | None:
    """Return the process-wide ``TTSStorage`` (or ``None`` if disabled)."""
    return getattr(request.app.state, "tts_storage", None)
