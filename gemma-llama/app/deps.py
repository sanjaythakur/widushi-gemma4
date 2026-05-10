"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Request

from .llama_adapter import LlamaAdapter
from .model_config import ModelConfig
from .tts import PiperEngine
from .tts.storage import TTSStorage


def get_adapter(request: Request) -> LlamaAdapter:
    return request.app.state.adapter


def get_model_config(request: Request) -> ModelConfig:
    return request.app.state.model_config


def get_tts_engine(request: Request) -> PiperEngine | None:
    """Return the process-wide ``PiperEngine`` (or ``None`` if disabled)."""
    return getattr(request.app.state, "tts_engine", None)


def get_tts_storage(request: Request) -> TTSStorage | None:
    """Return the process-wide ``TTSStorage`` (or ``None`` if disabled)."""
    return getattr(request.app.state, "tts_storage", None)
