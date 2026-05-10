"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .media import ImageInput
from .tts.voices import DEFAULT_PERSONALITY


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    llama_server_reachable: bool
    tts_ready: bool = False


# ---------------------------------------------------------------------------
# TTS opt-in mixin (composed into every request schema below)
# ---------------------------------------------------------------------------


class TTSOptions(BaseModel):
    """Mixin: opt into Piper TTS audio output on the response.

    ``tts=True`` triggers audio rendering. ``voice`` selects the curated
    personality (see :mod:`app.tts.voices`). The default personality is
    ``warm-academic``.
    """

    tts: bool = Field(
        default=False,
        description="If true, also render the response text via Piper TTS.",
    )
    voice: str = Field(
        default=DEFAULT_PERSONALITY,
        description=(
            "Curated voice personality id. "
            "Call `GET /tts/voices` to list available ids."
        ),
    )


# ---------------------------------------------------------------------------
# Generate / Chat
# ---------------------------------------------------------------------------


class GenerateRequest(TTSOptions):
    prompt: str = Field(..., description="User prompt.")
    system: str | None = Field(default=None, description="Optional system message.")
    images: list[ImageInput] | None = Field(
        default=None, description="Optional image attachments (URL or base64)."
    )
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking: bool = True
    stream: bool = False


class GenerateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    thinking_content: str | None = None
    model: str
    inference_time_ms: float
    tokens_predicted: int
    usage: dict[str, int]
    # File-URL TTS attach fields (populated when request.tts=true).
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[ImageInput] | None = None


class ChatRequest(TTSOptions):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking: bool = True
    stream: bool = False


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


class ClassifyRequest(TTSOptions):
    text: str
    labels: list[str] = Field(..., min_length=2)
    multi_label: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking: bool = True


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    labels: list[str]
    raw: str
    model: str
    inference_time_ms: float
    # Inline-base64 TTS attach (short output).
    audio_base64: str | None = None
    audio_mime: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class ExtractRequest(TTSOptions):
    text: str
    schema_: dict[str, Any] = Field(
        ...,
        alias="schema",
        description="JSON Schema-style description of the fields to extract.",
    )
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking: bool = True

    model_config = ConfigDict(populate_by_name=True)


class ExtractResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    data: dict[str, Any]
    model: str
    inference_time_ms: float
    # File-URL TTS attach (long output, JSON pretty-printed before TTS).
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


class SummarizeRequest(TTSOptions):
    text: str
    style: str | None = Field(default=None, description="e.g. 'bullet', 'tldr', 'executive'")
    max_sentences: int = Field(default=3, ge=1, le=20)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking: bool = True


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    summary: str
    model: str
    inference_time_ms: float
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# Multimodal tutor endpoints (vision / audio / video)
# ---------------------------------------------------------------------------


class VisionExplainResponse(BaseModel):
    """Response for ``POST /vision/explain-work`` (the Homework Checker)."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Tutor feedback text.")
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


class AudioListenResponse(BaseModel):
    """Response for ``POST /audio/listen`` (the Classroom Ear)."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Tutor's spoken-question answer in text.")
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


class AudioTranscribeResponse(BaseModel):
    """Response for ``POST /audio/transcribe`` (verbatim speech-to-text)."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Verbatim transcript of the spoken audio.")
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


class AudioTranslateResponse(BaseModel):
    """Response for ``POST /audio/translate``."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Translated text in the target language.")
    target_language: str
    source_language: str | None = None
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    # Inline-base64 (translations are typically short).
    audio_base64: str | None = None
    audio_mime: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


class VideoAnalyzeResponse(BaseModel):
    """Response for ``POST /video/analyze-process`` (the Lab Assistant)."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(
        ..., description="JSON-shaped tutor response (summary, observations, ...)."
    )
    frames_used: int
    audio_used: bool
    duration_s: float | None = None
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# Generic 'not yet implemented' shape (kept for any future placeholder routes)
# ---------------------------------------------------------------------------


class NotImplementedResponse(BaseModel):
    error: Literal["not_implemented"] = "not_implemented"
    reason: str
    tracking: str = "specification.md section 16"
