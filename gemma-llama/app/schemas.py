"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
# TTS opt-in mixin (composed into JSON-body request schemas; multipart
# routes carry the same two form fields by hand on the function signature).
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
        description="Curated voice personality id.",
    )


# ---------------------------------------------------------------------------
# /audio/listen (RolePlayMode driver + generic spoken-question fallback)
# ---------------------------------------------------------------------------


class AudioListenResponse(BaseModel):
    """Response for ``POST /audio/listen``."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Tutor's spoken-question answer in text.")
    learner_id: str | None = Field(
        default=None,
        description=(
            "Resolved learner id used to build the system prompt (post-fallback "
            "to the default profile). Phase 1A introduces this field."
        ),
    )
    session_id: int | None = Field(
        default=None,
        description=(
            "SQLite session row id this turn was recorded under. Echoed back "
            "so the client (or playground) can stamp it onto subsequent "
            "requests via the ``X-Session-Id`` header. Phase 1B."
        ),
    )
    episode_id: int | None = Field(
        default=None,
        description=(
            "SQLite episode row id this turn was recorded under (one episode "
            "per ``(session, mode)`` until Phase 2 introduces explicit "
            "close hints). Phase 1B."
        ),
    )
    episode_summary: str | None = Field(
        default=None,
        description=(
            "If the request closed the episode (``X-Episode-Hint: close``), the "
            "LLM-written summary persisted to ``episode.summary``. Otherwise "
            "``None``. Phase 2."
        ),
    )
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# /free-convo/turn (FreeConvoMode)
#
# The learner speaks freely; Gemma replies and tags whether the user expressed
# an intent to start a structured English lesson (``start_learning``). The
# learning-app reads ``start_learning`` to switch into VoiceMirrorMode.
# ---------------------------------------------------------------------------


class FreeConvoResponse(BaseModel):
    """Response for ``POST /free-convo/turn``."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Tutor's spoken reply in text.")
    start_learning: bool = Field(
        default=False,
        description=(
            "True when the learner expressed an intent to begin structured "
            "English practice (e.g. 'teach me english', 'I want to learn')."
        ),
    )
    transcript: str | None = Field(
        default=None, description="Best-effort transcript of what the learner said."
    )
    learner_id: str | None = Field(
        default=None,
        description=(
            "Resolved learner id used to build the system prompt (post-fallback "
            "to the default profile). Phase 1A introduces this field."
        ),
    )
    session_id: int | None = Field(
        default=None,
        description=(
            "SQLite session row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_id: int | None = Field(
        default=None,
        description=(
            "SQLite episode row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_summary: str | None = Field(
        default=None,
        description=(
            "If the request closed the episode (``X-Episode-Hint: close``), the "
            "LLM-written summary persisted to ``episode.summary``. Otherwise "
            "``None``. Phase 2."
        ),
    )
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# /voice-mirror/* (VoiceMirrorMode: pronunciation practice)
# ---------------------------------------------------------------------------


class VoiceMirrorSuggestRequest(TTSOptions):
    level: str | None = Field(
        default=None,
        description="Optional learner level hint, e.g. 'beginner'.",
    )
    history: list[str] | None = Field(
        default=None,
        description="Words already practiced in this session (avoid repeats).",
    )


class VoiceMirrorSuggestResponse(BaseModel):
    """Response for ``POST /voice-mirror/suggest``."""

    model_config = ConfigDict(protected_namespaces=())

    word: str = Field(..., description="The target English word to practise.")
    example_sentence: str = Field(
        ..., description="A short example sentence using the word."
    )
    ipa_hint: str | None = Field(
        default=None,
        description="Optional IPA-style pronunciation hint (e.g. 'AP-uhl').",
    )
    prompt_text: str = Field(
        ...,
        description=(
            "The spoken cue the device should play back, e.g. "
            "'Try saying: apple. A-P-P-L-E.'"
        ),
    )
    learner_id: str | None = Field(
        default=None,
        description=(
            "Resolved learner id used to build the system prompt (post-fallback "
            "to the default profile). Phase 1A introduces this field."
        ),
    )
    session_id: int | None = Field(
        default=None,
        description=(
            "SQLite session row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_id: int | None = Field(
        default=None,
        description=(
            "SQLite episode row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_summary: str | None = Field(
        default=None,
        description=(
            "If the request closed the episode (``X-Episode-Hint: close``), the "
            "LLM-written summary persisted to ``episode.summary``. Otherwise "
            "``None``. Phase 2."
        ),
    )
    model: str
    inference_time_ms: float
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


class VoiceMirrorScoreResponse(BaseModel):
    """Response for ``POST /voice-mirror/score``."""

    model_config = ConfigDict(protected_namespaces=())

    target_word: str
    transcript: str | None = Field(
        default=None, description="Best-effort transcript of the learner's attempt."
    )
    verdict: Literal["praise", "correct", "retry"] = Field(
        ..., description="Coarse outcome used to choose praise/correction/retry."
    )
    feedback_text: str = Field(
        ...,
        description="Short tutor feedback line to be spoken back to the learner.",
    )
    learner_id: str | None = Field(
        default=None,
        description=(
            "Resolved learner id used to build the system prompt (post-fallback "
            "to the default profile). Phase 1A introduces this field."
        ),
    )
    session_id: int | None = Field(
        default=None,
        description=(
            "SQLite session row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_id: int | None = Field(
        default=None,
        description=(
            "SQLite episode row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_summary: str | None = Field(
        default=None,
        description=(
            "If the request closed the episode (``X-Episode-Hint: close``), the "
            "LLM-written summary persisted to ``episode.summary``. Otherwise "
            "``None``. Phase 2."
        ),
    )
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# /vision/teach-object (VisionMode: camera frame + spoken guess)
# ---------------------------------------------------------------------------


class VisionTeachResponse(BaseModel):
    """Response for ``POST /vision/teach-object``."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(
        ...,
        description=(
            "Spoken teaching line, e.g. 'Yes, this is milk. Say: I drink milk.'"
        ),
    )
    object: str | None = Field(
        default=None, description="The object the tutor identified in the image."
    )
    transcript: str | None = Field(
        default=None, description="Best-effort transcript of the learner's guess."
    )
    learner_id: str | None = Field(
        default=None,
        description=(
            "Resolved learner id used to build the system prompt (post-fallback "
            "to the default profile). Phase 1A introduces this field."
        ),
    )
    session_id: int | None = Field(
        default=None,
        description=(
            "SQLite session row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_id: int | None = Field(
        default=None,
        description=(
            "SQLite episode row id this turn was recorded under. Phase 1B."
        ),
    )
    episode_summary: str | None = Field(
        default=None,
        description=(
            "If the request closed the episode (``X-Episode-Hint: close``), the "
            "LLM-written summary persisted to ``episode.summary``. Otherwise "
            "``None``. Phase 2."
        ),
    )
    model: str
    inference_time_ms: float
    usage: dict[str, int]
    audio_url: str | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None
    audio_error: str | None = None
