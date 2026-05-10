"""HTTP surface for the TTS subsystem.

* ``GET /tts/voices`` -- list curated personalities (id, description,
  language, downloaded flag, sample rate).
* ``GET /tts/output/{audio_id}.wav`` -- serve a previously-rendered file
  from the TTL cache. Returns 404 once the TTL has elapsed (or never
  existed).
* ``POST /tts/speak`` -- one-shot TTS for arbitrary text. Useful for the
  playground and for clients that just want the audio without running an
  inference. Defaults to ``warm-academic``.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .engine import PiperEngine, PiperError
from .integration import attach_inline_tts
from .storage import TTSStorage
from .voices import DEFAULT_PERSONALITY

router = APIRouter()


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------


def _get_engine(request: Request) -> PiperEngine:
    engine: PiperEngine | None = getattr(request.app.state, "tts_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="TTS engine is not initialised on this server.",
        )
    return engine


def _get_storage(request: Request) -> TTSStorage:
    storage: TTSStorage | None = getattr(request.app.state, "tts_storage", None)
    if storage is None:
        raise HTTPException(
            status_code=503,
            detail="TTS storage is not initialised on this server.",
        )
    return storage


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


class VoiceInfo(BaseModel):
    id: str
    description: str
    language: str
    voice_id: str
    downloaded: bool
    sample_rate: int | None
    is_default: bool


class VoiceListResponse(BaseModel):
    default: str
    voices: list[VoiceInfo]


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str = Field(default=DEFAULT_PERSONALITY)


class SpeakResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    audio_base64: str | None
    audio_mime: str = "audio/wav"
    audio_duration_ms: float | None = None
    voice: str
    audio_error: str | None = None


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/tts/voices", response_model=VoiceListResponse, tags=["tts"])
async def list_voices(engine: PiperEngine = Depends(_get_engine)):
    return VoiceListResponse(
        default=engine.default_voice,
        voices=[VoiceInfo(**v) for v in engine.available_voices()],
    )


@router.get("/tts/output/{audio_id}.wav", tags=["tts"])
async def get_output(
    audio_id: str,
    storage: TTSStorage = Depends(_get_storage),
):
    path = storage.resolve(audio_id)
    if path is None:
        raise HTTPException(status_code=404, detail="audio not found or expired")
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=f"{audio_id}.wav",
    )


@router.post("/tts/speak", response_model=SpeakResponse, tags=["tts"])
async def speak(
    req: SpeakRequest,
    engine: PiperEngine = Depends(_get_engine),
):
    try:
        attach = await attach_inline_tts(req.text, engine=engine, voice=req.voice)
    except PiperError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SpeakResponse(
        audio_base64=attach.get("audio_base64"),  # type: ignore[arg-type]
        audio_mime=attach.get("audio_mime", "audio/wav"),  # type: ignore[arg-type]
        audio_duration_ms=attach.get("audio_duration_ms"),  # type: ignore[arg-type]
        voice=attach.get("voice", req.voice),  # type: ignore[arg-type]
        audio_error=attach.get("audio_error"),  # type: ignore[arg-type]
    )
