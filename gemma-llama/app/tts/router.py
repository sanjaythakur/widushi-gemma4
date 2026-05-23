"""HTTP surface for the TTS subsystem.

Only one route remains: serving previously-rendered WAVs from the TTL
cache, so that the ``audio_url`` returned by every mode endpoint (when
``tts=true``) can actually be fetched by the device / playground.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from .storage import TTSStorage

router = APIRouter()


def _get_storage(request: Request) -> TTSStorage:
    storage: TTSStorage | None = getattr(request.app.state, "tts_storage", None)
    if storage is None:
        raise HTTPException(
            status_code=503,
            detail="TTS storage is not initialised on this server.",
        )
    return storage


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
