"""POST /free-convo/turn -- one turn of FreeConvoMode.

The learner speaks freely; the tutor replies in friendly English and tags
whether the learner expressed an intent to begin structured English practice
(``start_learning``). The learning-app uses ``start_learning`` to decide
whether to switch into ``VoiceMirrorMode``.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_usage
from ..media import build_user_content
from ..media_multipart import read_audio_upload
from ..model_config import ModelConfig
from ..prompts import render_free_convo_prompt
from ..schemas import FreeConvoResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

logger = logging.getLogger(__name__)
router = APIRouter()


_LEARN_KEYWORDS = (
    "learn english",
    "teach me english",
    "teach english",
    "practice english",
    "english seekh",
    "english sikh",
    "sikha do",
    "sikhao",
    "padhao",
    "i want to learn",
    "help me learn",
)


def _heuristic_start_learning(text: str | None) -> bool:
    """Cheap keyword fallback used when the JSON parse fails entirely."""
    if not text:
        return False
    haystack = text.lower()
    return any(kw in haystack for kw in _LEARN_KEYWORDS)


def _parse_free_convo(raw: str) -> dict[str, object]:
    """Best-effort parse of the model's JSON reply.

    Falls back to a plausible shape when the model emits prose instead of
    JSON so the endpoint never 500s on a bad completion.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Same Pi-5 failure mode as voice-mirror/vision: a `<think>`
        # preamble truncated at max_tokens leaves us with nothing to
        # parse. Log the raw response so the operator can diagnose
        # without redeploying with extra prints.
        logger.warning(
            "free-convo/turn: JSON parse failed; raw=%r", raw[:400]
        )
        return {
            "text": raw.strip() or "I'm here. Could you say that again?",
            "transcript": None,
            "start_learning": _heuristic_start_learning(raw),
        }
    if not isinstance(data, dict):
        return {
            "text": str(data),
            "transcript": None,
            "start_learning": False,
        }
    text = str(data.get("text") or "").strip()
    if not text:
        text = "I'm here. Could you say that again?"
    transcript = data.get("transcript")
    if transcript is not None:
        transcript = str(transcript).strip() or None
    start_learning = bool(data.get("start_learning"))
    if not start_learning:
        start_learning = _heuristic_start_learning(transcript) or _heuristic_start_learning(text)
    return {
        "text": text,
        "transcript": transcript,
        "start_learning": start_learning,
    }


@router.post(
    "/free-convo/turn",
    response_model=FreeConvoResponse,
    summary="FreeConvoMode turn -- chat reply + start_learning intent flag",
)
async def free_convo_turn(
    audio: UploadFile = File(..., description="Recording of the learner's spoken turn."),
    max_tokens: int | None = Form(None, ge=1, le=4096),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    tts: bool = Form(False, description="If true, also render the reply via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see app/tts/voices.py)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    if not cfg.modalities.audio:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Active model '{cfg.short_name}' is not configured for audio "
                "input. Switch to a Gemma 4 audio-capable model config."
            ),
        )

    audio_url = await read_audio_upload(audio)

    system_prompt = render_free_convo_prompt(cfg)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                "Reply to my spoken turn.",
                audio_urls=[audio_url],
            ),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or cfg.defaults.max_tokens,
            temperature=temperature if temperature is not None else 0.4,
            response_format={"type": "json_object"},
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw = extract_content(response)
    parsed = _parse_free_convo(raw)

    spoken_text = str(parsed["text"])
    tts_attach = await maybe_attach_file(
        spoken_text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return FreeConvoResponse(
        text=spoken_text,
        start_learning=bool(parsed["start_learning"]),
        transcript=parsed.get("transcript"),
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )
