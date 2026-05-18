"""Vision tutor endpoints.

* ``POST /vision/explain-work`` -- "The Homework Checker". Accepts a photo of
  a student's notebook or whiteboard and returns tutor-style feedback.
* ``POST /vision/teach-object`` -- "VisionMode" turn. Accepts a camera frame
  plus the learner's spoken guess and replies with a short teaching line of
  the form ``"Yes, this is milk. Say: I drink milk."``.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_usage
from ..media import build_user_content
from ..media_multipart import read_audio_upload, read_image_upload
from ..model_config import ModelConfig
from ..prompts import render_vision_explain_prompt, render_vision_teach_prompt
from ..schemas import VisionExplainResponse, VisionTeachResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

logger = logging.getLogger(__name__)
router = APIRouter()


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_teach(raw: str) -> dict[str, object]:
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "object": None,
            "transcript": None,
            "text": cleaned or "Let's try again. Hold the object steady.",
        }
    if not isinstance(data, dict):
        return {
            "object": None,
            "transcript": None,
            "text": str(data),
        }
    obj = data.get("object")
    if obj is not None:
        obj = str(obj).strip() or None
    transcript = data.get("transcript")
    if transcript is not None:
        transcript = str(transcript).strip() or None
    text = str(data.get("text") or "").strip()
    if not text:
        if obj:
            text = f"Yes, this is {obj}. Say: I see a {obj}."
        else:
            text = "Let's try again. Hold the object steady."
    return {"object": obj, "transcript": transcript, "text": text}


@router.post(
    "/vision/explain-work",
    response_model=VisionExplainResponse,
    summary="The Homework Checker - explain a photo of student work",
)
async def explain_work(
    image: UploadFile = File(..., description="Photo of the student's notebook or whiteboard."),
    question: str | None = Form(None, description="Optional question the student is asking."),
    subject: str | None = Form(None, description="Optional subject hint, e.g. 'algebra'."),
    max_tokens: int | None = Form(None, ge=1, le=4096),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    thinking: bool = Form(
        False,
        description=(
            "If true, preserve and expose the model's chain-of-thought. "
            "Default off because reasoning roughly doubles the decoded "
            "token count on Pi 5 for the same final feedback."
        ),
    ),
    tts: bool = Form(False, description="If true, also render the response via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see GET /tts/voices)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    if not cfg.modalities.image:
        raise HTTPException(
            status_code=409,
            detail=f"Active model '{cfg.short_name}' is not configured for image input.",
        )

    image_url = await read_image_upload(image)

    system_prompt = render_vision_explain_prompt(
        cfg, subject=subject, question=question
    )
    user_text = (
        question
        or "Please review the work in this image and give me tutor-style feedback."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(user_text, image_urls=[image_url]),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or cfg.defaults.max_tokens,
            temperature=temperature if temperature is not None else cfg.defaults.temperature,
            thinking=thinking,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = extract_content(response)
    tts_attach = await maybe_attach_file(
        text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return VisionExplainResponse(
        text=text,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )


@router.post(
    "/vision/teach-object",
    response_model=VisionTeachResponse,
    summary="VisionMode turn -- teach an English noun from a camera frame + spoken guess",
)
async def teach_object(
    image: UploadFile = File(..., description="Camera frame of the object."),
    audio: UploadFile = File(..., description="Learner's spoken guess for the object."),
    max_tokens: int | None = Form(None, ge=1, le=2048),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    tts: bool = Form(False, description="If true, also render the teaching line via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see GET /tts/voices)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    if not cfg.modalities.image:
        raise HTTPException(
            status_code=409,
            detail=f"Active model '{cfg.short_name}' is not configured for image input.",
        )
    if not cfg.modalities.audio:
        raise HTTPException(
            status_code=409,
            detail=f"Active model '{cfg.short_name}' is not configured for audio input.",
        )

    image_url = await read_image_upload(image)
    audio_url = await read_audio_upload(audio)

    system_prompt = render_vision_teach_prompt(cfg)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                "Identify the object in the image and teach me the English word.",
                image_urls=[image_url],
                audio_urls=[audio_url],
            ),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or 256,
            temperature=temperature if temperature is not None else 0.3,
            response_format={"type": "json_object"},
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw = extract_content(response)
    parsed = _parse_teach(raw)
    tts_attach = await maybe_attach_file(
        str(parsed["text"]),
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return VisionTeachResponse(
        text=str(parsed["text"]),
        object=parsed.get("object"),
        transcript=parsed.get("transcript"),
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )
