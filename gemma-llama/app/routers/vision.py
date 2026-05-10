"""POST /vision/explain-work -- "The Homework Checker".

Accepts a photo of a student's notebook or whiteboard via ``multipart/form-data``
and returns a tutor-style explanation: what is correct, where the first mistake
is, and what to try next.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_usage
from ..media import build_user_content
from ..media_multipart import read_image_upload
from ..model_config import ModelConfig
from ..prompts import render_vision_explain_prompt
from ..schemas import VisionExplainResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

router = APIRouter()


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
            thinking=cfg.defaults.thinking,
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
