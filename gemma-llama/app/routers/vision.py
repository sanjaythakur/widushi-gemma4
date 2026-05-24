"""Vision tutor endpoint.

* ``POST /vision/teach-object`` -- "VisionMode" turn. Accepts a camera frame
  plus the learner's spoken guess and replies with a short teaching line of
  the form ``"Yes, this is milk. Say: I drink milk."``.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from ..deps import (
    get_adapter,
    get_context_engine,
    get_episode_summariser,
    get_model_config,
    get_session_manager,
    get_tts_engine,
    get_tts_storage,
)
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_usage
from ..media import build_user_content
from ..media_multipart import read_audio_upload, read_image_upload
from ..memory import ContextEngine, EpisodeSummariser, SessionManager
from ..model_config import ModelConfig
from ..prompts import render_vision_teach_prompt
from ..schemas import VisionTeachResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

logger = logging.getLogger(__name__)
router = APIRouter()

_MODE = "vision"


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
        # Most common failure mode is hitting max_tokens before the JSON
        # closes (Gemma 4 frequently emits a `<think>` preamble even with
        # ``enable_thinking: false`` and our stripper drops the
        # unterminated tail). Log the raw response so we can tell
        # truncation from a genuinely confused model.
        logger.warning(
            "vision/teach-object: JSON parse failed; raw=%r", raw[:400]
        )
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
    "/vision/teach-object",
    response_model=VisionTeachResponse,
    summary="VisionMode turn -- teach an English noun from a camera frame + spoken guess",
)
async def teach_object(
    image: UploadFile = File(..., description="Camera frame of the object."),
    audio: UploadFile = File(..., description="Learner's spoken guess for the object."),
    # 1024 default because /vision/teach-object is the most token-hungry
    # JSON-mode endpoint: two media inputs PLUS Gemma 4's chat template
    # insists on emitting a ``<think>`` preamble despite
    # ``enable_thinking: false``. With 512 we were burning the entire
    # budget inside the open think block and the JSON envelope never
    # started, forcing the "Let's try again..." fallback.
    max_tokens: int | None = Form(None, ge=1, le=2048),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    tts: bool = Form(False, description="If true, also render the teaching line via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id."),
    x_learner_id: str | None = Header(
        default=None,
        alias="X-Learner-Id",
        description="Learner profile id; defaults to 'kalzy' (Phase 1A).",
    ),
    x_session_id: int | None = Header(
        default=None,
        alias="X-Session-Id",
        description="Session row id to resume; server-generated when absent (Phase 1B).",
    ),
    x_episode_hint: str | None = Header(
        default=None,
        alias="X-Episode-Hint",
        description=(
            "Runtime-orchestrator lifecycle hint. ``close`` closes the "
            "active episode after this turn is recorded and runs the "
            "Phase 2 summariser."
        ),
    ),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    context_engine: ContextEngine = Depends(get_context_engine),
    session_manager: SessionManager = Depends(get_session_manager),
    episode_summariser: EpisodeSummariser = Depends(get_episode_summariser),
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

    resolved_learner_id, learner_block = context_engine.build_learner_block(x_learner_id)
    session = await session_manager.open_or_resume_session(
        resolved_learner_id, x_session_id
    )
    episode = await session_manager.open_or_resume_episode(session.id, _MODE)
    user_prelude = await context_engine.build_user_prelude(session.id, episode.id)

    system_prompt = render_vision_teach_prompt(cfg, learner_block=learner_block)
    base_instruction = "Identify the object in the image and teach me the English word."
    user_text = (
        f"{user_prelude}\n\n{base_instruction}" if user_prelude else base_instruction
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                user_text,
                image_urls=[image_url],
                audio_urls=[audio_url],
            ),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or 1024,
            temperature=temperature if temperature is not None else 0.3,
            response_format={"type": "json_object"},
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw = extract_content(response)
    parsed = _parse_teach(raw)
    transcript = parsed.get("transcript")
    teach_text = str(parsed["text"])
    inference_ms = int(response.get("_inference_time_ms", 0.0))
    usage = extract_usage(response)

    await session_manager.record_turn(
        episode.id,
        role="user",
        text=transcript if isinstance(transcript, str) else None,
        media_kind="audio+image",
    )
    await session_manager.record_turn(
        episode.id,
        role="assistant",
        text=teach_text,
        inference_ms=inference_ms,
        usage=usage,
    )

    # Phase 2: honour X-Episode-Hint=close after the turn is persisted.
    episode_summary = await episode_summariser.maybe_close_episode(
        session, episode, hint=x_episode_hint
    )

    tts_attach = await maybe_attach_file(
        teach_text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return VisionTeachResponse(
        text=teach_text,
        object=parsed.get("object"),
        transcript=transcript,
        learner_id=resolved_learner_id,
        session_id=session.id,
        episode_id=episode.id,
        episode_summary=episode_summary,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=usage,
        **tts_attach,
    )
