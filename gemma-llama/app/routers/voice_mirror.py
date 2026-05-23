"""Voice-mirror (pronunciation practice) endpoints.

* ``POST /voice-mirror/suggest`` -- ask the tutor for the next word to
  practise plus a friendly spoken cue.
* ``POST /voice-mirror/score`` -- send the learner's spoken attempt and
  receive a verdict (``praise`` / ``correct`` / ``retry``) plus a short
  feedback line.
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
from ..prompts import (
    render_voice_mirror_score_prompt,
    render_voice_mirror_suggest_prompt,
)
from ..schemas import (
    VoiceMirrorScoreResponse,
    VoiceMirrorSuggestRequest,
    VoiceMirrorSuggestResponse,
)
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

logger = logging.getLogger(__name__)
router = APIRouter()


_VALID_VERDICTS = ("praise", "correct", "retry")


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_suggest(raw: str) -> dict[str, str]:
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Surface what the model actually returned so the operator can see
        # whether it was truncated, prefixed with prose, or wrapped in a
        # leftover ``<think>`` block.
        logger.warning(
            "voice-mirror/suggest: JSON parse failed; raw=%r",
            raw[:400],
        )
        # Fallback: treat the entire text as the cue and synthesise the word.
        word = (cleaned.split()[:1] or ["apple"])[0].strip(".,!?\"'").lower() or "apple"
        return {
            "word": word,
            "example_sentence": f"I like {word}.",
            "ipa_hint": "",
            "prompt_text": cleaned or f"Try saying: {word}.",
        }
    if not isinstance(data, dict):
        return {
            "word": "apple",
            "example_sentence": "I like apples.",
            "ipa_hint": "AP-uhl",
            "prompt_text": "Try saying: apple. AP-uhl.",
        }
    word = str(data.get("word") or "apple").strip().lower() or "apple"
    example = str(data.get("example_sentence") or f"I like {word}.").strip()
    ipa = str(data.get("ipa_hint") or "").strip()
    prompt_text = str(data.get("prompt_text") or "").strip()
    if not prompt_text:
        if ipa:
            prompt_text = f"Try saying: {word}. {ipa}."
        else:
            prompt_text = f"Try saying: {word}."
    return {
        "word": word,
        "example_sentence": example,
        "ipa_hint": ipa,
        "prompt_text": prompt_text,
    }


def _parse_score(raw: str, target_word: str) -> dict[str, object]:
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Most common cause on Pi 5 is hitting max_tokens mid-output (the
        # model emits a long ``<think>`` block, our stripper drops it, and
        # the JSON never started). Log the raw response so the operator
        # can tell truncation from genuine model confusion.
        logger.warning(
            "voice-mirror/score: JSON parse failed for target=%r; raw=%r",
            target_word, raw[:400],
        )
        return {
            "transcript": None,
            "verdict": "retry",
            "feedback_text": (
                f"I couldn't quite hear that. Let's try '{target_word}' once more."
            ),
        }
    if not isinstance(data, dict):
        return {
            "transcript": None,
            "verdict": "retry",
            "feedback_text": f"Let's try '{target_word}' once more.",
        }
    verdict = str(data.get("verdict") or "retry").strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "retry"
    transcript = data.get("transcript")
    if transcript is not None:
        transcript = str(transcript).strip() or None
    feedback = str(data.get("feedback_text") or "").strip()
    if not feedback:
        if verdict == "praise":
            feedback = f"Great job! That was a clean '{target_word}'."
        elif verdict == "correct":
            feedback = f"Good attempt. Try '{target_word}' a little slower."
        else:
            feedback = f"No worries. Let's try '{target_word}' again."
    return {
        "transcript": transcript,
        "verdict": verdict,
        "feedback_text": feedback,
    }


@router.post(
    "/voice-mirror/suggest",
    response_model=VoiceMirrorSuggestResponse,
    summary="Suggest the next word for the learner to practise",
)
async def suggest(
    req: VoiceMirrorSuggestRequest,
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    prompt = render_voice_mirror_suggest_prompt(
        cfg, level=req.level, history=req.history
    )
    try:
        response = await adapter.chat_completion(
            [{"role": "user", "content": prompt}],
            # 512 leaves headroom for any ``<think>`` preamble Gemma 4
            # emits despite ``enable_thinking: false`` -- 256 was tight
            # enough that the JSON envelope was truncated and we silently
            # fell back to the canned cue.
            max_tokens=512,
            temperature=0.7,
            response_format={"type": "json_object"},
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw = extract_content(response)
    parsed = _parse_suggest(raw)
    tts_attach = await maybe_attach_file(
        parsed["prompt_text"],
        enabled=req.tts,
        voice=req.voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return VoiceMirrorSuggestResponse(
        word=parsed["word"],
        example_sentence=parsed["example_sentence"],
        ipa_hint=parsed["ipa_hint"] or None,
        prompt_text=parsed["prompt_text"],
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        **tts_attach,
    )


@router.post(
    "/voice-mirror/score",
    response_model=VoiceMirrorScoreResponse,
    summary="Score a learner's pronunciation attempt for a target word",
)
async def score(
    audio: UploadFile = File(..., description="Recording of the learner's attempt."),
    target_word: str = Form(..., description="The word the learner was asked to say."),
    # 512 default leaves headroom for the ``<think>`` preamble that Gemma
    # 4's chat template emits despite ``enable_thinking: false`` -- 256
    # truncated the JSON envelope and forced the "couldn't quite hear"
    # fallback for every preset.
    max_tokens: int | None = Form(None, ge=1, le=2048),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    tts: bool = Form(False, description="If true, also render the feedback via Piper TTS."),
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

    target = target_word.strip()
    if not target:
        raise HTTPException(status_code=422, detail="target_word must be non-empty")

    audio_url = await read_audio_upload(audio)

    system_prompt = render_voice_mirror_score_prompt(cfg, target_word=target)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                f"My attempt at the word '{target}' is attached.",
                audio_urls=[audio_url],
            ),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or 512,
            temperature=temperature if temperature is not None else 0.2,
            response_format={"type": "json_object"},
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw = extract_content(response)
    parsed = _parse_score(raw, target)
    tts_attach = await maybe_attach_file(
        str(parsed["feedback_text"]),
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return VoiceMirrorScoreResponse(
        target_word=target,
        transcript=parsed.get("transcript"),
        verdict=parsed["verdict"],
        feedback_text=str(parsed["feedback_text"]),
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )
