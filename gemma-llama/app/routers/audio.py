"""Audio tutor endpoints.

* ``POST /audio/listen`` -- "The Classroom Ear". Student speaks a question;
  Gemma 4 listens (via the natively-supported audio path that landed in
  llama.cpp in early April 2026) and replies in text. Supports optional
  NDJSON streaming via ``stream=true``.
* ``POST /audio/translate`` -- audio in language A, text in language B.
  Gemma 4's instruct training covers 35+ languages out of the box.
* ``POST /audio/transcribe`` -- verbatim speech-to-text in the source
  language. Defaults to ``temperature=0`` and ``thinking=false`` for
  determinism.

All routes accept ``multipart/form-data`` only. The uploaded audio is
transcoded once to mono 16 kHz WAV (the format mtmd's Gemma 4 audio encoder
consumes) by :func:`app.media_multipart.read_audio_upload`.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_usage
from ..media import build_user_content
from ..media_multipart import read_audio_upload
from ..model_config import ModelConfig
from ..prompts import (
    render_audio_listen_prompt,
    render_audio_transcribe_prompt,
    render_audio_translate_prompt,
)
from ..schemas import (
    AudioListenResponse,
    AudioTranscribeResponse,
    AudioTranslateResponse,
)
from ..tts import PiperEngine
from ..tts._router_helpers import (
    maybe_attach_file,
    maybe_attach_inline,
    maybe_wrap_stream,
)
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

router = APIRouter()


def _require_audio(cfg: ModelConfig) -> None:
    if not cfg.modalities.audio:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Active model '{cfg.short_name}' is not configured for audio "
                "input. Switch to model_configs/gemma4-e4b.yaml or "
                "gemma4-e2b.yaml."
            ),
        )


_HEARTBEAT_INTERVAL_S = 5.0


def _heartbeat_stream(
    adapter: LlamaAdapter,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    thinking: bool,
) -> AsyncIterator[str]:
    """Yield upstream NDJSON lines plus periodic ``heartbeat`` markers.

    Multiplexes the upstream token stream with a 5 s heartbeat ticker so
    the HTTP connection (and the user's hope) survive multi-minute audio
    preprocessing on Pi 5. Heartbeat lines look like
    ``{"type":"heartbeat","elapsed_ms":N}``. Final ``{"type":"error"}`` line
    is emitted on ``LlamaServerError``.
    """

    async def gen() -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        started = time.perf_counter()

        async def producer() -> None:
            try:
                async for line in adapter.chat_completion_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                ):
                    await queue.put(line)
            except LlamaServerError as exc:
                await queue.put(
                    json.dumps({"type": "error", "content": str(exc)}) + "\n"
                )
            except Exception as exc:  # noqa: BLE001
                await queue.put(
                    json.dumps({"type": "error", "content": f"{type(exc).__name__}: {exc}"}) + "\n"
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(producer())
        try:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL_S)
                except asyncio.TimeoutError:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    yield json.dumps({
                        "type": "heartbeat",
                        "elapsed_ms": elapsed_ms,
                    }) + "\n"
                    continue
                if line is None:
                    break
                yield line
        finally:
            if not task.done():
                task.cancel()

    return gen()


@router.post(
    "/audio/listen",
    response_model=AudioListenResponse,
    summary="The Classroom Ear - answer a student's spoken question",
)
async def listen(
    audio: UploadFile = File(..., description="Recording of the student's question."),
    max_tokens: int | None = Form(None, ge=1, le=4096),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    stream: bool = Form(False, description="Stream NDJSON tokens instead of waiting for the full response."),
    tts: bool = Form(False, description="If true, also render the answer via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see GET /tts/voices)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    _require_audio(cfg)

    audio_url = await read_audio_upload(audio)

    system_prompt = render_audio_listen_prompt(cfg)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                "Listen to my question and answer it.",
                audio_urls=[audio_url],
            ),
        },
    ]

    resolved_max_tokens = max_tokens or cfg.defaults.max_tokens
    resolved_temperature = (
        temperature if temperature is not None else cfg.defaults.temperature
    )

    if stream:
        upstream = _heartbeat_stream(
            adapter,
            messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            thinking=False,
        )
        body = maybe_wrap_stream(
            upstream,
            enabled=tts,
            voice=voice,
            engine=tts_engine,
        )
        return StreamingResponse(body, media_type="application/x-ndjson")

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = extract_content(response).strip()
    if not text:
        text = "I could not hear that clearly. Could you please ask your question again?"
    tts_attach = await maybe_attach_file(
        text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return AudioListenResponse(
        text=text,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )


@router.post(
    "/audio/translate",
    response_model=AudioTranslateResponse,
    summary="Translate spoken audio in language A to text in language B",
)
async def translate(
    audio: UploadFile = File(..., description="Recording in the source language."),
    target_language: str = Form(
        ..., description="Target language name or BCP-47 code, e.g. 'English' or 'es'."
    ),
    source_language: str | None = Form(
        None, description="Optional source language hint to disambiguate."
    ),
    max_tokens: int | None = Form(None, ge=1, le=4096),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    tts: bool = Form(False, description="If true, also render the translation via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see GET /tts/voices)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
):
    _require_audio(cfg)

    audio_url = await read_audio_upload(audio)

    system_prompt = render_audio_translate_prompt(
        cfg,
        target_language=target_language,
        source_language=source_language,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                f"Translate this audio into {target_language}.",
                audio_urls=[audio_url],
            ),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or cfg.defaults.max_tokens,
            temperature=temperature if temperature is not None else 0.2,
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = extract_content(response).strip()
    tts_attach = await maybe_attach_inline(
        text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
    )
    return AudioTranslateResponse(
        text=text,
        target_language=target_language,
        source_language=source_language,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )


@router.post(
    "/audio/transcribe",
    response_model=AudioTranscribeResponse,
    summary="Verbatim speech-to-text (no translation, no commentary)",
)
async def transcribe(
    audio: UploadFile = File(..., description="Recording to transcribe verbatim."),
    max_tokens: int | None = Form(None, ge=1, le=4096),
    temperature: float | None = Form(
        None,
        ge=0.0,
        le=2.0,
        description="Defaults to 0.0 for deterministic transcripts.",
    ),
    tts: bool = Form(False, description="If true, also render the transcript via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see GET /tts/voices)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    _require_audio(cfg)

    audio_url = await read_audio_upload(audio)

    system_prompt = render_audio_transcribe_prompt(cfg)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                "Transcribe this audio verbatim.",
                audio_urls=[audio_url],
            ),
        },
    ]

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens or cfg.defaults.max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            thinking=False,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = extract_content(response).strip()
    tts_attach = await maybe_attach_file(
        text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return AudioTranscribeResponse(
        text=text,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )
