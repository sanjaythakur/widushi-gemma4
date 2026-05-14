"""POST /video/analyze-process -- "The Lab Assistant".

The student records a short clip of a science experiment or a physical task
(e.g. titrating, soldering, threading a needle). We sample evenly spaced
frames + extract the original audio track via ffmpeg in the api container,
then send everything to Gemma 4 in a single multimodal turn.

Modality order matches the official Gemma 4 docs: frames -> audio -> text.

Supports optional NDJSON streaming via ``stream=true`` -- the streamed body
is the raw JSON-shaped tutor response token-by-token; clients reassemble the
full text before parsing.
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
from ..media_multipart import (
    DEFAULT_VIDEO_FRAMES,
    MAX_VIDEO_FRAMES,
    extract_video_payload,
)
from ..model_config import ModelConfig
from ..prompts import render_video_lab_prompt
from ..schemas import VideoAnalyzeResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file, maybe_wrap_stream
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

router = APIRouter()


def _humanise_lab_json(text: str) -> str:
    """Pretty-print the JSON-shaped lab tutor response as a speakable script.

    Falls back to the raw text if it does not parse, so transient JSON
    breakage from llama.cpp still gets *something* spoken instead of going
    silent.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(data, dict):
        return text

    parts: list[str] = []
    if (s := data.get("summary")):
        parts.append(f"Summary: {s}.")
    if (obs := data.get("observations")):
        if isinstance(obs, list) and obs:
            parts.append("Observations: " + " ".join(f"{o}." for o in obs))
        elif isinstance(obs, str):
            parts.append(f"Observations: {obs}.")
    if (notes := data.get("safety_notes")):
        if isinstance(notes, list) and notes:
            parts.append("Safety notes: " + " ".join(f"{n}." for n in notes))
        elif isinstance(notes, str):
            parts.append(f"Safety notes: {notes}.")
    if (nxt := data.get("next_step")):
        parts.append(f"Next step: {nxt}.")
    return " ".join(parts) if parts else text


@router.post(
    "/video/analyze-process",
    response_model=VideoAnalyzeResponse,
    summary="The Lab Assistant - analyse a short clip of a hands-on task",
)
async def analyze_process(
    video: UploadFile = File(..., description="Short clip of the experiment or task."),
    task: str | None = Form(None, description="Optional one-line description of what the student is trying to do."),
    n_frames: int = Form(
        DEFAULT_VIDEO_FRAMES,
        ge=1,
        le=MAX_VIDEO_FRAMES,
        description="Number of evenly spaced frames to sample (Pi 5 cap = 12).",
    ),
    include_audio: bool = Form(True, description="If true, also send the clip's audio track."),
    max_tokens: int | None = Form(None, ge=1, le=4096),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    thinking: bool = Form(
        False,
        description=(
            "If true, preserve and expose the model's chain-of-thought. "
            "Default off because reasoning roughly doubles the decoded "
            "token count on Pi 5 for the same final analysis."
        ),
    ),
    stream: bool = Form(
        False,
        description="Stream NDJSON tokens (no JSON-format enforcement -- clients must concat before parsing).",
    ),
    tts: bool = Form(False, description="If true, also render the spoken summary via Piper TTS."),
    voice: str = Form(DEFAULT_PERSONALITY, description="TTS personality id (see GET /tts/voices)."),
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    if not cfg.modalities.video:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Active model '{cfg.short_name}' is not configured for video. "
                "Switch to model_configs/gemma4-e4b.yaml or gemma4-e2b.yaml."
            ),
        )
    if include_audio and not cfg.modalities.audio:
        include_audio = False

    image_urls, audio_url, meta = await extract_video_payload(
        video, n_frames=n_frames, include_audio=include_audio
    )

    audios = [audio_url] if audio_url else []
    audio_used = bool(audios)

    system_prompt = render_video_lab_prompt(cfg, task=task, include_audio=audio_used)
    user_text = (
        f"Watch this {meta['frames_used']}-frame clip"
        + (" with audio" if audio_used else "")
        + (f" and analyse the task: {task}" if task else " and analyse the task.")
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                user_text,
                image_urls=image_urls,
                audio_urls=audios,
            ),
        },
    ]

    resolved_max_tokens = max_tokens or cfg.defaults.max_tokens
    resolved_temperature = (
        temperature if temperature is not None else cfg.defaults.temperature
    )

    if stream:
        # Note: we deliberately do NOT request response_format=json_object on
        # the streaming path. llama.cpp's grammar enforcement interleaves
        # poorly with chunked output and most clients can re-parse the final
        # concatenated string trivially.
        #
        # We multiplex the upstream token stream with a heartbeat ticker so
        # the connection (and the user's hope) survives the multi-minute
        # vision-projector encode that precedes the first token on Pi 5. The
        # heartbeat lines look like {"type":"heartbeat","elapsed_ms":N}.
        HEARTBEAT_INTERVAL_S = 5.0

        async def upstream() -> AsyncIterator[str]:
            yield json.dumps({
                "type": "meta",
                "content": {
                    "frames_used": meta["frames_used"],
                    "audio_used": audio_used,
                    "duration_s": meta.get("duration_s"),
                },
            }) + "\n"

            queue: asyncio.Queue[str | None] = asyncio.Queue()
            started = time.perf_counter()

            async def producer() -> None:
                try:
                    async for line in adapter.chat_completion_stream(
                        messages,
                        max_tokens=resolved_max_tokens,
                        temperature=resolved_temperature,
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
                        line = await asyncio.wait_for(
                            queue.get(), timeout=HEARTBEAT_INTERVAL_S
                        )
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

        body = maybe_wrap_stream(
            upstream(),
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
            thinking=thinking,
            response_format={"type": "json_object"},
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = extract_content(response)
    tts_attach = await maybe_attach_file(
        _humanise_lab_json(text),
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return VideoAnalyzeResponse(
        text=text,
        frames_used=meta["frames_used"],
        audio_used=audio_used,
        duration_s=meta.get("duration_s"),
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=extract_usage(response),
        **tts_attach,
    )
