"""Audio tutor endpoint.

* ``POST /audio/listen`` -- "The Classroom Ear" (and Widushi's RolePlayMode
  driver). The learner speaks a question; Gemma 4 listens (via the natively-
  supported audio path that landed in llama.cpp in early April 2026) and
  replies in text. Supports optional NDJSON streaming via ``stream=true``.

Accepts ``multipart/form-data`` only. The uploaded audio is transcoded once
to mono 16 kHz WAV (the format mtmd's Gemma 4 audio encoder consumes) by
:func:`app.media_multipart.read_audio_upload`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

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
from ..media_multipart import read_audio_upload
from ..memory import (
    ContextEngine,
    Episode,
    EpisodeSummariser,
    Session,
    SessionManager,
)
from ..model_config import ModelConfig
from ..prompts import render_audio_listen_prompt
from ..schemas import AudioListenResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file, maybe_wrap_stream
from ..tts.storage import TTSStorage
from ..tts.voices import DEFAULT_PERSONALITY

router = APIRouter()

_MODE = "audio"


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


def _record_assistant_after_stream(
    body: AsyncIterator[str],
    *,
    session_manager: SessionManager,
    episode_summariser: EpisodeSummariser,
    session: Session,
    episode: Episode,
    episode_hint: str | None,
) -> AsyncIterator[str]:
    """Tee an NDJSON stream so the final assistant text gets persisted.

    Streaming responses don't carry a structured usage dict (llama.cpp
    emits per-token deltas), so we buffer ``{"type":"text","content":...}``
    lines into a single concatenated assistant text and record one turn
    row after the stream terminates. The TTS wrapper, when present,
    re-injects ``{"type":"audio",...}`` lines that we ignore for
    persistence purposes -- the text frames still pass through it
    unchanged.

    Phase 2: when the request carried ``X-Episode-Hint: close``, we also
    run the summariser *after* recording the assistant turn. The result
    is **not** echoed on the wire (we already chose the silent close
    behaviour for streaming responses) -- it lands on
    ``episode.summary`` and surfaces via ``GET /sessions/{id}``.
    """

    async def gen() -> AsyncIterator[str]:
        chunks: list[str] = []
        try:
            async for line in body:
                stripped = line.strip()
                if stripped:
                    try:
                        obj = json.loads(stripped)
                        if isinstance(obj, dict) and obj.get("type") == "text":
                            chunks.append(str(obj.get("content") or ""))
                    except json.JSONDecodeError:
                        pass
                yield line
        finally:
            text = "".join(chunks).strip()
            if text:
                try:
                    await session_manager.record_turn(
                        episode.id, role="assistant", text=text
                    )
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).exception(
                        "audio/listen: failed to persist streamed assistant turn"
                    )
            if episode_hint is not None:
                try:
                    await episode_summariser.maybe_close_episode(
                        session, episode, hint=episode_hint
                    )
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).exception(
                        "audio/listen: streaming episode close summariser failed"
                    )

    return gen()


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
    thinking: bool = Form(
        False,
        description=(
            "If true, preserve and expose the model's chain-of-thought. "
            "Default off for fastest responses; opt in for harder spoken "
            "questions where reasoning helps."
        ),
    ),
    stream: bool = Form(False, description="Stream NDJSON tokens instead of waiting for the full response."),
    tts: bool = Form(False, description="If true, also render the answer via Piper TTS."),
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
            "Phase 2 summariser. On streaming responses the summary is "
            "applied silently (no NDJSON event); inspect via "
            "``GET /sessions/{id}``."
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
    _require_audio(cfg)

    audio_url = await read_audio_upload(audio)

    resolved_learner_id, learner_block = context_engine.build_learner_block(x_learner_id)
    session = await session_manager.open_or_resume_session(
        resolved_learner_id, x_session_id
    )
    episode = await session_manager.open_or_resume_episode(session.id, _MODE)
    user_prelude = await context_engine.build_user_prelude(session.id, episode.id)

    system_prompt = render_audio_listen_prompt(cfg, learner_block=learner_block)
    base_instruction = "Listen to my question and answer it."
    user_text = (
        f"{user_prelude}\n\n{base_instruction}" if user_prelude else base_instruction
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(
                user_text,
                audio_urls=[audio_url],
            ),
        },
    ]

    resolved_max_tokens = max_tokens or cfg.defaults.max_tokens
    resolved_temperature = (
        temperature if temperature is not None else cfg.defaults.temperature
    )

    # Record the user turn up-front: even when the stream half-fails we
    # still want the "learner asked something" event in the timeline.
    await session_manager.record_turn(
        episode.id, role="user", text=None, media_kind="audio"
    )

    if stream:
        upstream = _heartbeat_stream(
            adapter,
            messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            thinking=thinking,
        )
        body = maybe_wrap_stream(
            upstream,
            enabled=tts,
            voice=voice,
            engine=tts_engine,
        )
        # Wrap the (possibly TTS-wrapped) stream so we can buffer the
        # assistant text and persist a turn row once the upstream
        # completes. Streaming responses don't carry our usage dict, so
        # the turn is logged with `inference_ms=None, usage=None`. When
        # ``X-Episode-Hint=close`` is set we *also* run the Phase 2
        # summariser silently inside the same finally block.
        body = _record_assistant_after_stream(
            body,
            session_manager=session_manager,
            episode_summariser=episode_summariser,
            session=session,
            episode=episode,
            episode_hint=x_episode_hint,
        )
        return StreamingResponse(
            body,
            media_type="application/x-ndjson",
            headers={
                "X-Session-Id": str(session.id),
                "X-Episode-Id": str(episode.id),
            },
        )

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            thinking=thinking,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = extract_content(response).strip()
    if not text:
        text = "I could not hear that clearly. Could you please ask your question again?"
    inference_ms = int(response.get("_inference_time_ms", 0.0))
    usage = extract_usage(response)

    await session_manager.record_turn(
        episode.id,
        role="assistant",
        text=text,
        inference_ms=inference_ms,
        usage=usage,
    )

    # Phase 2: honour X-Episode-Hint=close after the turn is persisted.
    episode_summary = await episode_summariser.maybe_close_episode(
        session, episode, hint=x_episode_hint
    )

    tts_attach = await maybe_attach_file(
        text,
        enabled=tts,
        voice=voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return AudioListenResponse(
        text=text,
        learner_id=resolved_learner_id,
        session_id=session.id,
        episode_id=episode.id,
        episode_summary=episode_summary,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        usage=usage,
        **tts_attach,
    )
