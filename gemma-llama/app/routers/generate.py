"""POST /generate -- single-prompt text generation, optionally multimodal."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_reasoning, extract_usage
from ..media import build_user_content
from ..model_config import ModelConfig
from ..schemas import GenerateRequest, GenerateResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file, maybe_wrap_stream
from ..tts.storage import TTSStorage

router = APIRouter()


def _build_messages(req: GenerateRequest) -> list[dict]:
    messages: list[dict] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": build_user_content(req.prompt, req.images)})
    return messages


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={200: {"description": "Generated text (or NDJSON stream when stream=true)"}},
)
async def generate(
    req: GenerateRequest,
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    max_tokens = req.max_tokens or cfg.defaults.max_tokens
    temperature = req.temperature if req.temperature is not None else cfg.defaults.temperature
    messages = _build_messages(req)

    if req.stream:
        async def upstream():
            try:
                async for line in adapter.chat_completion_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=req.thinking,
                ):
                    yield line
            except LlamaServerError as exc:
                yield ('{"type":"error","content":' f'"{str(exc)}"' "}\n")

        body = maybe_wrap_stream(
            upstream(),
            enabled=req.tts,
            voice=req.voice,
            engine=tts_engine,
        )
        return StreamingResponse(body, media_type="application/x-ndjson")

    try:
        response = await adapter.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=req.thinking,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    usage = extract_usage(response)
    text = extract_content(response)
    tts_attach = await maybe_attach_file(
        text,
        enabled=req.tts,
        voice=req.voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return GenerateResponse(
        text=text,
        thinking_content=extract_reasoning(response) or None,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        tokens_predicted=usage["tokens_predicted"],
        usage=usage,
        **tts_attach,
    )
