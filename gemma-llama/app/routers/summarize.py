"""POST /summarize -- abstractive text summarization."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content
from ..model_config import ModelConfig
from ..prompts import render_summarize_prompt
from ..schemas import SummarizeRequest, SummarizeResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    req: SummarizeRequest,
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    prompt = render_summarize_prompt(
        cfg, text=req.text, style=req.style, max_sentences=req.max_sentences
    )
    max_tokens = req.max_tokens or cfg.defaults.max_tokens
    temperature = req.temperature if req.temperature is not None else cfg.defaults.temperature

    try:
        response = await adapter.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=req.thinking,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    summary = extract_content(response).strip()
    tts_attach = await maybe_attach_file(
        summary,
        enabled=req.tts,
        voice=req.voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return SummarizeResponse(
        summary=summary,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        **tts_attach,
    )
