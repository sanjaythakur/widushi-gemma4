"""POST /extract -- structured-data extraction with JSON output."""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_adapter, get_model_config, get_tts_engine, get_tts_storage
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content
from ..model_config import ModelConfig
from ..prompts import render_extract_prompt
from ..schemas import ExtractRequest, ExtractResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_file
from ..tts.storage import TTSStorage

router = APIRouter()

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _try_parse_json(raw: str) -> tuple[dict | None, str | None]:
    cleaned = _FENCE_RE.sub("", raw).strip()
    if not cleaned:
        return None, "empty response"
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"
    if not isinstance(data, dict):
        return None, f"expected JSON object, got {type(data).__name__}"
    return data, None


def _humanise_for_tts(data: dict[str, Any]) -> str:
    """Turn a flat-ish JSON object into a speakable script.

    Skips internal ``_raw``/``_error`` keys and renders nested values as
    JSON so the speaker says "title colon Hello, comma, items colon a, b, c"
    instead of literal braces.
    """
    parts: list[str] = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        label = key.replace("_", " ")
        if value is None:
            parts.append(f"{label}: not available.")
        elif isinstance(value, (str, int, float, bool)):
            parts.append(f"{label}: {value}.")
        elif isinstance(value, list):
            joined = ", ".join(str(v) for v in value) if value else "none"
            parts.append(f"{label}: {joined}.")
        else:
            parts.append(f"{label}: {json.dumps(value, ensure_ascii=False)}.")
    return " ".join(parts) if parts else "No fields were extracted."


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    req: ExtractRequest,
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
    tts_storage: TTSStorage | None = Depends(get_tts_storage),
):
    prompt = render_extract_prompt(cfg, text=req.text, schema=req.schema_)
    max_tokens = req.max_tokens or cfg.defaults.max_tokens
    temperature = req.temperature if req.temperature is not None else 0.0

    try:
        response = await adapter.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            thinking=req.thinking,
        )
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw = extract_content(response)
    data, err = _try_parse_json(raw)
    if data is None:
        # Per spec: do not fail; surface the raw output and error so callers
        # can decide what to do.
        data = {"_raw": raw, "_error": err or "unknown parse failure"}

    tts_attach = await maybe_attach_file(
        _humanise_for_tts(data),
        enabled=req.tts,
        voice=req.voice,
        engine=tts_engine,
        storage=tts_storage,
    )
    return ExtractResponse(
        data=data,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        **tts_attach,
    )
