"""POST /classify -- single- or multi-label text classification."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_adapter, get_model_config, get_tts_engine
from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content
from ..model_config import ModelConfig
from ..prompts import render_classify_prompt
from ..schemas import ClassifyRequest, ClassifyResponse
from ..tts import PiperEngine
from ..tts._router_helpers import maybe_attach_inline

logger = logging.getLogger(__name__)
router = APIRouter()


def _coerce_labels(raw: str, allowed: list[str], multi_label: bool) -> list[str]:
    """Best-effort recovery of label list from the model's JSON output."""
    candidates: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Some models forget the JSON wrapper; try splitting plain text.
        cleaned = raw.strip().strip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return [lbl for lbl in allowed if lbl.lower() in raw.lower()][: None if multi_label else 1]

    if isinstance(data, dict) and "labels" in data:
        value = data["labels"]
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [str(v) for v in value]
    elif isinstance(data, list):
        candidates = [str(v) for v in data]
    elif isinstance(data, str):
        candidates = [data]

    allowed_lc = {lbl.lower(): lbl for lbl in allowed}
    matched = [allowed_lc[c.lower()] for c in candidates if c.lower() in allowed_lc]
    if not multi_label:
        matched = matched[:1]
    return matched


def _spoken_labels(labels: list[str]) -> str:
    """Render the chosen labels as a speakable sentence."""
    if not labels:
        return "No matching label was identified."
    if len(labels) == 1:
        return f"Classified as {labels[0]}."
    head = ", ".join(labels[:-1])
    return f"Classified as {head}, and {labels[-1]}."


@router.post("/classify", response_model=ClassifyResponse)
async def classify(
    req: ClassifyRequest,
    adapter: LlamaAdapter = Depends(get_adapter),
    cfg: ModelConfig = Depends(get_model_config),
    tts_engine: PiperEngine | None = Depends(get_tts_engine),
):
    prompt = render_classify_prompt(
        cfg, text=req.text, labels=req.labels, multi_label=req.multi_label
    )
    max_tokens = req.max_tokens or 128
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
    labels = _coerce_labels(raw, req.labels, req.multi_label)
    tts_attach = await maybe_attach_inline(
        _spoken_labels(labels),
        enabled=req.tts,
        voice=req.voice,
        engine=tts_engine,
    )
    return ClassifyResponse(
        labels=labels,
        raw=raw,
        model=cfg.short_name,
        inference_time_ms=response.get("_inference_time_ms", 0.0),
        **tts_attach,
    )
