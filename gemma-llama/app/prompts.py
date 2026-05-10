"""Jinja2-based prompt rendering helpers.

Templates live in the active model YAML config under ``prompts.{classify,
extract, summarize}``. We render them in a sandboxed environment with a small,
explicit set of variables to keep behaviour predictable across models.
"""
from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, StrictUndefined

from .model_config import ModelConfig

_env = Environment(
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)
_env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)


def render_classify_prompt(
    cfg: ModelConfig, *, text: str, labels: list[str], multi_label: bool
) -> str:
    template = _env.from_string(cfg.prompts.classify)
    return template.render(text=text, labels=labels, multi_label=multi_label)


def render_extract_prompt(
    cfg: ModelConfig, *, text: str, schema: dict[str, Any]
) -> str:
    template = _env.from_string(cfg.prompts.extract)
    return template.render(text=text, schema=schema)


def render_summarize_prompt(
    cfg: ModelConfig, *, text: str, style: str | None, max_sentences: int
) -> str:
    template = _env.from_string(cfg.prompts.summarize)
    return template.render(text=text, style=style or "", max_sentences=max_sentences)


def render_vision_explain_prompt(
    cfg: ModelConfig, *, subject: str | None, question: str | None
) -> str:
    template = _env.from_string(cfg.prompts.vision_explain)
    return template.render(subject=subject or "", question=question or "")


def render_audio_listen_prompt(cfg: ModelConfig) -> str:
    template = _env.from_string(cfg.prompts.audio_listen)
    return template.render()


def render_audio_translate_prompt(
    cfg: ModelConfig, *, target_language: str, source_language: str | None
) -> str:
    template = _env.from_string(cfg.prompts.audio_translate)
    return template.render(
        target_language=target_language,
        source_language=source_language or "",
    )


def render_audio_transcribe_prompt(cfg: ModelConfig) -> str:
    template = _env.from_string(cfg.prompts.audio_transcribe)
    return template.render()


def render_video_lab_prompt(
    cfg: ModelConfig, *, task: str | None, include_audio: bool
) -> str:
    template = _env.from_string(cfg.prompts.video_lab)
    return template.render(task=task or "", include_audio=include_audio)
