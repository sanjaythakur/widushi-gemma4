"""Jinja2-based prompt rendering helpers.

Templates live in the active model YAML config under ``prompts.{free_convo,
voice_mirror_suggest, voice_mirror_score, vision_teach, audio_listen}``. We
render them in a sandboxed environment with a small, explicit set of
variables to keep behaviour predictable across models.
"""
from __future__ import annotations

import json

from jinja2 import Environment, StrictUndefined

from .model_config import ModelConfig

_env = Environment(
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)
_env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)


def render_audio_listen_prompt(cfg: ModelConfig) -> str:
    template = _env.from_string(cfg.prompts.audio_listen)
    return template.render()


def render_free_convo_prompt(cfg: ModelConfig) -> str:
    template = _env.from_string(cfg.prompts.free_convo)
    return template.render()


def render_voice_mirror_suggest_prompt(
    cfg: ModelConfig, *, level: str | None, history: list[str] | None
) -> str:
    template = _env.from_string(cfg.prompts.voice_mirror_suggest)
    return template.render(level=level or "", history=history or [])


def render_voice_mirror_score_prompt(
    cfg: ModelConfig, *, target_word: str
) -> str:
    template = _env.from_string(cfg.prompts.voice_mirror_score)
    return template.render(target_word=target_word)


def render_vision_teach_prompt(cfg: ModelConfig) -> str:
    template = _env.from_string(cfg.prompts.vision_teach)
    return template.render()
