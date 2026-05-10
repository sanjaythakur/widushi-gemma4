"""Model YAML config loader.

The YAML schema is documented in `model_configs/*.yaml` and parsed into the
:class:`ModelConfig` dataclass below. The active config path comes from
``Settings.model_config_path``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Modalities(BaseModel):
    text: bool = True
    image: bool = False
    audio: bool = False
    video: bool = False


class GenerationDefaults(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.7
    thinking: bool = True


_DEFAULT_VISION_EXPLAIN = (
    "You are a patient tutor reviewing a student's handwritten work in the "
    "attached image. Read every visible step, point out what is correct, "
    "identify the first mistake (if any) and explain why, and suggest the next "
    "step the student should try without solving the whole problem for them."
)
_DEFAULT_AUDIO_LISTEN = (
    "You are a friendly classroom tutor. The student's spoken question is "
    "attached as audio. Understand it, then answer at a level appropriate for "
    "a curious learner. Keep the answer concise, accurate, and conversational."
)
_DEFAULT_AUDIO_TRANSLATE = (
    "You are a precise translator. The attached audio is spoken "
    "{% if source_language %}in {{ source_language }}{% else %}in the speaker's "
    "native language{% endif %}. Transcribe and translate it faithfully into "
    "{{ target_language }}. Respond ONLY with the translated text - no preamble, "
    "no quotation marks, no commentary."
)
_DEFAULT_AUDIO_TRANSCRIBE = (
    "You are an automatic speech-recognition engine. Transcribe the attached "
    "audio verbatim into the SAME language it is spoken in. Preserve filler "
    "words and proper nouns; do not translate, summarise, or add commentary. "
    "Respond ONLY with the transcript - no preamble, no quotation marks, no "
    "speaker labels."
)
_DEFAULT_VIDEO_LAB = (
    "You are a lab assistant tutor watching a short clip provided as evenly "
    "spaced frames{% if include_audio %} and the original audio track{% endif %}."
    "{% if task %} The student's stated task: \"{{ task }}\".{% endif %} "
    "Respond as a JSON object with keys 'summary', 'observations' (list), "
    "'safety_notes' (list), and 'next_step'."
)


class PromptTemplates(BaseModel):
    classify: str
    extract: str
    summarize: str
    vision_explain: str = _DEFAULT_VISION_EXPLAIN
    audio_listen: str = _DEFAULT_AUDIO_LISTEN
    audio_translate: str = _DEFAULT_AUDIO_TRANSLATE
    audio_transcribe: str = _DEFAULT_AUDIO_TRANSCRIBE
    video_lab: str = _DEFAULT_VIDEO_LAB


class ModelConfig(BaseModel):
    """Parsed model config (one per file in ``model_configs/``)."""

    display_name: str
    short_name: str

    hf_repo: str
    hf_file: str
    mmproj_repo: str | None = None
    mmproj_file: str | None = None

    context_size: int = 8192
    threads: int = 4
    gpu_layers: int = 0

    modalities: Modalities = Field(default_factory=Modalities)
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    prompts: PromptTemplates


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Model config {path} must be a YAML mapping")
    return data


@lru_cache(maxsize=8)
def load_model_config(path: str) -> ModelConfig:
    """Load and validate a model YAML config from ``path``.

    Resolved relative to the current working directory if not absolute.
    Cached to avoid re-parsing on every request.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"Model config not found: {p}")
    return ModelConfig.model_validate(_load_yaml(p))
