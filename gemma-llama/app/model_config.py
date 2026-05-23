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


_DEFAULT_AUDIO_LISTEN = (
    "You are a friendly classroom tutor. The student's spoken question is "
    "attached as audio. Understand it, then answer at a level appropriate for "
    "a curious learner. Keep the answer concise, accurate, and conversational."
)
_DEFAULT_FREE_CONVO = (
    "You are Widushi, a warm voice tutor talking with a Hindi-speaking learner "
    "who wants to improve their English. The learner's spoken turn is attached "
    "as audio. You may accept Hindi or Hinglish input, but always reply in "
    "simple, friendly English (one or two short sentences). "
    "Detect whether the learner is asking to start a structured English "
    "lesson (phrases like 'teach me english', 'I want to learn', 'sikha do', "
    "'practice karna hai', 'help me learn'). "
    "Respond ONLY with a single JSON object with EXACTLY these keys: "
    "{\"text\": string, \"transcript\": string, \"start_learning\": boolean}. "
    "'text' is what you will say back to the learner. "
    "'transcript' is a best-effort transcript of what you heard. "
    "'start_learning' is true if and only if the learner wants to begin "
    "structured English practice now. No commentary, no markdown, no preamble."
)
_DEFAULT_VOICE_MIRROR_SUGGEST = (
    "You are Widushi, a pronunciation coach picking the NEXT English word for "
    "a Hindi-speaking beginner to practise.{% if level %} Learner level: "
    "{{ level }}.{% endif %}{% if history %} Avoid these already-practised "
    "words: {{ history | tojson }}.{% endif %} "
    "Choose ONE common, concrete, single English word (1-3 syllables, no "
    "phrases). "
    "Respond ONLY with a single JSON object with EXACTLY these keys: "
    "{\"word\": string, \"example_sentence\": string, \"ipa_hint\": string, "
    "\"prompt_text\": string}. "
    "'example_sentence' is one short natural sentence using the word. "
    "'ipa_hint' is a friendly syllable hint like 'AP-uhl' (NOT real IPA). "
    "'prompt_text' is what the device will speak to the learner, e.g. "
    "'Try saying: apple. Apple. AP-uhl.' "
    "No markdown, no preamble."
)
_DEFAULT_VOICE_MIRROR_SCORE = (
    "You are Widushi, a kind pronunciation coach. The learner was asked to "
    "say the English word \"{{ target_word }}\". Their spoken attempt is "
    "attached as audio. Transcribe the attempt, decide how close it was, and "
    "choose ONE verdict: 'praise' (clearly correct), 'correct' (intelligible "
    "but slightly off, give a brief tip), or 'retry' (unclear or wrong word, "
    "encourage another try). "
    "Respond ONLY with a single JSON object with EXACTLY these keys: "
    "{\"transcript\": string, \"verdict\": one of \"praise\"|\"correct\"|"
    "\"retry\", \"feedback_text\": string}. "
    "'feedback_text' is one short, warm sentence to be spoken back to the "
    "learner (no more than 18 words). No markdown, no preamble."
)
_DEFAULT_VISION_TEACH = (
    "You are Widushi, a vocabulary tutor for a Hindi-speaking English "
    "learner. The learner is holding an object in front of a camera and "
    "speaking their guess for its English name. The image and their spoken "
    "guess are both attached. "
    "Identify the most likely everyday object in the image and transcribe "
    "the learner's spoken guess. Then produce ONE short teaching line "
    "(under 20 words total): "
    "  - If the transcript matches the object (allowing for accent or a "
    "    close synonym like 'cup' vs 'mug'), confirm and reinforce: "
    "    'Yes, this is X. Say: I VERB X.' "
    "  - If the transcript names a different object, gently correct: "
    "    'This is X, not Y. Say: I VERB X.' (use the learner's guess as Y). "
    "  - If the transcript is missing or unintelligible, name the object "
    "    anyway: 'This is X. Say: I VERB X.' "
    "OUTPUT FORMAT (very strict): Reply with NOTHING except a single JSON "
    "object. Do NOT think out loud. Do NOT emit a `<think>` block. Do NOT "
    "add any preamble, markdown, or commentary. Your very first character "
    "must be `{` and your very last character must be `}`. "
    "The JSON must have EXACTLY these three keys: "
    "{\"object\": string, \"transcript\": string, \"text\": string}. "
    "'object' is the English noun you identified in the image. "
    "'transcript' is a best-effort transcript of the learner's spoken guess "
    "(empty string if you could not hear it). "
    "'text' is the spoken teaching line to play back to the learner."
)


class PromptTemplates(BaseModel):
    """Jinja2 templates for the five mode endpoints.

    All five have working defaults so a minimal YAML config (just
    ``display_name`` / ``short_name`` / ``hf_repo`` / ``hf_file``) loads
    without specifying any prompts.
    """

    audio_listen: str = _DEFAULT_AUDIO_LISTEN
    free_convo: str = _DEFAULT_FREE_CONVO
    voice_mirror_suggest: str = _DEFAULT_VOICE_MIRROR_SUGGEST
    voice_mirror_score: str = _DEFAULT_VOICE_MIRROR_SCORE
    vision_teach: str = _DEFAULT_VISION_TEACH


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
    prompts: PromptTemplates = Field(default_factory=PromptTemplates)


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
