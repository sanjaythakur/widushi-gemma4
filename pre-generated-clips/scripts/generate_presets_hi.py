"""Generate the non-clean-English preset audio files used by the
gemma-llama playground (see ../gemma-llama/api-presets/).

Covers four classes of utterance, all rendered via AI4Bharat's
`indic-parler-tts` so they sound like a real Hindi-speaking learner:

1. **Hindi (Devanagari)** -- intent / greetings / object names ("मुझे
   इंग्लिश सिखा दो।", "नमस्ते, आप कैसे हैं?", "दूध।", ...).
2. **Hinglish (Devanagari)** -- code-mixed turns like "ये इंग्लिश में
   क्या है?" with English transliterated into Devanagari so Parler-TTS
   doesn't trip over the script boundary.
3. **Indian-accented English** -- single English words ("Apple.",
   "Plate.") rendered with the Indian female voice description, which
   gives the credibly-accented attempt the VoiceMirror score endpoint
   expects to flag as `correct` rather than `praise`.
4. **A deliberately mumbled attempt** -- the same English word as (3)
   but rendered with a per-clip voice description that asks for a
   hesitant, indistinct, low-volume delivery, so VoiceMirror score
   should emit `retry`.

Sibling script `generate_presets_en.py` covers the clean US-English
presets via Piper.

Idempotent: skips presets whose `.wav` already exists with non-zero
size. Output is `22050 Hz` mono WAV per clip; the gemma-llama api
transcodes to mono 16 kHz on upload.

This model is gated on HuggingFace -- accept the agreement at
https://huggingface.co/ai4bharat/indic-parler-tts and pass `HF_TOKEN`.
Easiest way is the Makefile target: `HF_TOKEN=hf_xxx make presets-hi`.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from scipy.signal import resample_poly
from transformers import AutoTokenizer

MODEL_NAME = "ai4bharat/indic-parler-tts"

# Default voice prompt. Identical "Divya" persona used by the phase-1
# Hindi clips so all preset audio shares a consistent learner voice.
# Heavy emphasis on female-pitch cues because Parler-TTS is prompt-
# conditioned and a bare "warm voice" frequently drifts toward male
# timbre.
DEFAULT_DESCRIPTION = (
    "Divya, a young Indian woman, speaks in a clearly feminine, "
    "soft, slightly high-pitched voice with a warm, friendly tone, "
    "like a curious learner addressing her tutor. Her articulation is "
    "clear and her pace is moderate. The recording is very close, with "
    "almost no background noise."
)

# Override used for `voice-mirror-score/river-mumbled.wav`. Same speaker,
# but the description biases Parler-TTS toward a hesitant, indistinct
# delivery so the VoiceMirror score endpoint plausibly flags it as
# `retry` rather than `correct`.
MUMBLED_DESCRIPTION = (
    "Divya, a young Indian woman, speaks in a hesitant, mumbled, "
    "low-volume voice with unclear pronunciation. She trails off at "
    "the end of the word as if she is unsure what to say. The "
    "recording is very close, with almost no background noise."
)

# Each entry: (out_relpath_under_OUT_DIR, prompt_text, description_override_or_None).
# The relpaths line up 1:1 with the `files.audio` entries in
# ../gemma-llama/api-presets/manifest.json. All Devanagari inputs use
# native script -- Parler-TTS Indic was not trained on romanised Hindi.
PRESETS: list[tuple[str, str, str | None]] = [
    # ---------- FreeConvoMode ----------
    # Should set start_learning=true.
    ("free-convo/teach-me-english.wav",
     "मुझे इंग्लिश सिखा दो।",
     None),
    # Casual greeting, no learning intent.
    ("free-convo/greeting-hindi.wav",
     "नमस्ते, आप कैसे हैं?",
     None),
    # Hinglish chit-chat, no learning intent. "scene" transliterated to
    # "सीन" so the model stays inside one script.
    ("free-convo/hinglish-chitchat.wav",
     "आज मौसम अच्छा है, क्या सीन है?",
     None),

    # ---------- VoiceMirror score (learner attempts at the target word) ----------
    # Intelligible but Indian-accented -- expected verdict: correct.
    ("voice-mirror-score/apple-accented.wav",
     "Apple.",
     None),
    # Wrong word entirely (Hindi: "grape") -- expected verdict: retry.
    ("voice-mirror-score/apple-wrong-word.wav",
     "अंगूर।",
     None),
    # Mumbled / unclear attempt at "river" -- expected verdict: retry.
    ("voice-mirror-score/river-mumbled.wav",
     "River.",
     MUMBLED_DESCRIPTION),

    # ---------- VisionMode (learner says the noun while showing the object) ----------
    ("vision-teach/milk-doodh.wav",
     "दूध।",
     None),
    ("vision-teach/book-kitaab.wav",
     "किताब।",
     None),
    # Wrong English guess for the cup image -- model should gently correct.
    ("vision-teach/cup-wrong-guess.wav",
     "Plate.",
     None),

    # ---------- AudioListen / RolePlay ----------
    # Hinglish "what is this in English?" with "in English" transliterated.
    ("audio-listen/hinglish-question.wav",
     "ये इंग्लिश में क्या है?",
     None),
]

TARGET_SR = 22050
TARGET_CHANNELS = 1
SEED = 42  # reproducible sampling


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Clip and convert a float32 [-1, 1] array to int16 PCM."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    # Anti-aliased polyphase filter; handles ratios like 44100 -> 22050
    # cleanly without needing librosa.
    from math import gcd
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g)


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        frames = w.getnframes()
    if sr != TARGET_SR or ch != TARGET_CHANNELS:
        raise RuntimeError(
            f"{path}: expected {TARGET_SR} Hz mono, got {sr} Hz / {ch} ch"
        )
    if frames == 0:
        raise RuntimeError(f"{path}: empty wav (0 frames)")


def main() -> int:
    workdir = Path(os.environ.get("WORKDIR", "/work")).resolve()
    # OUT_DIR is the per-target override. Defaults to a folder inside the
    # sub-project for standalone dev runs; the Makefile points it at
    # ../gemma-llama/api-presets via a bind mount.
    out_dir = Path(os.environ.get("OUT_DIR", str(workdir / "api-presets-out"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"workdir = {workdir}")
    _log(f"out_dir = {out_dir}")

    # Skip-pass: if every wav already exists we don't need to load the
    # ~3 GB model (saves ~1 min of cold start on a laptop).
    pending = [
        (rel, t, d) for rel, t, d in PRESETS
        if not ((out_dir / rel).exists()
                and (out_dir / rel).stat().st_size > 0)
    ]
    if not pending:
        _log(f"Done (hi). generated=0 skipped={len(PRESETS)} total={len(PRESETS)}")
        return 0

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not hf_token:
        raise SystemExit(
            "ERROR: HF_TOKEN is not set. The model "
            f"'{MODEL_NAME}' is gated on HuggingFace.\n"
            "  1. Log in to HF and accept the agreement at\n"
            f"     https://huggingface.co/{MODEL_NAME}\n"
            "  2. Create a read token at\n"
            "     https://huggingface.co/settings/tokens\n"
            "  3. Re-run with `HF_TOKEN=hf_xxx make presets-hi`\n"
            "     (the Makefile passes it through to docker)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # CPU inference; bfloat16/fp16 only on CUDA
    _log(f"device = {device}")
    _log(f"loading model {MODEL_NAME} (first run downloads ~3 GB) ...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, token=hf_token
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path, token=hf_token
    )
    src_sr = int(model.config.sampling_rate)
    _log(f"model loaded. src_sr={src_sr}")

    # Cache the tokenised description per unique override so we don't
    # re-tokenise the same prompt for every clip that shares it.
    desc_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _desc_inputs(text: str) -> dict[str, torch.Tensor]:
        if text not in desc_cache:
            desc_cache[text] = description_tokenizer(text, return_tensors="pt").to(device)
        return desc_cache[text]

    torch.manual_seed(SEED)

    generated = 0
    skipped = len(PRESETS) - len(pending)

    for rel, prompt_text, description in PRESETS:
        wav_path = out_dir / rel
        if wav_path.exists() and wav_path.stat().st_size > 0:
            _log(f"[skip] {rel}")
            continue

        desc_text = description or DEFAULT_DESCRIPTION
        desc = _desc_inputs(desc_text)

        _log(f"[gen]  {rel} :: {prompt_text!r}"
             + ("  (mumbled voice)" if description else ""))
        prompt = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
            )
        audio = generation.to("cpu").to(torch.float32).numpy().squeeze()
        if audio.ndim != 1:
            audio = audio.mean(axis=0)  # collapse to mono if needed

        audio = _resample(audio, src_sr, TARGET_SR)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), _to_int16(audio), TARGET_SR, subtype="PCM_16")
        validate_wav(wav_path)
        generated += 1

    _log(
        f"Done (hi). generated={generated} skipped={skipped} "
        f"total={generated + skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
