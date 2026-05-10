"""Generate the Hindi pre-recorded clips using AI4Bharat's Indic-Parler-TTS.

Why a separate engine: Piper Hindi is phonemizer-bound (espeak `hi`) and
sounds noticeably non-native, especially with romanized input. AI4Bharat's
`indic-parler-tts` is a Parler-TTS model fine-tuned on Indian languages
and produces markedly more natural Indian-Hindi prosody and accent.

The model is gated on HuggingFace: you must (1) accept the access
agreement at https://huggingface.co/ai4bharat/indic-parler-tts and
(2) provide an `HF_TOKEN` environment variable (read scope is enough)
when running this script / container.

Inputs are Devanagari (the model expects native script). Output is a
22050 Hz mono `.wav` per clip in `<WORKDIR>/clips/hi/<clip_id>.wav` —
matching the format used by the English Piper pipeline so the downstream
runtime doesn't care which engine produced each clip.

The HuggingFace model + tokenizers are cached under
`<WORKDIR>/models/.hf-cache` so the ~3 GB weights are only downloaded
once per host.

Idempotent: skips clips whose `.wav` already exists. CPU inference is
slow (≈10–30 s/clip on a laptop); a full 16-clip run takes a few
minutes once the model is cached.
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

# Single consistent voice description used for every clip so all 16
# utterances share timbre and pacing. "Divya" is one of the two
# highest-quality named female speakers AI4Bharat lists for Hindi
# (the other being Rani). The description leans hard on female
# gender / pitch cues because Parler-TTS is prompt-conditioned and a
# bare "warm voice" will frequently drift toward a male timbre.
VOICE_DESCRIPTION = (
    "Divya, a young Indian woman, speaks in a clearly feminine, "
    "soft, slightly high-pitched voice with a warm, friendly, "
    "encouraging tone, like a patient female teacher addressing a "
    "young child. Her articulation is very clear and her pace is "
    "moderate. The recording is very close, with almost no "
    "background noise."
)

# (clip_id, devanagari_text). Devanagari is required — Parler-TTS Indic
# was not trained on romanized Hindi. All first-person verb forms agree
# with a female speaker (e.g. `सुन रही हूँ`, not `सुन रहा हूँ`).
CLIPS: list[tuple[str, str]] = [
    ("ack_okay", "ठीक है।"),
    ("ack_got_it", "समझ गई।"),
    ("ack_on_it", "हो जाता है।"),
    ("ack_sure", "ज़रूर।"),
    ("greet_hello", "नमस्ते।"),
    ("greet_hi", "हाँ, बोलो।"),
    ("wait_moment", "एक सेकंड।"),
    ("wait_thinking", "सोच रही हूँ।"),
    ("wait_checking", "देख रही हूँ।"),
    ("done_here", "लो, ये रहा।"),
    ("done_set", "हो गया।"),
    ("error_repeat", "माफ़ी, दोबारा बोलोगे?"),
    ("error_unclear", "समझ नहीं आई।"),
    ("error_unknown", "मुझे पता नहीं।"),
    ("listen_start", "बोलो, मैं सुन रही हूँ।"),
    ("thanks", "शुक्रिया।"),
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
    # resample_poly uses an anti-aliased polyphase filter which is well
    # suited to integer ratios like 44100 -> 22050.
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
    out_dir = workdir / "clips" / "hi"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"workdir = {workdir}")
    _log(f"out_dir = {out_dir}")

    # Skip-pass: if every wav already exists we don't need to load the model.
    pending = [
        (cid, t) for cid, t in CLIPS
        if not ((out_dir / f"{cid}.wav").exists()
                and (out_dir / f"{cid}.wav").stat().st_size > 0)
    ]
    if not pending:
        _log(f"Done. generated=0 skipped={len(CLIPS)} total={len(CLIPS)}")
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
            "  3. Re-run with `HF_TOKEN=hf_xxx make clips-hi`\n"
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

    desc = description_tokenizer(VOICE_DESCRIPTION, return_tensors="pt").to(device)
    torch.manual_seed(SEED)

    generated = 0
    skipped = len(CLIPS) - len(pending)

    for clip_id, hi_text in CLIPS:
        wav_path = out_dir / f"{clip_id}.wav"
        rel = wav_path.relative_to(workdir)
        if wav_path.exists() and wav_path.stat().st_size > 0:
            _log(f"[skip] {rel}")
            continue

        _log(f"[gen]  {rel} :: {hi_text!r}")
        prompt = tokenizer(hi_text, return_tensors="pt").to(device)
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
        sf.write(str(wav_path), _to_int16(audio), TARGET_SR, subtype="PCM_16")
        validate_wav(wav_path)
        generated += 1

    _log(
        f"Done. generated={generated} skipped={skipped} "
        f"total={generated + skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
