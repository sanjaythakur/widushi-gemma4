"""Generate the English-language preset audio files used by the gemma-llama
playground's "Load preset" dropdowns (see ../gemma-llama/api-presets/).

Uses Piper TTS with the same `en_US-lessac-medium` voice the api itself
ships with, so the preset clips sit in the same timbre/cadence the device
will speak back. Output is `22050 Hz` mono WAV, matching the rest of the
sub-project; the gemma-llama api transcodes everything to mono 16 kHz on
upload anyway, so the slightly higher source rate just gives a cleaner
in-browser `<audio>` preview.

Sibling script `generate_presets_hi.py` covers the Hindi / Hinglish /
Indian-accented-English / mumbled-attempt presets via Indic-Parler-TTS.

Idempotent: skips presets whose `.wav` already exists with non-zero size.

Run via the Makefile target (`make presets-en`), which builds the
`piper-clips` image, mounts this folder at `/work`, and mounts
`../gemma-llama/api-presets` at `/out` so the generated files land directly
in the api container's read-only preset volume.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

# (out_relpath_under_OUT_DIR, english_text) for every Piper-generated preset.
# The relpaths line up 1:1 with the `files.audio` entries in
# ../gemma-llama/api-presets/manifest.json.
PRESETS: list[tuple[str, str]] = [
    # FreeConvoMode -- English intent that should flip start_learning=true.
    ("free-convo/i-want-to-learn-english.wav",
     "I want to learn English."),

    # AudioListen / RolePlayMode -- clean spoken questions / role-play turns.
    ("audio-listen/what-time-is-it.wav",
     "What time is it?"),
    ("audio-listen/roleplay-shopkeeper-greeting.wav",
     "Hello, I want to buy something."),
    ("audio-listen/roleplay-doctor-headache.wav",
     "I have a headache."),

    # VoiceMirror score -- the clean US-English baseline (expected praise).
    ("voice-mirror-score/apple-clean.wav",
     "Apple."),

    # VisionMode -- correct English guess paired with vision-teach/apple.jpg.
    ("vision-teach/apple-guess.wav",
     "Apple."),

    # ---------------------------------------------------------------------
    # Memory PRD presets (see ../gemma-llama/specs/memory.md). Each phase's
    # test plan references one or more of the clips below; the manifest
    # rows that wire them up land with that phase's PR.
    # ---------------------------------------------------------------------

    # Phase 1A -- learner profile reached the model. Kalzy's profile carries
    # her name + location, so these two questions are the cheapest possible
    # check that `learner_block` is hitting the prompt.
    ("free-convo/what-is-my-name.wav",
     "What is my name?"),
    ("free-convo/where-do-i-live.wav",
     "Where do I live?"),

    # Phase 1B -- multi-turn session continuity. Step 1 says something
    # opinionated; step 2 asks the model to recall it from the working
    # block. Wording is intentionally short so step 2's "I just said" is
    # unambiguous.
    ("free-convo/i-love-mountains.wav",
     "I love walking in the mountains."),
    ("free-convo/what-did-i-say-i-love.wav",
     "What did I just say I love?"),

    # Phase 1B / Phase 2 -- cross-episode recall ("we drilled apple, river
    # earlier"). Phase 1B uses the working-block version; Phase 2 reuses
    # the same clip after the summariser collapses the turns.
    ("audio-listen/words-practised-today.wav",
     "What words have we practised today?"),
    ("audio-listen/what-did-we-practise.wav",
     "What did we practise earlier?"),

    # Phase 3 -- post-onboarding "did you hear me" sanity ping for a brand
    # new learner_id. Re-used by the "switch back to Kalzy" preset to
    # demonstrate that the same audio routes differently per X-Learner-Id.
    ("free-convo/can-you-hear-me.wav",
     "Hi, can you hear me?"),

    # Phase 4 -- curriculum-block reached the model. "How do I order
    # coffee politely?" is the canonical Chapter 3 (Café) probe and is
    # also reused after the path flips to Accelerated so the operator can
    # eyeball the change in reply density.
    ("audio-listen/how-do-i-order-coffee.wav",
     "How do I order coffee politely?"),

    # Phase 5 -- learner asks the tutor to surface their stumble history.
    # The expected reply names words written to `skill_note`, proving
    # retrieval reached the prompt.
    ("audio-listen/what-words-wrong.wav",
     "What words do I keep getting wrong?"),
]

VOICE = "en_US-lessac-medium"

_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICE_URL = f"{_HF_BASE}/en/en_US/lessac/medium"

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` atomically (write to .part, then rename)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    _log(f"[dl]   {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def ensure_model(name: str, base_url: str, models_dir: Path) -> Path:
    """Ensure `<name>.onnx` and `<name>.onnx.json` exist in `models_dir`.

    Shares the Piper voice cache with `generate_clips_en.py`, so when
    that script has already been run on the host (the usual case) no
    download happens.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        target = models_dir / f"{name}{suffix}"
        if target.exists() and target.stat().st_size > 0:
            continue
        _download(f"{base_url}/{name}{suffix}", target)
    return models_dir / f"{name}.onnx"


def synth(text: str, model_path: Path, out_path: Path) -> None:
    """Synthesise `text` to `out_path` (22050 Hz mono wav) using Piper."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "piper",
        "--model", str(model_path),
        "--output_file", str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"piper failed (exit {proc.returncode}) for text {text!r}"
        )


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        frames = w.getnframes()
    if sr != EXPECTED_SAMPLE_RATE or ch != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_SAMPLE_RATE} Hz mono, "
            f"got {sr} Hz / {ch} ch"
        )
    if frames == 0:
        raise RuntimeError(f"{path}: empty wav (0 frames)")


def main() -> int:
    workdir = Path(os.environ.get("WORKDIR", "/work")).resolve()
    models_dir = workdir / "models"
    # Default OUT_DIR puts files inside this sub-project (handy for
    # standalone dev runs); the Makefile target overrides it with the
    # gemma-llama api-presets bind mount so the wavs land where the
    # playground actually reads them.
    out_dir = Path(os.environ.get("OUT_DIR", str(workdir / "api-presets-out"))).resolve()
    _log(f"workdir    = {workdir}")
    _log(f"models_dir = {models_dir}")
    _log(f"out_dir    = {out_dir}")

    model_path = ensure_model(VOICE, VOICE_URL, models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for relpath, text in PRESETS:
        wav_path = out_dir / relpath
        if wav_path.exists() and wav_path.stat().st_size > 0:
            _log(f"[skip] {relpath}")
            skipped += 1
            continue
        _log(f"[gen]  {relpath} :: {text!r}")
        synth(text, model_path, wav_path)
        validate_wav(wav_path)
        generated += 1

    _log(
        f"Done (en). generated={generated} skipped={skipped} "
        f"total={generated + skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
