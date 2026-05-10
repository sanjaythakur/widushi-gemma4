#!/usr/bin/env bash
# ============================================================================
# Idempotent Piper voice download for the api container.
#   - Reads TTS_VOICES_ENABLED (comma-separated personality ids defined in
#     app/tts/voices.py PERSONALITIES) and downloads the matching .onnx +
#     .onnx.json pairs from rhasspy/piper-voices on Hugging Face.
#   - Writes them flat into TTS_VOICES_DIR (default /voices) so the
#     PiperEngine can find them as <voice_id>.onnx / <voice_id>.onnx.json.
#   - Skips voices that already exist on the volume so subsequent boots are
#     near-instant.
#   - Failure to download a single voice is non-fatal: the engine just
#     marks it as "not present" and the /tts routes will 422 if requested.
# ============================================================================
set -uo pipefail

: "${TTS_VOICES_DIR:=/voices}"
: "${TTS_ENABLED:=true}"
: "${TTS_VOICES_ENABLED:=warm-academic,friendly-casual,neutral-news,energetic-kid,calm-storyteller}"

log() { printf '[download_voices] %s\n' "$*" >&2; }

if [ "${TTS_ENABLED,,}" != "true" ]; then
    log "TTS_ENABLED=${TTS_ENABLED}; skipping voice download"
    exit 0
fi

mkdir -p "$TTS_VOICES_DIR"

# Personality id -> upstream voice_id (mirrors app/tts/voices.py PERSONALITIES).
# Edit both files together if you change a mapping.
declare -A VOICE_MAP=(
    [warm-academic]="en_US-lessac-medium"
    [friendly-casual]="en_US-hfc_female-medium"
    [neutral-news]="en_GB-alan-medium"
    [energetic-kid]="en_US-kathleen-low"
    [calm-storyteller]="en_GB-jenny_dioco-medium"
)

# voice_id -> hf subdir under rhasspy/piper-voices (lang/lang_region/speaker/quality).
voice_subdir() {
    local vid="$1"
    local lang_region="${vid%%-*}"
    local rest="${vid#*-}"
    local speaker="${rest%-*}"
    local quality="${rest##*-}"
    local lang="${lang_region%%_*}"
    printf '%s/%s/%s/%s' "$lang" "$lang_region" "$speaker" "$quality"
}

download_voice() {
    local voice_id="$1"
    local subdir
    subdir=$(voice_subdir "$voice_id")
    local onnx="${TTS_VOICES_DIR}/${voice_id}.onnx"
    local cfg="${TTS_VOICES_DIR}/${voice_id}.onnx.json"

    if [ -f "$onnx" ] && [ -f "$cfg" ]; then
        log "exists: ${voice_id} -> skip"
        return 0
    fi

    log "downloading ${voice_id} from rhasspy/piper-voices/${subdir}"
    HF_REPO="rhasspy/piper-voices" \
    HF_SUBDIR="$subdir" \
    HF_VOICE="$voice_id" \
    HF_DEST_ONNX="$onnx" \
    HF_DEST_CFG="$cfg" \
        python3 - <<'PY'
import os
import shutil
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

repo = os.environ["HF_REPO"]
subdir = os.environ["HF_SUBDIR"]
voice = os.environ["HF_VOICE"]
dest_onnx = os.environ["HF_DEST_ONNX"]
dest_cfg = os.environ["HF_DEST_CFG"]
token = os.environ.get("HF_TOKEN") or None

pairs = [
    (f"{subdir}/{voice}.onnx", dest_onnx),
    (f"{subdir}/{voice}.onnx.json", dest_cfg),
]

try:
    for remote, dest in pairs:
        cached = hf_hub_download(
            repo_id=repo,
            filename=remote,
            token=token,
            cache_dir=os.path.dirname(dest) or ".",
        )
        shutil.copyfile(cached, dest)
except RepositoryNotFoundError as exc:
    print(f"FATAL: repo not found: {repo} ({exc})", file=sys.stderr)
    sys.exit(2)
except EntryNotFoundError as exc:
    print(f"FATAL: file not found in repo: {exc}", file=sys.stderr)
    sys.exit(3)
except HfHubHTTPError as exc:
    print(f"FATAL: HTTP error downloading {voice}: {exc}", file=sys.stderr)
    sys.exit(4)
except OSError as exc:
    print(f"FATAL: could not place voice files: {exc}", file=sys.stderr)
    sys.exit(5)
PY

    local rc=$?
    if [ "$rc" -ne 0 ] || [ ! -f "$onnx" ] || [ ! -f "$cfg" ]; then
        log "WARN: voice ${voice_id} failed to download (rc=${rc}); skipping"
        rm -f "$onnx" "$cfg"
        return 1
    fi
    log "done: ${voice_id}"
    return 0
}

IFS=',' read -ra requested <<< "$TTS_VOICES_ENABLED"
overall=0
for raw in "${requested[@]}"; do
    pid=$(echo "$raw" | xargs)  # trim whitespace
    [ -z "$pid" ] && continue
    voice_id="${VOICE_MAP[$pid]:-}"
    if [ -z "$voice_id" ]; then
        log "WARN: unknown personality id in TTS_VOICES_ENABLED: ${pid} (skipping)"
        overall=1
        continue
    fi
    if ! download_voice "$voice_id"; then
        overall=1
    fi
done

log "voice prep complete"
exit "$overall"
