#!/usr/bin/env bash
# ============================================================================
# Idempotent model download for the llama container.
#   - Reads MODEL_REPO/MODEL_FILE -> $MODEL_PATH
#   - Reads MMPROJ_REPO/MMPROJ_FILE -> $MMPROJ_PATH (optional)
#   - Skips download when the target file already exists on the volume.
#   - Hard-fails if the file is still missing after a "successful" download.
#
# Implementation note: we avoid the deprecated `huggingface-cli download` /
# new `hf download` CLI surface and call the Python `hf_hub_download` API
# directly. That API has been stable across every huggingface_hub release we
# care about and never silently no-ops the way the deprecated CLI does.
# ============================================================================
set -euo pipefail

: "${MODEL_DIR:=/models}"
: "${MODEL_PATH:=/models/model.gguf}"
: "${MMPROJ_PATH:=/models/mmproj.gguf}"
: "${DRAFT_PATH:=/models/draft.gguf}"
: "${MIN_BYTES:=1048576}"   # 1 MiB sanity floor for a usable GGUF

mkdir -p "$MODEL_DIR"

log() { printf '[download_model] %s\n' "$*" >&2; }

# Usage: download <repo> <file> <dest>
#
# Writes a sidecar `<dest>.source` recording "<repo>|<file>" so we can detect
# when MODEL_REPO/MODEL_FILE (or MMPROJ_*) was changed in .env and force a
# re-download. Without this, the bare "exists -> skip" check happily serves
# stale weights from a previous configuration -- which silently mismatches a
# freshly-downloaded mmproj from a different model family.
download() {
    local repo="$1" file="$2" dest="$3"
    local source_tag="${repo}|${file}"
    local source_file="${dest}.source"

    if [ -f "$dest" ]; then
        local size
        size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
        local recorded_source=""
        if [ -f "$source_file" ]; then
            recorded_source=$(cat "$source_file" 2>/dev/null || echo "")
        fi

        if [ "$size" -ge "$MIN_BYTES" ] && [ "$recorded_source" = "$source_tag" ]; then
            log "exists: $dest ($(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size bytes")) from $source_tag -> skip"
            return 0
        fi

        if [ "$size" -lt "$MIN_BYTES" ]; then
            log "stale/empty $dest (${size} bytes) -> redownloading"
        elif [ -z "$recorded_source" ]; then
            log "no source sidecar for $dest -> assuming stale, redownloading as $source_tag"
        else
            log "source changed for $dest ($recorded_source -> $source_tag) -> redownloading"
        fi
        rm -f "$dest" "$source_file"
    fi

    log "downloading $repo / $file -> $dest"
    HF_REPO="$repo" HF_FILE="$file" HF_DEST="$dest" python3 - <<'PY'
import os
import shutil
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError

repo = os.environ["HF_REPO"]
filename = os.environ["HF_FILE"]
dest = os.environ["HF_DEST"]
token = os.environ.get("HF_TOKEN") or None

try:
    cached = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=token,
        # Keep the cache on the same volume so the move stays cheap.
        cache_dir=os.path.dirname(dest) or ".",
    )
except RepositoryNotFoundError as exc:
    print(f"FATAL: repo not found: {repo} ({exc})", file=sys.stderr)
    sys.exit(2)
except EntryNotFoundError as exc:
    print(f"FATAL: file not found in repo: {repo}/{filename} ({exc})", file=sys.stderr)
    sys.exit(3)
except HfHubHTTPError as exc:
    print(f"FATAL: HTTP error downloading {repo}/{filename}: {exc}", file=sys.stderr)
    sys.exit(4)

# Move (or copy if cross-fs) the resolved file to the deterministic dest path.
try:
    shutil.copyfile(cached, dest)
except OSError as exc:
    print(f"FATAL: could not place {cached} at {dest}: {exc}", file=sys.stderr)
    sys.exit(5)
PY

    if [ ! -f "$dest" ]; then
        log "FATAL: download reported success but $dest does not exist"
        exit 1
    fi
    local size
    size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
    if [ "$size" -lt "$MIN_BYTES" ]; then
        log "FATAL: $dest is only ${size} bytes (< MIN_BYTES=${MIN_BYTES})"
        exit 1
    fi
    printf '%s' "$source_tag" > "$source_file"
    log "done: $dest ($(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size bytes")) from $source_tag"
}

if [ -z "${MODEL_REPO:-}" ] || [ -z "${MODEL_FILE:-}" ]; then
    log "FATAL: MODEL_REPO and MODEL_FILE must be set"
    exit 1
fi

download "$MODEL_REPO" "$MODEL_FILE" "$MODEL_PATH"

if [ -n "${DRAFT_REPO:-}" ] && [ -n "${DRAFT_FILE:-}" ]; then
    # Draft model for speculative decoding. Optional: if the configured file
    # is missing on the hub, warn and keep going without spec-decode (the
    # llama-server CMD silently drops `--model-draft` when the file is absent).
    if ! ( download "$DRAFT_REPO" "$DRAFT_FILE" "$DRAFT_PATH" ); then
        log "WARN: draft model download failed for $DRAFT_REPO/$DRAFT_FILE; continuing without speculative decoding"
        rm -f "$DRAFT_PATH" "${DRAFT_PATH}.source"
    fi
else
    log "no DRAFT_REPO/DRAFT_FILE configured; skipping speculative-decoding draft model"
    # Clean up any leftover draft from a previous configuration so it does not
    # silently get loaded after the user has disabled spec-decode.
    rm -f "$DRAFT_PATH" "${DRAFT_PATH}.source"
fi

if [ -n "${MMPROJ_REPO:-}" ] && [ -n "${MMPROJ_FILE:-}" ]; then
    # The mmproj projector is optional: if the configured file is missing on
    # the hub (e.g. an upstream rename, or a model whose mmproj has not been
    # published yet), warn and keep going text-only. Run in a subshell so any
    # `exit` inside `download()` only kills the subshell, not this script.
    if ! ( download "$MMPROJ_REPO" "$MMPROJ_FILE" "$MMPROJ_PATH" ); then
        log "WARN: mmproj download failed for $MMPROJ_REPO/$MMPROJ_FILE; continuing without vision/audio support"
        rm -f "$MMPROJ_PATH" "${MMPROJ_PATH}.source"
    fi
else
    log "no MMPROJ_REPO/MMPROJ_FILE configured; skipping vision projector"
    # Clean up any leftover mmproj from a previous configuration so a
    # text-only deployment doesn't load a stale projector.
    rm -f "$MMPROJ_PATH" "${MMPROJ_PATH}.source"
fi

# Best-effort projector inspection: count vision vs audio tensors so a
# misconfigured / vision-only mmproj is loud at boot rather than silently
# returning empty audio responses. We only read the GGUF header so this stays
# fast on a Pi 5. Failure is non-fatal (older builds may not expose the names).
if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_PATH="$MMPROJ_PATH" python3 - <<'PY' || true
import os
import struct
import sys

path = os.environ["MMPROJ_PATH"]

GGUF_MAGIC = b"GGUF"

def read(fh, fmt):
    size = struct.calcsize(fmt)
    buf = fh.read(size)
    if len(buf) != size:
        raise EOFError
    return struct.unpack(fmt, buf)[0]

def read_string(fh):
    n = read(fh, "<Q")
    return fh.read(n).decode("utf-8", errors="replace")

try:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != GGUF_MAGIC:
            print(f"[download_model] mmproj sanity: not a GGUF file ({magic!r})", file=sys.stderr)
            sys.exit(0)
        _version = read(fh, "<I")
        n_tensors = read(fh, "<Q")
        n_kv = read(fh, "<Q")
        # Skip the KV section: each entry is name + type + value(s). We only
        # care about tensor names below, which come after the KV block. To keep
        # this self-contained without re-implementing the full KV parser, we
        # rely on the fact that Unsloth Gemma 4 mmproj GGUFs surface the
        # vision/audio split via tensor names of the form `mm.[vision|audio].*`.
        # If parsing the KV block is too involved here, we just bail.
except Exception as exc:
    print(f"[download_model] mmproj sanity: skipped ({exc})", file=sys.stderr)
    sys.exit(0)

# Lightweight name probe: scan the raw bytes for the marker substrings. This
# is good enough as a smoke check without parsing the full GGUF KV section.
try:
    head = open(path, "rb").read(4 * 1024 * 1024)
    has_vision = b"mm.vision" in head or b"v.blk" in head
    has_audio = b"mm.audio" in head or b"a.blk" in head or b"audio" in head
    flags = []
    if has_vision: flags.append("vision")
    if has_audio:  flags.append("audio")
    if not flags:
        flags = ["unknown"]
    print(f"[download_model] mmproj at {path}: detected modalities = {','.join(flags)}", file=sys.stderr)
except Exception as exc:
    print(f"[download_model] mmproj sanity: probe failed ({exc})", file=sys.stderr)
PY
fi

log "model ready"
