# Specification

This document is the source-of-truth for `gemma-llama`. A reimplementation
that follows the contracts and defaults below should be functionally
equivalent to the current code in `app/`, `docker/`, `model_configs/`, and
`eval/`.

For long-form prose see [`docs/README.md`](docs/README.md) (operator guide)
and [`docs/endpoints.md`](docs/endpoints.md) (HTTP surface).

## 1. Objective

Build a portable, production-ready local LLM inference service for the
**Gemma 4** family (April 2026 release: `E2B`, `E4B`, `26B A4B` MoE, `31B`
dense) that:

- Runs as a two-container Docker Compose stack (`llama` + `api`).
- Wraps `ggml-org/llama.cpp`'s OpenAI-compatible `llama-server`.
- Exposes a small, opinionated FastAPI HTTP surface positioned as a
  **personal educational tutor** (homework checker, classroom ear,
  translator, lab assistant) on top of generic text/classify/extract/
  summarize endpoints.
- Supports **text, image, audio, and video** inputs through Gemma 4's
  unified vision/audio projector (default `mmproj-F16.gguf`, with
  `mmproj-BF16.gguf` / `mmproj-F32.gguf` also available) shipped inside
  Unsloth's GGUF repos. Audio + video are gated to E2B/E4B (the only
  variants Google released audio adapters for).
- Includes opt-in **Piper TTS** so any endpoint can also return audio.
- Targets:
  - Raspberry Pi 5 16 GB (`aarch64`, CPU-only) -- production, 24/7.
  - Apple Silicon -- development, no Docker GPU.
  - Linux x86_64 with optional NVIDIA CUDA via a compose override.

A `README.md` (linked above) covers operator setup; this spec covers shape.

## 2. Target Hardware & Operations

### Production host (baseline)

| Spec | Detail |
|------|--------|
| Board | Raspberry Pi 5 |
| RAM | 16 GB |
| Storage | 512 GB A2 microSD |
| OS | 64-bit Raspberry Pi OS (Debian Trixie port) |
| Architecture | `aarch64` / `arm64` |
| GPU | None usable for inference (`GPU_LAYERS=0`) |

Constraints this imposes:

- All Docker images and Python wheels must work on `linux/arm64`.
- llama.cpp must compile for aarch64 CPU (no CUDA, no Metal).
- Default model is **E4B** (4B effective, Q4_0 ≈ 2.7 GB on disk). E2B is the
  smaller alternative; 26B-A4B and 31B configs ship but require a bigger
  host (`>= 32 GB` and `>= 48 GB` RAM respectively).
- `THREADS=4` matches the Pi 5's 4 physical cores. `TTS_MAX_CONCURRENCY=2`
  leaves headroom for llama.cpp.
- Model + projector download happens **once** to the `./models` volume on
  first boot; subsequent boots skip the download.

### Unattended operation

- All services use `restart: unless-stopped`.
- The `llama` container's healthcheck has a `start_period: 3600s` so the
  first-run model download cannot trip a restart loop.
- The `api` container `depends_on: llama (service_healthy)` and only
  starts once `GET http://llama:8080/health` returns 200.

### Dev workflow

```
MacBook (dev)                       Raspberry Pi 5 (prod)
docker compose up --build  ───►     docker compose up -d --build
```

Iterate on the Mac, transfer the repo to the Pi (`git`/`rsync`), bring up
the same Compose file, and let it run. The default `.env` works on both
hosts; only `THREADS` is typically bumped on dev boxes.

## 3. System Architecture

```
Client
 │
FastAPI (host :8010, container :8000)         ── api container
 │   middleware: timing + structured logging
 │   routers: /generate /chat /classify /extract /summarize
 │            /vision/* /audio/* /video/* /tts/*
 │
LlamaAdapter  (httpx → /v1/chat/completions, NDJSON streaming, retries)
 │
Piper subprocess pool (TTS, opt-in via tts=true)
 │
llama-server (container :8080)                 ── llama container
 │   --jinja --mmproj <unified vision+audio projector>
 │
GGUF model + mmproj (./models volume)
```

| Container | Base image | Role |
|-----------|-----------|------|
| `llama` | `ubuntu:22.04` (multi-stage) | Builds `llama-server` from a pinned `LLAMA_CPP_REF`. On boot, downloads `model.gguf` (+ optional `mmproj.gguf`) from Hugging Face via `download_model.sh`, then runs `llama-server` with `--mmproj` + `--no-mmproj-offload` (CPU) or projector-on-GPU when `MMPROJ_USE_GPU=true`. |
| `api` | `python:3.12-slim` | Installs `ffmpeg`, `espeak-ng`, the prebuilt `piper` binary (arm64 / x86_64 picked via `${TARGETARCH}`), and Python deps. On boot, `download_voices.sh` populates the `./voices` volume; then `uvicorn app.main:app` serves the API. |

## 4. Design Principles

1. **Model-agnostic.** Any GGUF that llama.cpp can load works; per-model
   knobs live in `model_configs/*.yaml`. Active config is selected via
   `MODEL_CONFIG_PATH`.
2. **CPU-first.** No GPU required. CUDA is opt-in via
   `docker-compose.gpu.yml` (sets `GGML_CUDA=ON`, `GPU_LAYERS=99`,
   `MMPROJ_USE_GPU=true`).
3. **Stateless API layer.** No model state in FastAPI; llama.cpp owns
   inference. FastAPI handles routing, prompt rendering, multipart media
   prep, response shaping, and the optional TTS attach.
4. **Externalised models.** Models are never baked into images. They live
   under `./models:/models` and are auto-downloaded on first run with a
   sidecar `<file>.source` to detect repo/file changes (`MODEL_REPO`,
   `MMPROJ_REPO`) and force re-download.
5. **Per-model YAML config.** `display_name`, `short_name`, HF source,
   context size, threads, GPU layers, modality flags, generation defaults,
   and Jinja-rendered prompt templates all live in YAML.
6. **Modality order.** Per Gemma 4 docs, image and audio parts MUST come
   **before** the text part in the user message. Enforced server-side in
   `app/media.py::build_user_content` (images → audio → text).
7. **Graceful degradation for TTS.** If `TTS_ENABLED=false`, the Piper
   binary is missing, or no voices are downloaded, every endpoint still
   answers; `tts: true` requests get an `audio_error` field instead of
   audio so clients never need to branch on engine state.

## 5. Supported Models

Default: **`gemma4-e4b.yaml`** (Gemma 4 E4B it, Q4_0 on Pi 5; switch to Q4_K_M on CUDA / Apple-silicon).

| Config file | Display | Modalities | RAM | Notes |
|-------------|---------|------------|-----|-------|
| `gemma4-e4b.yaml` | Gemma 4 E4B (it) | text, image, audio, video | ~6 GB | **Default.** Pi-5-friendly. |
| `gemma4-e2b.yaml` | Gemma 4 E2B (it) | text, image, audio, video | ~4 GB | Lighter Pi sibling. |
| `gemma4-26b-a4b.yaml` | Gemma 4 26B A4B (it) | text, image | ~16 GB | MoE; no audio/video. |
| `gemma4-31b.yaml` | Gemma 4 31B (it) | text, image | ~20 GB | Dense; GPU recommended. |

Each YAML includes:

```yaml
display_name, short_name
hf_repo, hf_file              # main GGUF
mmproj_repo, mmproj_file      # unified vision+audio projector (optional)
context_size, threads, gpu_layers
modalities: { text, image, audio, video }
defaults:   { max_tokens, temperature, thinking }
prompts:    { classify, extract, summarize,
              vision_explain, audio_listen, audio_translate,
              audio_transcribe, video_lab }   # Jinja2 templates
```

Sensible defaults for every prompt template exist in
`app/model_config.py` so a partial YAML still loads.

## 6. API

All routes live at the root, served on host port `${API_PORT:-8010}`
(container port `8000`). Every successful response includes
`inference_time_ms` and an `X-Process-Time-Ms` header on the HTTP response.

### 6.1 Core

- `GET /` -- HTML playground (`app/templates/playground.html`).
- `GET /health` -- `{"status","model","llama_server_reachable","tts_ready"}`;
  HTTP 503 when llama is unreachable.

### 6.2 Text

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/generate` | `prompt`, optional `system`, `images`, `max_tokens`, `temperature`, `thinking`, `stream`, `tts`, `voice` | Single-prompt generation. Streams NDJSON when `stream=true`. |
| `POST` | `/chat`     | `messages[{role,content,images?}]`, same opts as above | Multi-turn chat with per-message images. |
| `POST` | `/classify` | `text`, `labels[]`, `multi_label` | `response_format={"type":"json_object"}`; returns `{labels, raw}`. |
| `POST` | `/extract`  | `text`, `schema` | JSON-schema-style extraction. Tolerates malformed JSON via `_raw`/`_error` (or `data._error`). |
| `POST` | `/summarize`| `text`, `style?`, `max_sentences` | Abstractive summarization. |

### 6.3 Multimodal tutor (multipart/form-data)

| Method | Path | Form fields | Tutor role |
|--------|------|-------------|------------|
| `POST` | `/vision/explain-work` | `image`, `question?`, `subject?`, `max_tokens?`, `temperature?`, `tts?`, `voice?` | The Homework Checker. |
| `POST` | `/audio/listen` | `audio`, `max_tokens?`, `temperature?`, `stream?`, `tts?`, `voice?` | The Classroom Ear (NDJSON streaming supported). |
| `POST` | `/audio/translate` | `audio`, `target_language`, `source_language?`, `max_tokens?`, `temperature?`, `tts?`, `voice?` | Audio in language A → text in B. Defaults `temperature=0.2`, `thinking=false`. |
| `POST` | `/audio/transcribe`| `audio`, `max_tokens?`, `temperature?`, `tts?`, `voice?` | Verbatim STT. Defaults `temperature=0.0`, `thinking=false`. |
| `POST` | `/video/analyze-process` | `video`, `task?`, `n_frames?`, `include_audio?`, `max_tokens?`, `temperature?`, `stream?`, `tts?`, `voice?` | The Lab Assistant. Returns JSON-shaped `{summary, observations, safety_notes, next_step}`. |

Modality gating: a `409 Conflict` is returned when an audio/video route is
hit on a model whose `modalities.audio` / `modalities.video` is `false`
(26B-A4B, 31B).

### 6.4 TTS subsystem

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/tts/voices` | List curated personalities (id, description, language, voice_id, downloaded, sample_rate, is_default). |
| `GET`  | `/tts/output/{audio_id}.wav` | Serve a previously-rendered WAV from the TTL cache. 404 once expired. |
| `POST` | `/tts/speak` | One-shot Piper synth from `{text, voice}`. |

Curated personalities (in `app/tts/voices.py`): `warm-academic` (default),
`friendly-casual`, `neutral-news`, `energetic-kid`, `calm-storyteller`.
Each maps to a Piper ONNX voice from `rhasspy/piper-voices`.

### 6.5 Streaming format (NDJSON)

`stream=true` returns `application/x-ndjson`; one JSON object per line:

```json
{"type": "thinking",  "content": "..."}
{"type": "text",      "content": "..."}
{"type": "meta",      "frames_used": 6, "audio_used": true, "duration_s": 11.4}
{"type": "heartbeat", "elapsed_ms": 5000}
{"type": "audio",     "seq": 1, "sentence": "...", "content": "<base64-wav>", "mime": "audio/wav", "sample_rate": 22050, "voice": "warm-academic"}
{"type": "audio_error","content": "..."}
{"type": "error",     "content": "..."}
```

- `thinking` lines are emitted when the model exposes `reasoning_content`
  / `reasoning` in deltas, or via inline `<think>...</think>` spans.
- `heartbeat` lines (every 5 s) are emitted by `/audio/listen` and
  `/video/analyze-process` so the HTTP connection (and the user's hope)
  survives multi-minute multimodal preprocessing on Pi 5.
- `audio` lines are only emitted when `tts=true` -- one per completed
  sentence, interleaved with `text` chunks.
- `meta` is emitted as the first line of `/video/analyze-process` streams.

### 6.6 Thinking toggle

`thinking: true` (default) preserves chain-of-thought in the response
(separately surfaced as `thinking_content` for non-stream, or `type:
"thinking"` for stream). `thinking: false` appends `/no_think` to the
last user message and trims latency.

### 6.7 TTS attach shapes (response side)

Three response shapes, picked per endpoint based on expected audio length:

- **Inline base64** (`/classify`, `/audio/translate`, `/tts/speak`):
  `audio_base64`, `audio_mime`, `audio_duration_ms`, `voice` are inlined
  in the JSON response.
- **File URL** (`/generate` non-stream, `/chat` non-stream, `/summarize`,
  `/extract`, `/vision/explain-work`, `/audio/listen` non-stream,
  `/audio/transcribe`, `/video/analyze-process` non-stream): response
  carries `audio_url` -> `GET /tts/output/{id}.wav`. Files evicted after
  `TTS_OUTPUT_TTL_SECONDS`.
- **Streaming** (`/generate`, `/chat`, `/audio/listen`, `/video/analyze-process`
  with `stream=true`): per-sentence `{"type":"audio", ...}` NDJSON lines.

On any TTS failure the response carries `audio_error: <reason>` instead.

## 7. llama.cpp Integration

### Build

`docker/Dockerfile.llama` is multi-stage on `ubuntu:22.04`:

- **Builder**: installs `build-essential`, `cmake`, `git`,
  `libcurl4-openssl-dev`; clones `ggml-org/llama.cpp` at
  `${LLAMA_CPP_REF}` (default `b8837` -- a tag that ships the April 2026
  native Gemma 4 audio merge); builds the `llama-server` target with
  `cmake -DGGML_CUDA=${GGML_CUDA} -DLLAMA_CURL=ON
  -DBUILD_SHARED_LIBS=ON`.
- **Runtime**: minimal Ubuntu with `curl`, `libgomp1`, `python3-pip`,
  and `huggingface_hub[cli]` for the download script. Copies
  `llama-server` and the built `.so` files. Healthcheck:
  `curl -fsS http://localhost:${PORT}/health` (15s interval, 3600s start
  period). CMD runs `download_model.sh` then `exec llama-server …`.

### llama-server flags

```
llama-server -m $MODEL_PATH \
  [--mmproj $MMPROJ_PATH [--no-mmproj-offload]] \
  --host $HOST --port $PORT \
  -c $CONTEXT_SIZE -t $THREADS -ngl $GPU_LAYERS \
  --jinja
```

- `--jinja` is required so the Gemma chat template handles system + tools.
- `--mmproj` is added when `MMPROJ_PATH` exists. The same unified
  Gemma 4 mmproj GGUF carries **both** vision and audio adapters, so a
  single flag enables `/vision/*`, `/audio/*`, and `/video/*`. Default
  precision is F16 (fastest on Pi 5 / consumer CPUs); BF16 and F32
  builds are available in the same repo for hosts with native BF16.
- `--no-mmproj-offload` is appended unless `MMPROJ_USE_GPU=true`
  (CPU-resident projector is the right default on Pi 5 / CPU-only Macs).

### `download_model.sh`

Idempotent. For each of `MODEL_REPO/MODEL_FILE` and (optionally)
`MMPROJ_REPO/MMPROJ_FILE`:

1. Skip if the destination file exists, is `>= 1 MiB`, and the sidecar
   `<dest>.source` records the same `repo|file` pair.
2. Otherwise, call Python `huggingface_hub.hf_hub_download(repo, file,
   token=$HF_TOKEN, cache_dir=$MODEL_DIR)` and copy to the deterministic
   `MODEL_PATH` / `MMPROJ_PATH` (`/models/model.gguf`,
   `/models/mmproj.gguf`).
3. Hard-fail if the file is missing or below the size floor; the script
   exits non-zero so the container never boots with stale weights.
4. Best-effort: scan the first 4 MiB of the mmproj for `mm.vision` /
   `mm.audio` substrings and log which modalities the projector
   appears to carry.

If `MMPROJ_REPO`/`MMPROJ_FILE` are blank, vision/audio support is
disabled cleanly (the api container returns 409 on those routes).

### llama.cpp HTTP surface used

- `GET /health`
- `POST /v1/chat/completions` (with `stream` true/false). All endpoints
  fan in here; `response_format={"type":"json_object"}` is set for
  `/classify` and `/extract`.

## 8. FastAPI ↔ llama.cpp adapter

### `LlamaAdapter` (`app/llama_adapter.py`)

```python
class LlamaAdapter:
    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3): ...
    async def start(self): ...              # builds httpx.AsyncClient
    async def stop(self): ...
    async def health_check(self) -> bool: ...
    async def chat_completion(
        self, messages, *, max_tokens, temperature,
        response_format=None, thinking=True, extra=None,
    ) -> dict: ...
    async def chat_completion_stream(
        self, messages, *, max_tokens, temperature,
        thinking=True, extra=None,
    ) -> AsyncIterator[str]: ...
```

Behaviour:

- `httpx.AsyncClient` with `timeout=LLM_TIMEOUT_SECONDS` (default 1800s in
  `Settings`; `connect=30s`). Streaming uses `read=None` so multi-minute
  encodes do not collapse mid-stream.
- Retries (`LLM_RETRIES`, default 3) with exponential backoff `2^(n-1)`
  capped at 10 s. **Timeouts are NOT retried** (a retry would just cancel
  the in-flight llama.cpp task and restart the encode from zero); they
  raise `LlamaServerError` immediately. 5xx is retried, 4xx is not.
- `chat_completion` injects `_inference_time_ms` (round-trip ms) into the
  parsed response.
- `chat_completion_stream` yields NDJSON lines `{"type":"thinking" |
  "text", "content":"..."}`. Inline `<think>...</think>` blocks are split
  out into `thinking` lines.
- `_apply_no_think(messages, thinking=False)` appends `/no_think` to the
  last user message (string or list-content) so the model suppresses
  reasoning.

Helpers in the same module: `extract_content`, `extract_reasoning`,
`extract_usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`,
`tokens_predicted`).

### Multipart media prep (`app/media_multipart.py`)

Tuned for Pi 5; constants are part of the contract:

| Constant | Value | Meaning |
|----------|-------|---------|
| `_MAX_IMAGE_EDGE` | 896 | Gemma 4's vision tile; longest edge cap. |
| `_JPEG_QUALITY` | 88 | Re-encode quality. |
| `_AUDIO_SAMPLE_RATE` | 16000 | mtmd Gemma 4 audio path expects 16 kHz mono WAV. |
| `_AUDIO_CHANNELS` | 1 | Mono. |
| `MAX_IMAGE_BYTES` | 10 MiB | Per-upload cap. |
| `MAX_AUDIO_BYTES` | 25 MiB | Per-upload cap. |
| `MAX_VIDEO_BYTES` | 75 MiB | Per-upload cap. |
| `DEFAULT_VIDEO_FRAMES` | 3 | Frames sampled per `/video/analyze-process` call. |
| `MAX_VIDEO_FRAMES` | 6 | Hard cap (each frame ~ 75 s through the projector on Pi 5 CPU). |

Helpers: `read_image_upload`, `read_audio_upload`,
`extract_video_payload(file, n_frames, include_audio)` -- all return
`data:<mime>;base64,...` URIs (so the rest of the codebase can stay on
OpenAI-style `image_url` / `input_audio` content parts). One `ffmpeg` /
`ffprobe` pass per asset.

### Image / audio content part shapes (`app/media.py`)

- `ImageInput` accepts `{url}`, `{base64, mime_type}`, or `{url:
  data:image/...}`. Disallows mixing `url` + `base64`; validates base64.
- `build_user_content(text, images=None, *, image_urls=None,
  audio_urls=None)` -- always orders `images → audio → text`.
- `audio_part_from_data_url` produces `{"type":"input_audio",
  "input_audio":{"data": "<b64>", "format": "wav"}}`.

## 9. TTS subsystem (`app/tts/`)

- **`engine.py`** -- `PiperEngine`. Lifecycle: `start()` validates the
  binary + voices dir and warms a per-voice metadata cache (sample rate
  read from `<voice_id>.onnx.json`); `synth(text, voice)` invokes
  `piper --model … --config … --output_file - --quiet` with the input on
  stdin and reads a complete WAV from stdout. Concurrency is bounded by
  `asyncio.Semaphore(TTS_MAX_CONCURRENCY)`. Failures are surfaced as
  `PiperError` and translated to `audio_error`.
- **`voices.py`** -- `PERSONALITIES` registry mapping personality id ->
  `PiperVoice(voice_id, description)`. Provides `resolve_personality`,
  `voice_paths`, `read_sample_rate`. `DEFAULT_PERSONALITY =
  "warm-academic"`.
- **`sentences.py`** -- `SentenceBuffer` for the streaming wrapper. Hard
  flush at `TTS_MAX_SENTENCE_CHARS` so unpunctuated output still ships
  audio in bounded time.
- **`storage.py`** -- `TTSStorage` writes WAVs to `TTS_OUTPUT_DIR`,
  exposes a TTL janitor (`TTS_OUTPUT_TTL_SECONDS`) started during
  `lifespan`, and serves them via `GET /tts/output/{id}.wav`.
- **`integration.py`** + **`_router_helpers.py`** -- three glue helpers
  (`maybe_attach_inline`, `maybe_attach_file`, `maybe_wrap_stream`) used
  by every router so the TTS attach is uniform and never raises.
- **`router.py`** -- the `/tts/*` HTTP surface.
- **`docker/download_voices.sh`** -- idempotent voice fetch from
  `rhasspy/piper-voices` on Hugging Face into `TTS_VOICES_DIR`. Failure
  to download a single voice is non-fatal.

Piper itself is the prebuilt static binary from
`rhasspy/piper@2023.11.14-2`, picked per `${TARGETARCH}`
(aarch64 / x86_64 / armv7l) in `Dockerfile.api`.

## 10. Docker

### `docker-compose.yml`

Two services:

- **`llama`** -- builds `Dockerfile.llama` (`GGML_CUDA=OFF`,
  `LLAMA_CPP_REF=${LLAMA_CPP_REF:-b8837}`); volume mount
  `./models:/models`; env carries `MODEL_REPO/FILE`, `MMPROJ_REPO/FILE`,
  `MMPROJ_USE_GPU`, `HF_TOKEN`, `CONTEXT_SIZE`, `THREADS`, `GPU_LAYERS`,
  `MODEL_DIR`, `MODEL_PATH`, `MMPROJ_PATH`. Healthcheck on `:8080/health`.
  `restart: unless-stopped`. Exposes `8080` only inside the Compose
  network.
- **`api`** -- builds `Dockerfile.api`; ports `${API_PORT:-8010}:8000`;
  volumes `./voices:/voices` and `./tts_outputs:/tmp/tts_outputs`; env
  carries `LLAMA_SERVER_URL`, `MODEL_CONFIG_PATH`, `LLM_TIMEOUT_SECONDS`,
  `LLM_RETRIES`, `LOG_LEVEL`, all `TTS_*`, and `HF_TOKEN`.
  `depends_on: llama (service_healthy)`. `restart: unless-stopped`.

### `docker-compose.gpu.yml` (override)

Sets `GGML_CUDA=ON` build arg, `GPU_LAYERS=99`, `MMPROJ_USE_GPU=true`,
and reserves `nvidia` GPU devices. Use:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Python deps (`requirements.txt`)

```
fastapi>=0.115
uvicorn[standard]>=0.34
httpx>=0.28
pydantic>=2.10
pydantic-settings>=2.7
pyyaml>=6.0
jinja2>=3.1
python-multipart>=0.0.20
Pillow>=10.4
huggingface_hub>=0.26
```

## 11. Configuration

All settings come from environment variables (typically via `.env`; see
`.env.example` for the canonical list). The `Settings` class
(`pydantic-settings`) reads them case-insensitively.

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_PORT` | `8010` | Host port for FastAPI. |
| `LOG_LEVEL` | `info` | Python logger level. |
| `MODEL_CONFIG_PATH` | `model_configs/gemma4-e4b.yaml` | Active YAML config. |
| `MODEL_REPO` / `MODEL_FILE` | `unsloth/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_0.gguf` | HF source for the GGUF. Q4_0 default targets the Pi 5's repacked dotprod SGEMM path; switch to a K-quant on hosts with native i8mm or GPU. |
| `MMPROJ_REPO` / `MMPROJ_FILE` | same repo / `mmproj-F16.gguf` | Unified vision+audio projector (F16 default; BF16 / F32 also in the same repo). Blank for text-only. |
| `HF_TOKEN` | _unset_ | Required only for gated repos. |
| `CONTEXT_SIZE` | `8192` | `llama-server -c`. |
| `THREADS` | `4` | Pi 5 default; bump to physical core count on dev. |
| `GPU_LAYERS` | `0` | `llama-server -ngl`; `99` on the GPU override. |
| `MMPROJ_USE_GPU` | `false` | `--no-mmproj-offload` unless `true`. |
| `LLAMA_CPP_REF` | `b8837` | llama.cpp git tag built into the image. |
| `LLAMA_SERVER_URL` | `http://llama:8080` | Adapter base URL. |
| `LLM_TIMEOUT_SECONDS` | `1800` | HTTP timeout for llama.cpp requests. |
| `LLM_RETRIES` | `3` | Retry count (timeouts are NOT retried). |
| `TTS_ENABLED` | `true` | Master switch for the Piper subsystem. |
| `PIPER_BINARY` | `/usr/local/bin/piper` | Binary path inside the api container. |
| `TTS_VOICES_DIR` | `/voices` | Where downloaded `.onnx` voices live. |
| `TTS_VOICES_ENABLED` | _all 5 personalities_ | Comma-separated ids to download at boot. |
| `TTS_DEFAULT_VOICE` | `warm-academic` | Voice when callers omit `voice`. |
| `TTS_OUTPUT_DIR` / `TTS_OUTPUT_TTL_SECONDS` | `/tmp/tts_outputs` / `600` | TTL cache for `audio_url`. |
| `TTS_MAX_SENTENCE_CHARS` | `400` | Streaming sentence-buffer hard flush. |
| `TTS_MAX_CONCURRENCY` | `2` | Max concurrent Piper subprocesses. |

## 12. Observability

- Per-request structured log (`%(asctime)s %(levelname)s %(name)s
  %(message)s`) with method, path, status, elapsed ms.
- `X-Process-Time-Ms` header added by the `timing_and_logging`
  middleware.
- Every endpoint response includes `inference_time_ms` (the llama.cpp
  round-trip).
- `/generate` and `/chat` additionally surface `tokens_predicted` and
  the full `usage` dict (`prompt_tokens`, `completion_tokens`,
  `total_tokens`, `tokens_predicted`).

## 13. Failure handling

- llama unreachable → `GET /health` returns 503 with
  `llama_server_reachable=false`; other endpoints raise 502
  (`LlamaServerError` mapped to `HTTPException(502)` in routers).
- `LLM_TIMEOUT_SECONDS` exceeded → 502 with a message hinting at the
  cause; not retried (would restart the encode from zero).
- 4xx from llama.cpp → not retried; 5xx → retried with exponential
  backoff up to `LLM_RETRIES`.
- `/extract` JSON parse failures → response includes `_raw` + `_error`
  keys instead of failing the request.
- Modality mismatch (audio/video on a vision-only model) → 409.
- Multipart caps exceeded → 413; bad media → 400 with the ffmpeg /
  Pillow error.
- Unknown TTS voice → 422; engine unavailable → endpoint still answers,
  with `audio_error` instead of audio.
- Model download failure → `download_model.sh` exits non-zero, the llama
  container fails to start (and `api` waits on the healthcheck).

## 14. Evaluation harness (`eval/`)

Runs against a live API (`docker compose up` first).

```bash
pip install -r requirements-eval.txt

# CLI runner -- writes timestamped JSON into eval/reports/
python -m eval.runner --all
python -m eval.runner --task classify   # | extract | summarize | audio | video

# Pytest -- skipped automatically if the API is unreachable
pytest eval/ -v -s
```

`requirements-eval.txt`:

```
-r requirements.txt
pytest>=8.0
pytest-asyncio>=0.24
rouge-score>=0.1.2
```

### Tasks, datasets, thresholds

| Task | Dataset | Metrics | Pytest threshold |
|------|---------|---------|------------------|
| `classify` | `classification.json` | accuracy, micro P/R, macro F1 | `accuracy >= 0.5` |
| `extract` | `extraction.json` | JSON validity rate, per-field accuracy | `json_valid_rate >= 0.8` |
| `summarize` | `summarization.json` | ROUGE-1 / ROUGE-2 / ROUGE-L | `ROUGE-1 >= 0.2` |
| `audio` | `audio.json` (+ `eval/datasets/media/`) | non_empty_rate, keyword_pass_rate (≥ 0.5 overlap) | -- |
| `video` | `video.json` (+ media) | non_empty_rate, json_valid_rate, expected_keys_rate | -- |

All tasks also report `latency_ms` (`p50/p95/p99/mean`) computed from the
API's `inference_time_ms`. Reports land in `eval/reports/` as
`report-<task>-<UTC ts>.json`.

### Dataset shapes

- `classification.json`: `{"text", "labels":[...], "expected":[...],
  "multi_label": false}`
- `extraction.json`: `{"text", "schema":{...}, "expected":{...}}`
- `summarization.json`: `{"text", "reference", "style?",
  "max_sentences?"}`
- `audio.json`: `{"file":"media/…", "mode":"listen|transcribe|translate",
  "expected_keywords?":[...], "target_language?", "source_language?"}`
- `video.json`: `{"file":"media/…", "task?", "n_frames?",
  "include_audio?", "expected_keys?":[...]}`

## 15. Performance Tuning

- **Threads**: `THREADS` to physical core count (4 on Pi 5, 8+ on dev).
- **Context size**: lower `CONTEXT_SIZE` reduces RAM; raise to `16384`
  for long audio (audio tokens are denser than text).
- **Quantization**: `Q4_0` is the Pi 5 default (hits the repacked dotprod SGEMM kernel); switch to `Q4_K_M` on CUDA / Apple-silicon. The per-config YAML
  picks the file. Bigger quants improve quality at the cost of RAM.
- **GPU offload**: `docker-compose.gpu.yml` flips `GGML_CUDA=ON`,
  `GPU_LAYERS=99`, `MMPROJ_USE_GPU=true`.
- **TTS**: trim `TTS_VOICES_ENABLED` to shorten cold start; raise
  `TTS_MAX_CONCURRENCY` only if the host has spare cores.
- **Video**: stay at `n_frames <= MAX_VIDEO_FRAMES` (6) on Pi 5; each
  added frame costs ~250-400 ms of vision encode (multiple seconds on
  CPU-only Pi 5).

## 16. Future Work

Already shipped (no longer "future"):

- Native audio support for E2B/E4B (`/audio/listen`, `/audio/translate`,
  `/audio/transcribe`).
- Video tutor (`/video/analyze-process`, ffmpeg-sampled frames + audio).
- Vision tutor (`/vision/explain-work`).
- NDJSON streaming on text + audio + video routes, with `meta` and
  `heartbeat` framing.
- Opt-in Piper TTS on every endpoint (inline-base64, file-URL, and
  per-sentence streaming attach shapes), curated voice registry, and
  `POST /tts/speak`.

Designed-for but not yet implemented:

- Embeddings endpoint.
- GPU auto-detection.
- Model hot-swap without restart.
