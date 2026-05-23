# gemma-llama

Portable, production-ready local LLM inference for the **Gemma 4** family (April 2026 release: E2B, E4B, 26B A4B MoE, and 31B dense GGUFs from Unsloth), packaged as a two-container Docker Compose stack:

- **`llama`** — `llama.cpp`'s `llama-server` with the GGUF model (and the unified vision + audio projector) auto-downloaded from Hugging Face on first boot.
- **`api`** — FastAPI service exposing five mode-specific tutor endpoints — `/free-convo/turn`, `/voice-mirror/suggest`, `/voice-mirror/score`, `/vision/teach-object`, `/audio/listen` — plus a built-in playground UI.

Designed to run unattended 24/7 on a **Raspberry Pi 5 (16 GB, arm64)** while remaining first-class on **Apple Silicon** and **Linux x86_64** dev boxes.

```text
Client -> FastAPI (:8010) -> httpx -> llama-server (:8080 internal) -> GGUF (./models)
```

---

## Quickstart

### macOS (development)

```bash
cp .env.example .env
# (optional) bump THREADS for your core count
docker compose up --build
open http://localhost:8010/   # playground
```

### Raspberry Pi 5 (production, 24/7)

```bash
git clone <this repo> gemma-llama && cd gemma-llama
cp .env.example .env
docker compose up -d --build
docker compose logs -f llama  # watch first-run model download
```

The `llama` container's health-check has a **3600 s start period** to absorb slow SD-card model downloads. The `api` container only starts once `llama` reports healthy.

### Linux x86_64 with NVIDIA GPU (optional)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Requires the NVIDIA Container Toolkit. Sets `GGML_CUDA=ON` at build time and `GPU_LAYERS=99` at runtime.

---

## API

All endpoints live at the root, served on host port `${API_PORT:-8010}` (container port `8000`).

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Interactive playground (HTML). |
| `GET`  | `/health` | `{"status","model","llama_server_reachable","tts_ready"}`; `503` when `llama` is unreachable. |
| `POST` | `/free-convo/turn` | **FreeConvoMode.** Multipart audio → warm Hinglish reply + `start_learning` intent flag. |
| `POST` | `/voice-mirror/suggest` | **VoiceMirrorMode.** JSON `{level?, history?}` → next target word + spoken cue text. |
| `POST` | `/voice-mirror/score` | **VoiceMirrorMode.** Multipart audio + `target_word` → `praise` / `correct` / `retry` verdict + feedback line. |
| `POST` | `/vision/teach-object` | **VisionMode.** Multipart `image + audio` (camera frame + spoken guess) → teaching line ("Yes, this is milk. Say: I drink milk."). |
| `POST` | `/audio/listen` | **RolePlayMode / fallback.** Multipart audio → text answer. Supports `stream=true` NDJSON. |

Every successful response includes `inference_time_ms` and an `X-Process-Time-Ms` header. Every endpoint accepts the optional `tts` / `voice` fields and returns an `audio_url` to the rendered Piper WAV when `tts=true`.

### Defaults you should know about

- **`thinking` is off by default** on every endpoint. Chain-of-thought roughly doubles the decoded token count on Pi 5 for the same final answer; the mode endpoints all run with `thinking=false` for snappy turn latency. Only `/audio/listen` exposes a `thinking` form field for harder spoken questions where reasoning helps quality.
- **Image / audio downsizing**: uploads to the multimodal routes are normalised in [`app/media_multipart.py`](../app/media_multipart.py) before they reach the projector — images are resized to fit Gemma 4's 896 px tile (longest edge) and re-encoded as JPEG; audio is transcoded to mono 16 kHz WAV.

### Streaming

Only `/audio/listen` streams (`stream=true`). The body is `application/x-ndjson`; each line is one of:

```json
{"type": "text",      "content": "..."}
{"type": "heartbeat", "elapsed_ms": 5000}
{"type": "audio",     "seq": 1, "sentence": "...", "content": "<base64-wav>", "mime": "audio/wav", "sample_rate": 22050, "voice": "warm-academic"}
{"type": "error",     "content": "..."}
```

`heartbeat` lines fire every 5 s so the HTTP connection survives multi-minute audio prefill on Pi 5. `audio` lines are emitted only when `tts=true` — one per completed sentence — so audio plays back while Gemma is still generating the next sentence. See [TTS audio attach](endpoints.md#tts-audio-attach) for the full opt-in surface.

### Example calls

```bash
# FreeConvoMode: post-wake spoken turn -> reply + start_learning flag
curl -s -X POST localhost:8010/free-convo/turn \
  -F "audio=@hello.wav" -F "tts=true" | jq
```

```bash
# VoiceMirrorMode: ask for the next word to practise
curl -s localhost:8010/voice-mirror/suggest \
  -H 'content-type: application/json' \
  -d '{"level":"beginner","history":["apple","water"],"tts":true}' | jq
```

```bash
# VoiceMirrorMode: score the learner's attempt at "river"
curl -s -X POST localhost:8010/voice-mirror/score \
  -F "audio=@attempt.wav" -F "target_word=river" -F "tts=true" | jq
```

```bash
# VisionMode: camera frame + spoken guess -> teaching line
curl -s -X POST localhost:8010/vision/teach-object \
  -F "image=@frame.jpg" -F "audio=@guess.wav" -F "tts=true" | jq
```

```bash
# RolePlayMode: in-character NPC turn (NDJSON streamed)
curl -s -N -X POST localhost:8010/audio/listen \
  -F "audio=@learner-turn.wav" -F "stream=true" -F "tts=true"
```

---

## Configuration

All settings come from environment variables (typically via `.env`). See [`.env.example`](.env.example) for the canonical list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_PORT` | `8010` | Host port for the FastAPI service. |
| `MODEL_CONFIG_PATH` | `model_configs/gemma4-e4b.yaml` | Active model YAML (consumed by the `api` container). |
| `MODEL_REPO` / `MODEL_FILE` | `unsloth/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_0.gguf` | HF source for the GGUF (consumed by `download_model.sh`). Q4_0 is preferred on Pi 5 because llama.cpp's repacked dotprod SGEMM kernel makes it the fastest matmul path on Cortex-A76; switch to a K-quant (e.g. `Q4_K_M`) on CUDA / Apple-silicon hosts. |
| `MMPROJ_REPO` / `MMPROJ_FILE` | `unsloth/gemma-4-E4B-it-GGUF` / `mmproj-F16.gguf` | Official Google vision/audio projector. F16 is the fastest precision on Pi 5 / consumer CPUs (no native BF16); switch to `mmproj-BF16.gguf` on CUDA / Apple-silicon hosts. Set both to empty for a text-only deployment. |
| `HF_TOKEN` | _unset_ | Required only for gated repos. |
| `CONTEXT_SIZE` | `8192` | `llama-server -c`. |
| `THREADS` | `4` (Pi) | `llama-server -t`; bump to your physical core count on dev boxes. |
| `GPU_LAYERS` | `0` | `llama-server -ngl`; flipped to `99` by `docker-compose.gpu.yml`. |
| `FLASH_ATTN` | `true` | Adds `-fa` to `llama-server`. Required for non-`f16` V-cache; setting to `false` silently drops `CACHE_TYPE_V` back to the default. |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` / `q8_0` | KV-cache quantisation. `q8_0` typically gives 20-40% faster decode at 8k ctx on Pi 5 with no measurable quality loss; `f16` reverts to the lossless baseline; `q4_0` is more aggressive. |
| `MLOCK` | `true` | Pins model weights in RAM via `mlock(2)` so a memory spike (ffmpeg, etc.) cannot page them out. Safe on Pi 5 16 GB; set to `false` on memory-tight hosts. |
| `LLAMA_SERVER_URL` | `http://llama:8080` | Adapter base URL. |
| `LLM_TIMEOUT_SECONDS` | `300` | HTTP timeout for inference requests. |
| `LLM_RETRIES` | `3` | Exponential-backoff retries on transient failures. |
| `LOG_LEVEL` | `info` | Python logger level. |
| `TTS_ENABLED` | `true` | Master switch for the Piper TTS subsystem. |
| `PIPER_BINARY` | `/usr/local/bin/piper` | Path to the Piper binary inside the api container. |
| `TTS_VOICES_DIR` | `/voices` | Where downloaded `.onnx` voices live (volume-mounted `./voices:/voices`). |
| `TTS_VOICES_ENABLED` | _all 5_ | Comma-separated personality ids to download at api boot. |
| `TTS_DEFAULT_VOICE` | `warm-academic` | Personality used when callers omit `voice`. |
| `TTS_OUTPUT_DIR` / `TTS_OUTPUT_TTL_SECONDS` | `/tmp/tts_outputs` / `600` | TTL-managed cache for the long-output `audio_url` shape. |
| `TTS_MAX_SENTENCE_CHARS` | `400` | Hard-flush threshold for the streaming sentence buffer. |
| `TTS_MAX_CONCURRENCY` | `2` | Max concurrent Piper subprocesses (Pi-5-friendly default). |
| `LLAMA_CPP_REF` | `b8837` | Build-time pin for llama.cpp. The default tag includes the early-April 2026 native Gemma 4 audio merge. Bump to any newer tag from [`ggml-org/llama.cpp/tags`](https://github.com/ggml-org/llama.cpp/tags). |
| `MMPROJ_USE_GPU` | `false` | When `false` (Pi 5 default), `llama-server` is launched with `--no-mmproj-offload` so the unified vision/audio projector stays on CPU. The GPU compose override flips this to `true`. |

### Switching model

1. Pick a config from `model_configs/` (HF filenames are case-sensitive — keep the uppercase `E4B`/`E2B`/`A4B`/`B`):
   - `gemma4-e4b.yaml` -- **Gemma 4 E4B** (4B effective). Default. Fits Pi 5 16 GB. Image + audio (powers all five mode endpoints).
   - `gemma4-e2b.yaml` -- **Gemma 4 E2B** (2B effective). Lighter Pi-friendly sibling. Image + audio.
   - `gemma4-26b-a4b.yaml` -- **Gemma 4 26B A4B** (MoE; ~3.8B active). Needs ~16 GB RAM. Image only — `/audio/listen`, `/free-convo/turn`, `/voice-mirror/score`, and `/vision/teach-object` return `409` on this config.
   - `gemma4-31b.yaml` -- **Gemma 4 31B** (dense). Needs ~20 GB RAM or GPU offload. Image only (same `409` caveat).
2. Update `.env`:
   ```env
   MODEL_CONFIG_PATH=model_configs/gemma4-e4b.yaml
   MODEL_REPO=unsloth/gemma-4-E4B-it-GGUF
   MODEL_FILE=gemma-4-E4B-it-Q4_0.gguf
   MMPROJ_REPO=unsloth/gemma-4-E4B-it-GGUF
   MMPROJ_FILE=mmproj-F16.gguf
   ```
3. `docker compose up -d --build` (the new GGUF downloads to `./models` once and is reused thereafter).

If a default repo becomes unavailable, override `MODEL_REPO`/`MODEL_FILE` to point at any community mirror that exposes the same `.gguf` (e.g. a `bartowski/...` or `ggml-org/...` quantization).

---

## Operating the stack

- **Logs:** `docker compose logs -f api` and `docker compose logs -f llama`.
- **Restart on boot:** both services use `restart: unless-stopped`; ensure the Docker daemon itself starts on boot (`sudo systemctl enable docker`).
- **Updating code:** `git pull && docker compose up -d --build` rebuilds only what changed; `./models/` is preserved across rebuilds.
- **Swapping models:** edit `.env`, then `docker compose up -d`; you can `docker compose restart llama` after the new GGUF finishes downloading.
- **Disk usage:** the GGUF lives under `./models/`. Q4_0 E2B is ~1.5 GB (Q4_0 E4B is ~2.7 GB); the `mmproj` adds ~950 MB at F16.

### Host-side tuning on Pi 5

These knobs live outside the container but materially affect inference speed:

- **CPU governor**: switch to `performance` so the Pi doesn't downclock the A76 cores between requests:

  ```bash
  sudo cpupower frequency-set -g performance
  # persist across reboot:
  echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
  ```

- **Active cooling**: a fan-equipped case (the official Pi 5 Active Cooler is plenty) is non-negotiable for sustained inference. Without one, the SoC throttles from 2.4 GHz down to ~1.5 GHz under sustained load, which silently halves your tokens/sec. Verify with `vcgencmd measure_clock arm` while a request is in flight; you should see ~2.4 GHz, not lower.

- **NVMe SSD**: doesn't change steady-state throughput, but eliminates the cold-start model-load minute on first boot (an SD-card load of a 4-5 GB GGUF can take 60-90 s). Either an HAT-mounted NVMe or a USB 3.0 SSD works.

- **Container memlock limit**: `MLOCK=true` (default) needs an unlimited `RLIMIT_MEMLOCK` inside the container, which Docker's defaults don't grant. The `llama` service in `docker-compose.yml` therefore ships with `ulimits: { memlock: -1 }`. If you set `MLOCK=false`, you can drop that block; if you ever see `failed to mlock ... Cannot allocate memory` in `docker compose logs llama`, the ulimit is the thing to check first.

---

## Repository layout

```
.
├── docker-compose.yml            # llama + api
├── docker-compose.gpu.yml        # CUDA override
├── docker/
│   ├── Dockerfile.llama          # multi-stage llama.cpp build
│   ├── Dockerfile.api            # FastAPI image
│   └── download_model.sh         # idempotent HF download
├── model_configs/                # per-model YAML
├── app/
│   ├── main.py                   # FastAPI app + middleware + lifespan
│   ├── llama_adapter.py          # async client for llama.cpp
│   ├── routers/                  # one file per endpoint
│   ├── prompts.py                # Jinja2 prompt rendering
│   ├── media.py                  # URL / base64 image normalization
│   ├── schemas.py                # Pydantic request/response models
│   └── templates/playground.html # GET /
├── models/                       # mounted volume (gitignored)
└── specification.md              # source-of-truth spec
```

---

## Notes & caveats

- **Audio is live on E2B/E4B.** The April 2026 llama.cpp release bundles the native Gemma 4 audio path; the unified mmproj projector that ships in the same Unsloth GGUF repo (default: `mmproj-F16.gguf`; `mmproj-BF16.gguf` and `mmproj-F32.gguf` also available) carries both vision and audio adapters, so a single `--mmproj` flag powers `/free-convo/turn`, `/voice-mirror/score`, `/vision/teach-object`, and `/audio/listen` end-to-end. The 26B-A4B and 31B model configs intentionally keep `audio: false` because Google did not release native audio adapters for those sizes.
- **Audio context budget.** Audio tokens are denser than text. `CONTEXT_SIZE=8192` (default) handles ~30 s clips comfortably; bump to `16384` for longer recordings if Pi RAM allows.
- **Model availability:** the `unsloth/gemma-4-*` GGUF mirrors are the default; other community mirrors (e.g. `bartowski/...`, `ggml-org/...`) are equally usable via `MODEL_REPO`/`MODEL_FILE`.
- **Pi performance:** ~4–8 s for a 5–10 s spoken question via `/audio/listen`; ~6–10 s for `/vision/teach-object` (single 896 px frame + spoken guess). With flash attention + q8_0 KV cache + F16 mmproj + Q4_0 weights enabled (see `.env.example`), end-to-end latency is roughly 30–50% lower than the original BF16 / Q4_K_M defaults.
- **Future work:** see `specification.md` section 15 (RolePlay summative scoring, embeddings-driven stumble memory, GPU auto-detection, model hot-swap).

---

## Text-to-Speech (Piper)

Every mode endpoint above accepts two optional fields -- `tts: true` and a curated
`voice` id (default `warm-academic`) -- to opt into Piper TTS audio output.

- **Non-streaming responses** (`/free-convo/turn`, `/voice-mirror/suggest`,
  `/voice-mirror/score`, `/vision/teach-object`, and `/audio/listen`
  without `stream=true`) return an `audio_url` field on the JSON response
  pointing at a cached WAV. The device fetches that URL and plays it back;
  files are evicted after `TTS_OUTPUT_TTL_SECONDS` (default `600`).
- **`/audio/listen` with `stream=true`** interleaves
  `{"type":"audio", ...}` NDJSON lines (base64 WAV per completed sentence)
  alongside the existing `{"type":"text"}` chunks. Audio playback can start
  as soon as the first sentence ends, in parallel with token generation.

The TTS subsystem ships with five curated voice personalities
(`warm-academic`, `friendly-casual`, `neutral-news`, `energetic-kid`,
`calm-storyteller`) registered in `app/tts/voices.py`. Trim
`TTS_VOICES_ENABLED` to shorten api container cold-start.

The api container installs the prebuilt `piper_linux_aarch64` binary from
[`rhasspy/piper`](https://github.com/rhasspy/piper/releases) so audio runs
on Raspberry Pi 5 with no native dependencies. The Dockerfile also fetches
`piper_linux_x86_64` for dev hosts via `${TARGETARCH}`.
