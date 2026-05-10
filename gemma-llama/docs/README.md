# gemma-llama

Portable, production-ready local LLM inference for the **Gemma 4** family (April 2026 release: E2B, E4B, 26B A4B MoE, and 31B dense GGUFs from Unsloth), packaged as a two-container Docker Compose stack:

- **`llama`** — `llama.cpp`'s `llama-server` with the GGUF model (and vision projector) auto-downloaded from Hugging Face on first boot.
- **`api`** — FastAPI service exposing OpenAI-style chat completions plus task-specific endpoints (`/generate`, `/chat`, `/classify`, `/extract`, `/summarize`) and a built-in playground UI.

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
| `GET`  | `/health` | `{"status","model","llama_server_reachable"}`; `503` when `llama` is unreachable. |
| `POST` | `/generate` | Single-prompt text generation; supports image attachments and SSE-style NDJSON streaming. |
| `POST` | `/chat` | Multi-turn chat; per-message image attachments. |
| `POST` | `/classify` | Single- or multi-label classification (uses `response_format=json_object`). |
| `POST` | `/extract` | Schema-driven structured extraction; tolerant to malformed JSON via `_raw` / `_error`. |
| `POST` | `/summarize` | Abstractive summarization with optional `style` and `max_sentences`. |
| `POST` | `/vision/explain-work` | "The Homework Checker" -- multipart image + question, tutor-style feedback. |
| `POST` | `/audio/listen` | "The Classroom Ear" -- multipart audio question, tutor answer in text. Supports `stream=true`. |
| `POST` | `/audio/translate` | Multipart audio in language A -> text in language B. |
| `POST` | `/audio/transcribe` | Multipart audio -> verbatim transcript in the source language. |
| `POST` | `/video/analyze-process` | "The Lab Assistant" -- multipart short clip; samples N frames + audio via ffmpeg. Supports `stream=true`. |
| `GET`  | `/tts/voices` | List the curated Piper voice "personalities" (default, language, downloaded?). |
| `GET`  | `/tts/output/{id}.wav` | Serve a previously-rendered WAV from the TTL cache. |
| `POST` | `/tts/speak` | One-shot Piper synthesis for arbitrary text. |

Every successful response includes `inference_time_ms`; `/generate` and `/chat` additionally surface `tokens_predicted` and full `usage`. Every response carries an `X-Process-Time-Ms` header.

### Streaming

Set `"stream": true` on `/generate` or `/chat`. The body is `application/x-ndjson`; each line is one of:

```json
{"type": "thinking", "content": "..."}
{"type": "text",     "content": "..."}
{"type": "audio",    "seq": 1, "sentence": "...", "content": "<base64-wav>", "mime": "audio/wav", "sample_rate": 22050, "voice": "warm-academic"}
{"type": "error",    "content": "..."}
```

`audio` lines are emitted only when `tts: true` is set on the request -- one
per completed sentence -- so audio plays back while Gemma is still
generating the next sentence. See [Text-to-Speech (Piper)](endpoints.md#text-to-speech-piper)
for the full opt-in surface.

### Thinking toggle

`thinking: true` (default) preserves the model's chain-of-thought. `thinking: false` (a) sends `chat_template_kwargs.enable_thinking=false` to llama.cpp so chat templates that honour it (Qwen3, newer Gemma builds) skip emitting `<think>` blocks, (b) appends `/no_think` to the last user message as a Qwen3-style fallback, and (c) filters any `reasoning_content` deltas or stray `<think>...</think>` spans in the adapter so neither the streaming NDJSON nor the response `text` ever contains reasoning. With `thinking: false` the dashboard will not show `[think] ...` lines.

### Image attachments

Both `/generate` (top-level `images`) and `/chat` (per-message `images`) accept an array of `ImageInput`s; each is either:

```json
{"url": "https://example.com/cat.jpg"}
{"url": "data:image/png;base64,iVBORw0K..."}
{"base64": "iVBORw0K...", "mime_type": "image/png"}
```

Images are forwarded to `llama-server` as OpenAI-style `image_url` content parts and routed through the configured `mmproj` projector.

### Example calls

```bash
curl -s localhost:8010/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"In one sentence, what is an SLM?","thinking":false}' | jq
```

```bash
curl -s localhost:8010/classify \
  -H 'content-type: application/json' \
  -d '{"text":"Battery dies fast.","labels":["positive","negative","neutral"]}' | jq
```

```bash
curl -s localhost:8010/extract \
  -H 'content-type: application/json' \
  -d '{"text":"Anjali, party of 4, 7:30 PM Friday at Ocean Grill.",
       "schema":{"name":"string","party_size":"integer","time":"string","day":"string","venue":"string"}}' | jq
```

```bash
# vision: notebook -> tutor feedback
curl -s -X POST localhost:8010/vision/explain-work \
  -F "image=@notebook.jpg" -F "subject=algebra" \
  -F "question=Where did I go wrong on step 3?" | jq
```

```bash
# audio: spoken question -> answer (NDJSON streamed)
curl -s -N -X POST localhost:8010/audio/listen \
  -F "audio=@question.m4a" -F "stream=true"

# audio: verbatim transcription
curl -s -X POST localhost:8010/audio/transcribe \
  -F "audio=@question.m4a" | jq

# audio: French clip -> English text
curl -s -X POST localhost:8010/audio/translate \
  -F "audio=@bonjour.m4a" -F "target_language=English" -F "source_language=French" | jq
```

```bash
# video: short experiment clip -> JSON tutor response
curl -s -X POST localhost:8010/video/analyze-process \
  -F "video=@titration.mp4" \
  -F "task=titrate NaOH into HCl until colour change" \
  -F "n_frames=8" | jq
```

---

## Configuration

All settings come from environment variables (typically via `.env`). See [`.env.example`](.env.example) for the canonical list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_PORT` | `8010` | Host port for the FastAPI service. |
| `MODEL_CONFIG_PATH` | `model_configs/gemma4-e4b.yaml` | Active model YAML (consumed by the `api` container). |
| `MODEL_REPO` / `MODEL_FILE` | `unsloth/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_K_M.gguf` | HF source for the GGUF (consumed by `download_model.sh`). |
| `MMPROJ_REPO` / `MMPROJ_FILE` | `unsloth/gemma-4-E4B-it-GGUF` / `mmproj-BF16.gguf` | Official Google vision/audio projector. Set both to empty for a text-only deployment. |
| `HF_TOKEN` | _unset_ | Required only for gated repos. |
| `CONTEXT_SIZE` | `8192` | `llama-server -c`. |
| `THREADS` | `4` (Pi) | `llama-server -t`; bump to your physical core count on dev boxes. |
| `GPU_LAYERS` | `0` | `llama-server -ngl`; flipped to `99` by `docker-compose.gpu.yml`. |
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
   - `gemma4-e4b.yaml` -- **Gemma 4 E4B** (4B effective). Default. Fits Pi 5 16 GB. Text + image + audio + video.
   - `gemma4-e2b.yaml` -- **Gemma 4 E2B** (2B effective). Lighter Pi-friendly sibling. Text + image + audio + video.
   - `gemma4-26b-a4b.yaml` -- **Gemma 4 26B A4B** (MoE; ~3.8B active). Needs ~16 GB RAM. Text + image only.
   - `gemma4-31b.yaml` -- **Gemma 4 31B** (dense). Needs ~20 GB RAM or GPU offload. Text + image only.
2. Update `.env`:
   ```env
   MODEL_CONFIG_PATH=model_configs/gemma4-e4b.yaml
   MODEL_REPO=unsloth/gemma-4-E4B-it-GGUF
   MODEL_FILE=gemma-4-E4B-it-Q4_K_M.gguf
   MMPROJ_REPO=unsloth/gemma-4-E4B-it-GGUF
   MMPROJ_FILE=mmproj-BF16.gguf
   ```
3. `docker compose up -d --build` (the new GGUF downloads to `./models` once and is reused thereafter).

If a default repo becomes unavailable, override `MODEL_REPO`/`MODEL_FILE` to point at any community mirror that exposes the same `.gguf` (e.g. a `bartowski/...` or `ggml-org/...` quantization).

---

## Evaluation harness

A small offline harness lives in `eval/` and exercises the running API.

```bash
pip install -r requirements-eval.txt

# CLI runner -- writes timestamped JSON into eval/reports/
python -m eval.runner --all
python -m eval.runner --task classify
python -m eval.runner --task extract
python -m eval.runner --task summarize

# Pytest -- skipped automatically if the API is not reachable
pytest eval/ -v -s
```

Datasets live in `eval/datasets/`; thresholds enforced by the pytest tests:

| Task | Pass threshold |
|------|----------------|
| Classification | `accuracy >= 0.5` |
| Extraction | `json_valid_rate >= 0.8` |
| Summarization | `ROUGE-1 >= 0.2` |

All tasks also report latency `p50/p95/p99/mean` from the API's `inference_time_ms` field.

---

## Operating the stack

- **Logs:** `docker compose logs -f api` and `docker compose logs -f llama`.
- **Restart on boot:** both services use `restart: unless-stopped`; ensure the Docker daemon itself starts on boot (`sudo systemctl enable docker`).
- **Updating code:** `git pull && docker compose up -d --build` rebuilds only what changed; `./models/` is preserved across rebuilds.
- **Swapping models:** edit `.env`, then `docker compose up -d`; you can `docker compose restart llama` after the new GGUF finishes downloading.
- **Disk usage:** the GGUF lives under `./models/`. Q4_K_M E2B is ~1.6 GB; the `mmproj` adds a few hundred MB.

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
├── eval/                         # offline evaluation harness
├── models/                       # mounted volume (gitignored)
└── specification.md              # source-of-truth spec
```

---

## Notes & caveats

- **Audio + Video are live on E2B/E4B.** The April 2026 llama.cpp release bundles the native Gemma 4 audio path; the unified `mmproj-BF16.gguf` projector that ships in the same Unsloth GGUF repo carries both vision and audio adapters, so a single `--mmproj` flag enables `/vision/*`, `/audio/*`, and `/video/*` end-to-end. The 26B-A4B and 31B model configs intentionally keep `audio: false` / `video: false` because Google did not release native audio adapters for those sizes.
- **Audio context budget.** Audio tokens are denser than text. `CONTEXT_SIZE=8192` (default) handles ~30 s clips comfortably; bump to `16384` for longer recordings if Pi RAM allows.
- **Video has no native llama.cpp path yet.** `/video/analyze-process` sidesteps that by sampling N evenly spaced frames + the audio track in one ffmpeg pass and feeding both to Gemma 4 in a single multimodal turn.
- **Model availability:** the `unsloth/gemma-4-*` GGUF mirrors are the default; other community mirrors (e.g. `bartowski/...`, `ggml-org/...`) are equally usable via `MODEL_REPO`/`MODEL_FILE`.
- **Pi performance:** expect ~3-6 tokens/sec on Pi 5 with E2B Q4_K_M for text; ~4-8 s for a 5-10 s spoken question via `/audio/listen`; ~12-25 s for a 6-frame `/video/analyze-process` clip with audio.
- **Future work:** see `specification.md` section 16 (embeddings, GPU auto-detection, model hot-swap).

---

## Text-to-Speech (Piper)

Every endpoint above accepts two optional fields -- `tts: true` and a curated
`voice` id (default `warm-academic`) -- to opt into Piper TTS audio output:

- **Streaming endpoints** (`/generate`, `/chat`, `/audio/listen`,
  `/video/analyze-process` with `stream=true`) interleave new
  `{"type":"audio", ...}` NDJSON lines (base64 WAV per completed sentence)
  alongside the existing `{"type":"text"}` chunks. Audio playback can start
  as soon as the first sentence ends, in parallel with token generation.
- **Short non-streaming endpoints** (`/classify`, `/audio/translate`)
  inline the audio as `audio_base64` in the JSON response.
- **Long non-streaming endpoints** (`/generate`, `/chat`, `/summarize`,
  `/extract`, `/vision/explain-work`, `/audio/listen`, `/audio/transcribe`,
  `/video/analyze-process`) return an `audio_url` pointing at
  `GET /tts/output/{id}.wav`; files are evicted after
  `TTS_OUTPUT_TTL_SECONDS`.

The TTS subsystem ships with five curated voice personalities
(`warm-academic`, `friendly-casual`, `neutral-news`, `energetic-kid`,
`calm-storyteller`); list / inspect them via `GET /tts/voices`. Trim
`TTS_VOICES_ENABLED` to shorten api container cold-start.

The api container installs the prebuilt `piper_linux_aarch64` binary from
[`rhasspy/piper`](https://github.com/rhasspy/piper/releases) so audio runs
on Raspberry Pi 5 with no native dependencies. The Dockerfile also fetches
`piper_linux_x86_64` for dev hosts via `${TARGETARCH}`.
