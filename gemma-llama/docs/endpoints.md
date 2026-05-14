# Personalized Educational Tutor API

`gemma-llama` exposes a small, opinionated HTTP surface designed around a
single use-case: an **on-device personal tutor** that runs 24/7 on a
Raspberry Pi 5 (16 GB) with **Gemma 4 E4B** as the default model. Each
endpoint is named after the role it plays in a student's day — homework
checker, classroom ear, translator, lab assistant — so the API reads as the
tutor's set of senses.

The service speaks plain JSON for text routes and `multipart/form-data` for
the media-bearing tutor routes (vision / audio / video). All routes return
JSON.

> **Defaults you should know about**
>
> - **Active model**: Gemma 4 E4B (it), Q4_0 quantisation (Pi 5 default; flip to a K-quant on CUDA / Apple-silicon), with vision +
>   audio + video enabled by default. The April 2026 Gemma 4 release ships
>   the official Google projector inside `unsloth/gemma-4-E4B-it-GGUF`
>   (default `mmproj-F16.gguf`; F16 is the fastest precision on Pi 5 /
>   consumer CPUs, switch to `mmproj-BF16.gguf` on CUDA / Apple-silicon
>   hosts), so the multimodal tutor routes work out of the box. To run
>   text-only, blank `MMPROJ_REPO`/`MMPROJ_FILE` in
>   [`.env.example`](../.env.example) and flip the modality flags off in
>   `model_configs/gemma4-e4b.yaml`.
> - **Modality order (per official Gemma 4 docs)**: visual and audio tokens
>   are placed **before** the text prompt in the internal payload for the
>   best multimodal reasoning. This is enforced server-side by
>   [`app/media.py`](../app/media.py).
> - **Pi 5 tuning**: images are downscaled to 896 px (Gemma 4's vision
>   tile), audio is transcoded to mono 16 kHz WAV (the format mtmd's Gemma 4
>   audio encoder expects), video is sampled into ≤ 12 evenly spaced JPEG
>   frames + an optional audio track via a single `ffmpeg` pass.
> - **Audio support** uses llama.cpp's natively-merged Gemma 4 audio path
>   plus the official unified mmproj projector (F16 / BF16 / F32 all live
>   in the same Unsloth Gemma 4 repo as the model weights).

---

## Table of contents

- [Core](#core)
  - [`GET /`](#get-)
  - [`GET /health`](#get-health)
- [Text tutor](#text-tutor)
  - [`POST /generate`](#post-generate)
  - [`POST /chat`](#post-chat)
  - [`POST /classify`](#post-classify)
  - [`POST /extract`](#post-extract)
  - [`POST /summarize`](#post-summarize)
- [Vision tutor](#vision-tutor)
  - [`POST /vision/explain-work`](#post-visionexplain-work) — *The Homework Checker*
- [Audio tutor](#audio-tutor)
  - [`POST /audio/listen`](#post-audiolisten) — *The Classroom Ear*
  - [`POST /audio/translate`](#post-audiotranslate)
  - [`POST /audio/transcribe`](#post-audiotranscribe)
- [Video tutor](#video-tutor)
  - [`POST /video/analyze-process`](#post-videoanalyze-process) — *The Lab Assistant*
- [Text-to-Speech (Piper)](#text-to-speech-piper)
  - [Opt-in audio output on every endpoint](#opt-in-audio-output-on-every-endpoint)
  - [`GET /tts/voices`](#get-ttsvoices)
  - [`GET /tts/output/{id}.wav`](#get-ttsoutputidwav)
  - [`POST /tts/speak`](#post-ttsspeak)

---

## Core

### `GET /`

Interactive HTML playground for poking at every endpoint from the browser.
Not part of the JSON API.

### `GET /health`

```json
{
  "status": "ok",
  "model": "E4B",
  "llama_server_reachable": true,
  "tts_ready": true
}
```

Returns `503` with `llama_server_reachable: false` if the upstream
`llama-server` cannot be reached. `tts_ready` is `true` when the Piper
binary loaded **and** at least one voice was found in
`${TTS_VOICES_DIR}` at boot.

---

## Text tutor

These are the building-block routes that the higher-level tutor endpoints
also use under the hood.

### `POST /generate`

Single-prompt text (or text + image) generation. Set `stream: true` to
receive an NDJSON stream of `{"type": "thinking" | "text", "content": "..."}`
chunks.

Request body:

```json
{
  "prompt": "Explain photosynthesis to a 9-year-old.",
  "system": null,
  "images": null,
  "max_tokens": 512,
  "temperature": 0.7,
  "thinking": true,
  "stream": false
}
```

`images[]` is the typed `ImageInput` form (`url` *or* `base64` + optional
`mime_type`).

### `POST /chat`

Multi-turn chat with the same multimodal support as `/generate`. The
`messages` array follows OpenAI's `{role, content}` convention and may carry
`images` per user message.

### `POST /classify`

Force-shapes a single-label or multi-label classification into structured
JSON. Useful for routing a student's question into a subject area.

```json
{
  "text": "What's the difference between mitosis and meiosis?",
  "labels": ["biology", "chemistry", "physics", "math"],
  "multi_label": false
}
```

### `POST /extract`

JSON Schema-style structured extraction from arbitrary text. Useful for
pulling key fields out of a student's notes.

### `POST /summarize`

ROUGE-friendly summarisation. Useful for condensing a long passage into a
quick recap.

---

## Vision tutor

### `POST /vision/explain-work`

> **Tutor use-case: The Homework Checker.** The student snaps a photo of
> their notebook or whiteboard. Gemma reads the work, points out what is
> correct, identifies the *first* mistake (if any), and suggests the next
> step — without solving the whole problem.

`Content-Type: multipart/form-data`

| Field         | Type   | Required | Notes                                                           |
| ------------- | ------ | -------- | --------------------------------------------------------------- |
| `image`       | file   | yes      | JPEG/PNG/WebP. Resized server-side to ≤ 896 px long edge.       |
| `question`    | string | no       | What the student is asking ("Did I solve part b correctly?").   |
| `subject`     | string | no       | Hint, e.g. `"algebra"`, `"organic chemistry"`.                  |
| `max_tokens`  | int    | no       | Defaults to model config.                                       |
| `temperature` | float  | no       | Defaults to model config.                                       |

Sample call:

```bash
curl -X POST http://localhost:8010/vision/explain-work \
    -F "image=@notebook.jpg" \
    -F "subject=algebra" \
    -F "question=Where did I go wrong on step 3?"
```

Sample response:

```json
{
  "text": "Nice work setting up the equation. Steps 1 and 2 are correct...",
  "model": "E4B",
  "inference_time_ms": 4830.12,
  "usage": {"prompt_tokens": 612, "completion_tokens": 188, "total_tokens": 800, "tokens_predicted": 188}
}
```

Pi 5 latency ballpark: ~4–7 s end-to-end for a single 896 px page.

---

## Audio tutor

Both audio routes accept any common container (`wav`, `mp3`, `m4a`, `ogg`,
`flac`, `webm`, ...). The api container transcodes uploads once with
`ffmpeg` to **mono 16 kHz WAV** before forwarding to Gemma.

### `POST /audio/listen`

> **Tutor use-case: The Classroom Ear.** Processes raw audio of a student's
> spoken question and returns a text answer.

`Content-Type: multipart/form-data`

| Field         | Type  | Required | Notes                                                                 |
| ------------- | ----- | -------- | --------------------------------------------------------------------- |
| `audio`       | file  | yes      | Up to 25 MiB; transcoded to mono 16 kHz WAV.                          |
| `max_tokens`  | int   | no       | Defaults to model config.                                             |
| `temperature` | float | no       | Defaults to model config.                                             |
| `stream`      | bool  | no       | If `true`, returns `application/x-ndjson` (`{"type","content"}` lines). |

Sample call (non-streaming):

```bash
curl -X POST http://localhost:8010/audio/listen \
    -F "audio=@question.m4a"
```

Sample call (NDJSON streaming, low-latency UX):

```bash
curl -N -X POST http://localhost:8010/audio/listen \
    -F "audio=@question.m4a" -F "stream=true"
```

Sample response:

```json
{
  "text": "Great question! Mitosis produces two genetically identical cells...",
  "model": "E4B",
  "inference_time_ms": 5210.55,
  "usage": {"prompt_tokens": 412, "completion_tokens": 220, "total_tokens": 632, "tokens_predicted": 220}
}
```

Pi 5 latency ballpark: ~4–8 s for a 5–10 s spoken question.

### `POST /audio/translate`

> **Tutor use-case:** For non-native speakers. Takes audio in language A and
> returns text in language B. Gemma 4's instruct training covers 35+
> languages out of the box.

`Content-Type: multipart/form-data`

| Field             | Type   | Required | Notes                                                              |
| ----------------- | ------ | -------- | ------------------------------------------------------------------ |
| `audio`           | file   | yes      | Up to 25 MiB; transcoded to mono 16 kHz WAV.                       |
| `target_language` | string | yes      | Free-form name or BCP-47 code, e.g. `"English"`, `"es"`, `"hi"`.   |
| `source_language` | string | no       | Optional hint to disambiguate similar languages.                   |
| `max_tokens`      | int    | no       | Defaults to model config.                                          |
| `temperature`     | float  | no       | Defaults to `0.2` for translation determinism.                     |

Sample call:

```bash
curl -X POST http://localhost:8010/audio/translate \
    -F "audio=@bonjour.m4a" \
    -F "target_language=English" \
    -F "source_language=French"
```

Sample response:

```json
{
  "text": "Hello, how are you today?",
  "target_language": "English",
  "source_language": "French",
  "model": "E4B",
  "inference_time_ms": 4480.10,
  "usage": {"prompt_tokens": 388, "completion_tokens": 12, "total_tokens": 400, "tokens_predicted": 12}
}
```

### `POST /audio/transcribe`

> **Tutor use-case:** Verbatim speech-to-text in the source language. Useful
> as a building block (drop-in ASR) or for archiving spoken homework
> reflections. Defaults to `temperature=0` and `thinking=false` server-side
> for deterministic transcripts.

`Content-Type: multipart/form-data`

| Field         | Type  | Required | Notes                                                  |
| ------------- | ----- | -------- | ------------------------------------------------------ |
| `audio`       | file  | yes      | Up to 25 MiB; transcoded to mono 16 kHz WAV.           |
| `max_tokens`  | int   | no       | Defaults to model config.                              |
| `temperature` | float | no       | Defaults to `0.0` for deterministic transcripts.       |

Sample call:

```bash
curl -X POST http://localhost:8010/audio/transcribe \
    -F "audio=@lecture-snippet.m4a"
```

Sample response:

```json
{
  "text": "Mitosis produces two genetically identical daughter cells.",
  "model": "E4B",
  "inference_time_ms": 3120.45,
  "usage": {"prompt_tokens": 388, "completion_tokens": 14, "total_tokens": 402, "tokens_predicted": 14}
}
```

---

## Video tutor

### `POST /video/analyze-process`

> **Tutor use-case: The Lab Assistant.** The student records a short clip
> of a science experiment or a hands-on task (titration, soldering,
> threading a needle, knife skills, ...) and asks "did I do that right?".
> The api container samples evenly spaced frames + extracts the audio track
> via a single `ffmpeg` pass, then sends both to Gemma 4 in one multimodal
> turn.

`Content-Type: multipart/form-data`

| Field           | Type   | Required | Notes                                                               |
| --------------- | ------ | -------- | ------------------------------------------------------------------- |
| `video`         | file   | yes      | Up to 75 MiB; any container ffmpeg can read.                        |
| `task`          | string | no       | One-line description of what the student is trying to do.           |
| `n_frames`      | int    | no       | Default `6`, capped at `12` on Pi 5.                                |
| `include_audio` | bool   | no       | Default `true`. Falls back to frames-only if there is no audio.     |
| `max_tokens`    | int    | no       | Defaults to model config.                                           |
| `temperature`   | float  | no       | Defaults to model config.                                           |
| `stream`        | bool   | no       | If `true`, returns NDJSON. The first line is a `{"type":"meta"}` frame with `frames_used` / `audio_used` / `duration_s`; subsequent lines are token deltas. JSON-format enforcement is disabled on the streaming path -- clients must concatenate `text` chunks before parsing. |

The model is asked to reply as a JSON object with keys `summary`,
`observations`, `safety_notes`, and `next_step`. The endpoint returns that
JSON verbatim in `text` so callers can either parse it or pass it straight
through to a UI.

Sample call:

```bash
curl -X POST http://localhost:8010/video/analyze-process \
    -F "video=@titration.mp4" \
    -F "task=titrate NaOH into HCl until colour change" \
    -F "n_frames=4"
```

Sample response:

```json
{
  "text": "{\"summary\": \"Student adds NaOH dropwise...\", \"observations\": [...], \"safety_notes\": [...], \"next_step\": \"...\"}",
  "frames_used": 8,
  "audio_used": true,
  "duration_s": 11.42,
  "model": "E4B",
  "inference_time_ms": 18420.55,
  "usage": {"prompt_tokens": 2104, "completion_tokens": 312, "total_tokens": 2416, "tokens_predicted": 312}
}
```

Pi 5 latency ballpark: ~12–25 s for a 6-frame clip with audio. Each
additional frame adds ~250–400 ms of vision-encode time.

---

## Text-to-Speech (Piper)

Every endpoint above accepts two extra opt-in parameters:

| Field   | Type   | Default          | Notes                                                                 |
| ------- | ------ | ---------------- | --------------------------------------------------------------------- |
| `tts`   | bool   | `false`          | If `true`, also render the response text via Piper TTS.               |
| `voice` | string | `warm-academic`  | Curated personality id. Call `GET /tts/voices` to list available ids. |

The TTS subsystem runs the prebuilt Piper binary from
[`rhasspy/piper`](https://github.com/rhasspy/piper/releases) installed in the
`api` container. Piper is statically linked and ships an `aarch64` build that
runs unmodified on Raspberry Pi 5 (and `x86_64` for dev hosts).

### Opt-in audio output on every endpoint

There are three response shapes, picked per endpoint based on expected audio
length:

* **Inline base64** -- `/classify`, `/audio/translate`. The response JSON
  gains:
  ```json
  {"audio_base64": "...", "audio_mime": "audio/wav", "audio_duration_ms": 1200, "voice": "warm-academic"}
  ```
* **File URL** -- `/generate` (non-stream), `/chat` (non-stream),
  `/summarize`, `/extract`, `/vision/explain-work`, `/audio/listen`
  (non-stream), `/audio/transcribe`, `/video/analyze-process`
  (non-stream). The response JSON gains:
  ```json
  {"audio_url": "/tts/output/abc123.wav", "audio_duration_ms": 7200, "voice": "warm-academic"}
  ```
  Files are evicted after `TTS_OUTPUT_TTL_SECONDS` (default 600).
* **Streaming NDJSON** -- on `/generate`, `/chat`, `/audio/listen`, and
  `/video/analyze-process` when `stream=true`. The wrapper buffers Gemma's
  text deltas until a sentence boundary, hands the sentence to Piper, and
  emits a new line per completed sentence:
  ```json
  {"type":"audio","seq":1,"sentence":"Photosynthesis is...","content":"<base64-wav>","mime":"audio/wav","sample_rate":22050,"voice":"warm-academic"}
  ```
  These lines are interleaved with the existing `{"type":"thinking" | "text" | "meta" | "heartbeat"}`
  lines and are produced asynchronously so token throughput is never blocked
  by audio synth latency. A `{"type":"audio_error"}` line is emitted in
  place of the WAV when synthesis fails for a single sentence; the rest of
  the stream is unaffected.

If the engine is disabled (`TTS_ENABLED=false`) or no voices were
downloaded, every response gains `audio_error` instead of audio so callers
can degrade gracefully without branching on engine state.

Sample call (curl):

```bash
curl -X POST http://localhost:8010/summarize \
    -H 'content-type: application/json' \
    -d '{"text":"Small language models compress capability...","tts":true,"voice":"calm-storyteller"}'
```

### `GET /tts/voices`

```json
{
  "default": "warm-academic",
  "voices": [
    {
      "id": "warm-academic",
      "description": "Warm, clear academic narration. The default tutor voice.",
      "language": "en_US",
      "voice_id": "en_US-lessac-medium",
      "downloaded": true,
      "sample_rate": 22050,
      "is_default": true
    },
    ...
  ]
}
```

`downloaded=false` means the voice is in the registry but not present on
disk -- add it to `TTS_VOICES_ENABLED` and restart the api container, or
call the registry's underlying voice from `rhasspy/piper-voices` manually.

### `GET /tts/output/{id}.wav`

Serves a previously-rendered WAV from the file cache. Returns `404` once
`TTS_OUTPUT_TTL_SECONDS` has elapsed (or if the id never existed).

### `POST /tts/speak`

One-shot TTS for arbitrary text. Useful for the playground and for clients
that want audio without running a full inference round-trip.

```json
{"text": "Hello, world.", "voice": "friendly-casual"}
```

Response:

```json
{
  "audio_base64": "...",
  "audio_mime": "audio/wav",
  "audio_duration_ms": 980.0,
  "voice": "friendly-casual"
}
```

---

## Notes & gotchas

- **Modality order matters.** Per the Gemma 4 docs, image and audio parts
  must precede the text part in the user message. The server enforces this
  in `build_user_content` ([`app/media.py`](../app/media.py)); clients of
  `/chat` who hand-craft `messages` should follow the same rule.
- **Audio formats**: clients can send anything `ffmpeg` understands; the
  server normalises to mono 16 kHz WAV before forwarding to Gemma.
- **Video has no native llama.cpp path** as of April 2026. The
  `/video/analyze-process` endpoint sidesteps that by sampling frames + an
  audio track in the api container.
- **Switching models**: set `MODEL_CONFIG_PATH` (and the matching
  `MODEL_REPO`/`MODEL_FILE`/`MMPROJ_*`) in `.env`. Vision/audio/video
  endpoints return `409` if the active config has the corresponding
  modality disabled.
- **Upload caps** (defined in [`app/media_multipart.py`](../app/media_multipart.py)):
  10 MiB images, 25 MiB audio, 75 MiB video. Adjust there if your hardware
  budget allows.
