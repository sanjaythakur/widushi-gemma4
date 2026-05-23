# Personalized Educational Tutor API

`gemma-llama` exposes a small, opinionated HTTP surface designed around a
single use-case: an **on-device personal English tutor** that runs 24/7 on
a Raspberry Pi 5 (16 GB) with **Gemma 4 E4B** as the default model. The
surface is intentionally narrow — exactly five Mic / Mic+Camera endpoints,
one per learner-facing interaction mode, plus two operational routes.

The service speaks `multipart/form-data` on the audio/image-bearing tutor
routes and plain JSON on `/voice-mirror/suggest`. All routes return JSON.

> **Defaults you should know about**
>
> - **Active model**: Gemma 4 E4B (it), Q4_0 quantisation (Pi 5 default;
>   flip to a K-quant on CUDA / Apple-silicon), with vision + audio enabled
>   by default. The April 2026 Gemma 4 release ships the official Google
>   projector inside `unsloth/gemma-4-E4B-it-GGUF` (default
>   `mmproj-F16.gguf`; F16 is the fastest precision on Pi 5 / consumer
>   CPUs, switch to `mmproj-BF16.gguf` on CUDA / Apple-silicon hosts), so
>   the multimodal tutor routes work out of the box.
> - **Modality order (per official Gemma 4 docs)**: visual and audio tokens
>   are placed **before** the text prompt in the internal payload for the
>   best multimodal reasoning. This is enforced server-side by
>   [`app/media.py`](../app/media.py).
> - **Pi 5 tuning**: images are downscaled to 896 px (Gemma 4's vision
>   tile), audio is transcoded to mono 16 kHz WAV (the format mtmd's Gemma
>   4 audio encoder expects).
> - **Audio support** uses llama.cpp's natively-merged Gemma 4 audio path
>   plus the official unified mmproj projector (F16 / BF16 / F32 all live
>   in the same Unsloth Gemma 4 repo as the model weights).

---

## Table of contents

- [Core](#core)
  - [`GET /`](#get-)
  - [`GET /health`](#get-health)
- [Mode endpoints](#mode-endpoints)
  - [`POST /free-convo/turn`](#post-free-convoturn) — *FreeConvoMode*
  - [`POST /voice-mirror/suggest`](#post-voice-mirrorsuggest) — *VoiceMirrorMode (pick word)*
  - [`POST /voice-mirror/score`](#post-voice-mirrorscore) — *VoiceMirrorMode (score attempt)*
  - [`POST /vision/teach-object`](#post-visionteach-object) — *VisionMode*
  - [`POST /audio/listen`](#post-audiolisten) — *RolePlayMode / general fallback*
- [TTS audio attach](#tts-audio-attach)

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

## Mode endpoints

Each endpoint below maps 1:1 to a runtime mode in the Widushi app
(`FreeConvoMode`, `VoiceMirrorMode`, `VisionMode`, `RolePlayMode`). They
all accept the optional `tts` / `voice` form fields and, when `tts=true`,
return a `audio_url` pointing at the rendered Piper WAV (TTL-cached on the
api container; eviction after `TTS_OUTPUT_TTL_SECONDS`).

### `POST /free-convo/turn`

> **Mode: FreeConvoMode (post-wake default).** Warm Hindi / English /
> Hinglish chat. Each turn replies to the learner's spoken audio **and**
> sets a `start_learning` boolean the device uses to gate the swap into
> structured English practice (`VoiceMirrorMode`).

`Content-Type: multipart/form-data`

| Field         | Type  | Required | Notes                                                                                |
| ------------- | ----- | -------- | ------------------------------------------------------------------------------------ |
| `audio`       | file  | yes      | Recording of the learner's spoken turn. Transcoded to mono 16 kHz WAV server-side.   |
| `max_tokens`  | int   | no       | 1–4096. Defaults to model config.                                                    |
| `temperature` | float | no       | 0.0–2.0. Defaults to `0.4` for friendly-but-grounded chat.                           |
| `tts`         | bool  | no       | If `true`, also render the reply via Piper TTS.                                      |
| `voice`       | str   | no       | Piper personality id (see Widushi's curated set). Defaults to `warm-academic`.       |

Sample call:

```bash
curl -X POST http://localhost:8010/free-convo/turn \
    -F "audio=@hello.wav" \
    -F "tts=true"
```

Sample response:

```json
{
  "text": "Hi! Want to practise some English with me today?",
  "start_learning": true,
  "transcript": "haan mujhe english seekhni hai",
  "model": "E4B",
  "inference_time_ms": 3120.45,
  "usage": {"prompt_tokens": 280, "completion_tokens": 32, "total_tokens": 312, "tokens_predicted": 32},
  "audio_url": "/tts/output/abc123.wav",
  "audio_duration_ms": 2400.0,
  "voice": "warm-academic"
}
```

`start_learning` is the JSON the model returned plus a keyword fallback
(`"learn english"`, `"sikha do"`, etc.) so the device can swap modes even
when the model emits prose instead of JSON. The endpoint never 500s on a
bad completion.

### `POST /voice-mirror/suggest`

> **Mode: VoiceMirrorMode — picking the next target word.** Text-only:
> ask the tutor which English word the learner should practise next.
> Returns the target word, an example sentence, an optional IPA-style
> pronunciation hint, and the **spoken cue text** the device should send
> through TTS (`"Try saying: apple. AP-uhl."`).

`Content-Type: application/json`

```json
{
  "level": "beginner",
  "history": ["apple", "water"],
  "tts": true,
  "voice": "warm-academic"
}
```

| Field    | Type     | Required | Notes                                                            |
| -------- | -------- | -------- | ---------------------------------------------------------------- |
| `level`  | string   | no       | Optional learner-level hint, e.g. `"beginner"`.                  |
| `history`| string[] | no       | Words already practised in this session — avoid repeating them.  |
| `tts`    | bool     | no       | If `true`, also render `prompt_text` via Piper TTS.              |
| `voice`  | string   | no       | Piper personality id. Defaults to `warm-academic`.               |

Sample response:

```json
{
  "word": "river",
  "example_sentence": "The river is wide.",
  "ipa_hint": "RIH-ver",
  "prompt_text": "Try saying: river. RIH-ver.",
  "model": "E4B",
  "inference_time_ms": 1820.5,
  "audio_url": "/tts/output/9f12c.wav",
  "audio_duration_ms": 1900.0,
  "voice": "warm-academic"
}
```

### `POST /voice-mirror/score`

> **Mode: VoiceMirrorMode — scoring the learner's attempt.** The device
> records the learner saying the target word and posts it alongside the
> word that was asked for. The endpoint returns a coarse `verdict`
> (`praise` / `correct` / `retry`) plus a short coaching line to be spoken
> back. `retry` keeps the same word; the other two unlock the next one.

`Content-Type: multipart/form-data`

| Field          | Type   | Required | Notes                                                                                |
| -------------- | ------ | -------- | ------------------------------------------------------------------------------------ |
| `audio`        | file   | yes      | Recording of the learner's pronunciation attempt.                                    |
| `target_word`  | string | yes      | The word the learner was asked to say (echoed back to the model in the system prompt). |
| `max_tokens`   | int    | no       | 1–2048. Defaults to `256`.                                                           |
| `temperature`  | float  | no       | 0.0–2.0. Defaults to `0.2` for stable verdicts.                                      |
| `tts`          | bool   | no       | If `true`, also render `feedback_text` via Piper TTS.                                |
| `voice`        | string | no       | Piper personality id. Defaults to `warm-academic`.                                   |

Sample response:

```json
{
  "target_word": "river",
  "transcript": "riwer",
  "verdict": "correct",
  "feedback_text": "Good attempt. Try 'river' a little slower — RIH-ver.",
  "model": "E4B",
  "inference_time_ms": 4210.1,
  "usage": {"prompt_tokens": 410, "completion_tokens": 28, "total_tokens": 438, "tokens_predicted": 28},
  "audio_url": "/tts/output/d3a4e.wav",
  "audio_duration_ms": 3000.0,
  "voice": "warm-academic"
}
```

Verdict semantics in `VoiceMirrorMode`:

- `praise` — clean attempt, advance immediately.
- `correct` — recognisable attempt, advance with a small correction.
- `retry` — unrecognised or wrong word; the mode replays the same target on the next turn.

### `POST /vision/teach-object`

> **Mode: VisionMode.** The device snaps a camera frame **and** records
> the learner's spoken guess at the same time, then posts both together.
> The tutor identifies the object and replies with a short teaching line
> of the form `"Yes, this is milk. Say: I drink milk."` — the learner
> hears it spoken back on the device.

`Content-Type: multipart/form-data`

| Field         | Type  | Required | Notes                                                                                |
| ------------- | ----- | -------- | ------------------------------------------------------------------------------------ |
| `image`       | file  | yes      | JPEG/PNG/WebP camera frame. Resized server-side to ≤ 896 px long edge.               |
| `audio`       | file  | yes      | Recording of the learner's spoken guess for the object.                              |
| `max_tokens`  | int   | no       | 1–2048. Defaults to `256`.                                                           |
| `temperature` | float | no       | 0.0–2.0. Defaults to `0.3` for stable identifications.                               |
| `tts`         | bool  | no       | If `true`, also render the teaching line via Piper TTS.                              |
| `voice`       | str   | no       | Piper personality id. Defaults to `warm-academic`.                                   |

Sample call:

```bash
curl -X POST http://localhost:8010/vision/teach-object \
    -F "image=@frame.jpg" \
    -F "audio=@guess.wav" \
    -F "tts=true"
```

Sample response:

```json
{
  "text": "Yes, this is milk. Say: I drink milk.",
  "object": "milk",
  "transcript": "doodh",
  "model": "E4B",
  "inference_time_ms": 7820.55,
  "usage": {"prompt_tokens": 740, "completion_tokens": 36, "total_tokens": 776, "tokens_predicted": 36},
  "audio_url": "/tts/output/7c81b.wav",
  "audio_duration_ms": 2200.0,
  "voice": "warm-academic"
}
```

Returns `409 Conflict` if the active model config has `modalities.image`
**or** `modalities.audio` disabled (only the E2B/E4B configs support
both; 26B-A4B and 31B are image-only).

Pi 5 latency ballpark: ~6–10 s for a single 896 px frame plus a short
spoken guess.

### `POST /audio/listen`

> **Mode: RolePlayMode (and the default `Mode.run_thinking` fallback for
> any audio-only turn).** Audio-in, text-out. Used by `RolePlayMode` to
> drive in-character NPC replies (shopkeeper, doctor, …) and by any other
> mode that needs a free-form spoken-question answer. Supports optional
> NDJSON streaming via `stream=true` so the UI can start playing audio
> while Gemma is still decoding.

`Content-Type: multipart/form-data`

| Field         | Type  | Required | Notes                                                                                |
| ------------- | ----- | -------- | ------------------------------------------------------------------------------------ |
| `audio`       | file  | yes      | Up to 25 MiB; transcoded to mono 16 kHz WAV.                                         |
| `max_tokens`  | int   | no       | 1–4096. Defaults to model config.                                                    |
| `temperature` | float | no       | 0.0–2.0. Defaults to model config.                                                   |
| `thinking`    | bool  | no       | If `true`, preserve and expose chain-of-thought. **Default off** on Pi 5.            |
| `stream`      | bool  | no       | If `true`, returns `application/x-ndjson` (`{"type","content"}` lines with periodic `{"type":"heartbeat"}` markers). |
| `tts`         | bool  | no       | If `true`, also render the answer via Piper TTS.                                     |
| `voice`       | str   | no       | Piper personality id. Defaults to `warm-academic`.                                   |

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

Sample response (non-streaming):

```json
{
  "text": "Sure — what would you like to buy today?",
  "model": "E4B",
  "inference_time_ms": 5210.55,
  "usage": {"prompt_tokens": 412, "completion_tokens": 22, "total_tokens": 434, "tokens_predicted": 22},
  "audio_url": "/tts/output/55ab2.wav",
  "audio_duration_ms": 2100.0,
  "voice": "warm-academic"
}
```

Streaming NDJSON lines (one per line):

```json
{"type": "text",      "content": "Sure — "}
{"type": "text",      "content": "what would you like to buy today?"}
{"type": "heartbeat", "elapsed_ms": 5000}
{"type": "audio",     "seq": 1, "sentence": "Sure — what would you like to buy today?", "content": "<base64-wav>", "mime": "audio/wav", "sample_rate": 22050, "voice": "warm-academic"}
{"type": "error",     "content": "..."}
```

- `heartbeat` lines are emitted every 5 s so the HTTP connection (and the
  learner's hope) survive multi-minute multimodal preprocessing on Pi 5.
- `audio` lines are only emitted when `tts=true` — one per completed
  sentence, interleaved with `text` chunks.

Pi 5 latency ballpark: ~4–8 s for a 5–10 s spoken question.

Returns `409 Conflict` if the active model config has `modalities.audio`
disabled.

---

## TTS audio attach

Every mode endpoint above accepts two opt-in fields:

| Field   | Type   | Default          | Notes                                                                 |
| ------- | ------ | ---------------- | --------------------------------------------------------------------- |
| `tts`   | bool   | `false`          | If `true`, also render the response text via Piper TTS.               |
| `voice` | string | `warm-academic`  | Curated personality id. Five personalities ship in `app/tts/voices.py`: `warm-academic` (default), `friendly-casual`, `neutral-news`, `energetic-kid`, `calm-storyteller`. |

When TTS is on, every endpoint above returns a **file-URL** attach shape
on the response JSON:

```json
{"audio_url": "/tts/output/<id>.wav", "audio_duration_ms": 7200.0, "voice": "warm-academic"}
```

The device fetches the WAV from `audio_url` (relative to the api container)
and plays it back. Files are evicted after `TTS_OUTPUT_TTL_SECONDS`
(default `600`).

`/audio/listen` with `stream=true` instead interleaves per-sentence
`{"type":"audio", ...}` NDJSON lines (base64 WAV) into the token stream,
so audio playback can begin as soon as the first sentence ends.

If the engine is disabled (`TTS_ENABLED=false`), the binary is missing,
or no voices were downloaded, every response gains an
`audio_error: "<reason>"` field instead of audio so callers can degrade
gracefully without branching on engine state.

---

## Notes & gotchas

- **Modality order matters.** Per the Gemma 4 docs, image and audio parts
  must precede the text part in the user message. The server enforces this
  in `build_user_content` ([`app/media.py`](../app/media.py)).
- **Audio formats**: clients can send anything `ffmpeg` understands; the
  server normalises to mono 16 kHz WAV before forwarding to Gemma.
- **Switching models**: set `MODEL_CONFIG_PATH` (and the matching
  `MODEL_REPO`/`MODEL_FILE`/`MMPROJ_*`) in `.env`. The vision / audio
  endpoints return `409` if the active config has the corresponding
  modality disabled.
- **Upload caps** (defined in [`app/media_multipart.py`](../app/media_multipart.py)):
  10 MiB images, 25 MiB audio. Adjust there if your hardware budget allows.
