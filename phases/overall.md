# Widushi — Project Brief

Widushi is an interactive learning tutor that runs on a Raspberry Pi 5 with
a 3.5-inch SPI TFT (480×320, ILI9486 / fbtft), a USB mic, and a USB
speaker. A separate locally-hosted Gemma service (`gemma-llama`) at port
`8010` answers spoken questions and renders Piper TTS replies.

The student says **"Widushi"**, asks a question, and the tutor speaks the
answer. The on-screen face animates per state (`IDLE`, `LISTENING`,
`THINKING`, `SPEAKING`).

## Subprojects

The repo holds independent subprojects in sibling folders. Each one is
buildable on its own; the learning app is the runtime that ties them
together.

| Folder | Purpose | PRD |
|---|---|---|
| `gemma-llama/` | Local Gemma + Piper service (port 8010, `/audio/listen` etc.) | `gemma-llama/specification.md` (linked from [phase 1](1-local-llama-gemma.md)) |
| `pre-generated-clips/` | Offline TTS pipeline that builds the `.wav` ack/status library | [phase 2](2-pre-generated-clips.md) |
| `openwakeword/` | Trained "Widushi" wake-word model (`vDu_shee.onnx`) | (out of scope; produced by upstream tooling) |
| `learning-app/` | Pi-side runtime: pygame UI + asyncio FSM + FastAPI + wake word + Gemma client + clip + TTS playback | [phases 3–8](.) |

## Stack

- Python 3.11
- `pygame` for UI on the SPI TFT (rendered via custom framebuffer sink)
- `asyncio` orchestrator + `FastAPI`/`uvicorn` for the local control plane,
  all on a single event loop on the main thread
- `httpx` async client for Gemma
- `openwakeword` + `onnxruntime` + `sounddevice` for wake-word detection
  and mic capture
- `aiosqlite` for any persistence (kept as a stub through phase 8)
- Docker for both Mac dev (headless) and Pi deploy; same image for both

## Development flow

- Develop on a Mac (Apple Silicon).
- Test in a Mac-side Docker container that's binary-compatible with the Pi.
- Deploy by `scp`-ing the workspace to the Pi 5 and running the container
  there with the framebuffer + USB audio devices mapped in.

## PRDs in this folder

The PRDs are written so the codebase can be recreated from them, in order:

1. [`1-local-llama-gemma.md`](1-local-llama-gemma.md) — pointer to
   `gemma-llama/specification.md`.
2. [`2-pre-generated-clips.md`](2-pre-generated-clips.md) — offline ack /
   status WAV library.
3. [`3-system-scaffolding.md`](3-system-scaffolding.md) — process model,
   FSM, event taxonomy, services bundle, headless dev image.
4. [`4-open-wake-word-integration.md`](4-open-wake-word-integration.md) —
   wake-word source, mic capture, FSM-aware gating.
5. [`5-gemma-integration.md`](5-gemma-integration.md) — recording the
   spoken question and posting it to Gemma's `/audio/listen` (text reply).
6. [`6-piper-tts-output.md`](6-piper-tts-output.md) — extending the Gemma
   call with `tts=true` so the reply ships back as a Piper WAV.
7. [`7-pre-generated-clip-integration.md`](7-pre-generated-clip-integration.md)
   — `listen_start` / `wait_thinking` / `wait_checking` clip playback to
   mask Gemma latency.
8. [`8-pi-deploy.md`](8-pi-deploy.md) — Pi 5 deploy: framebuffer sink,
   USB audio routing, networking to the Gemma container, clip bind mount,
   RMS calibration recipe.
