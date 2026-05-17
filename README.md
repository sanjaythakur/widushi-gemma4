# Widushi

Widushi is a local, voice-first learning tutor for a Raspberry Pi 5. A student
says "Widushi", asks a question, and the device answers out loud while a small
480x320 face UI moves through `IDLE`, `LISTENING`, `THINKING`, and `SPEAKING`
states.

The repo is split into independent subprojects so the LLM service, the Pi app,
and the offline audio assets can be built and tested separately.

## Subprojects

| Folder | Purpose | Start here |
| --- | --- | --- |
| [`gemma-llama/`](gemma-llama/) | Local Gemma + Piper service. Runs a two-container `llama.cpp` + FastAPI stack on port `8010`, with text, image, audio, video, and TTS endpoints. | [`gemma-llama/docs/README.md`](gemma-llama/docs/README.md) |
| [`learning-app/`](learning-app/) | Pi-side runtime. Runs the pygame face UI, asyncio FSM, FastAPI control plane, wake-word listener, Gemma client, and audio playback. | [`learning-app/README.md`](learning-app/README.md) |
| [`pre-generated-clips/`](pre-generated-clips/) | Offline TTS pipeline for short acknowledgement/status WAV clips such as `listen_start`, `wait_thinking`, and `wait_checking`. | [`pre-generated-clips/README.md`](pre-generated-clips/README.md) |
| [`phases/`](phases/) | Product brief and phased implementation notes. | [`phases/overall.md`](phases/overall.md) |

The trained openWakeWord model is expected as a sibling `openwakeword/` folder
when running the app on host or building the Pi image. See the learning app
README for the expected model path and override variables.

## How The Pieces Fit

```text
USB mic + wake word
        |
        v
learning-app (:8020 control API, pygame UI)
        |
        | POST /audio/listen tts=true
        v
gemma-llama (:8010 FastAPI)
        |
        v
llama.cpp + Gemma 4 + Piper TTS
```

`learning-app` records the student's question, sends it to `gemma-llama`, fetches
the generated Piper WAV, plays it locally, then returns to listening for a
follow-up. `pre-generated-clips` supplies short local cues that mask latency
while the app starts listening or waits for Gemma.

## Quickstart

### 1. Start Gemma

```bash
cd gemma-llama
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8010/` for the playground, or check
`http://localhost:8010/health`.

### 2. Generate Status Clips

```bash
cd pre-generated-clips
make clips-en
```

Hindi clips use the gated AI4Bharat model and require `HF_TOKEN`:

```bash
HF_TOKEN=<token> make clips-hi
```

### 3. Run The Learning App

```bash
cd learning-app
make install
make run
```

For headless development without microphone access:

```bash
WAKE_WORD_ENABLED=0 make run
```

The app expects Gemma at `http://localhost:8010` by default. Override with
`GEMMA_URL` if the service is elsewhere.

## Development And Testing

Each subproject owns its own workflow:

```bash
# Gemma service evaluation harness
cd gemma-llama
pip install -r requirements-eval.txt
python -m eval.runner --all
pytest eval/ -v -s

# Learning app checks
cd learning-app
make lint
make test

# Clip generation
cd pre-generated-clips
make clips
```

## Deployment Notes

The target runtime is a Raspberry Pi 5 with a 3.5-inch SPI TFT, USB microphone,
and USB speaker. The learning app image renders through a framebuffer sink and
bind-mounts the generated clips at runtime, while the Gemma stack runs
separately and exposes port `8010`.

For the full Pi command, audio routing, framebuffer setup, and RMS calibration
recipe, read [`learning-app/README.md`](learning-app/README.md) and
[`phases/8-pi-deploy.md`](phases/8-pi-deploy.md).

## Useful Docs

- [`phases/overall.md`](phases/overall.md) — product brief and phase index.
- [`gemma-llama/specification.md`](gemma-llama/specification.md) — source of truth for the Gemma service contracts.
- [`gemma-llama/docs/endpoints.md`](gemma-llama/docs/endpoints.md) — detailed HTTP endpoint reference.
- [`learning-app/README.md`](learning-app/README.md) — app architecture, wake-word flow, Gemma integration, and Pi deployment.
- [`pre-generated-clips/README.md`](pre-generated-clips/README.md) — clip engines and generation commands.
