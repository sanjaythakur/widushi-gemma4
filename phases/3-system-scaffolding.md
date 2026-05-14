# Phase 3 PRD: System Scaffolding

## Goal

Stand up the `learning-app/` runtime as a single-process pygame + asyncio
+ FastAPI app that walks the four-state FSM end-to-end against stub
services, runs headless in Docker, and is ready for the wake-word /
Gemma / clip / TTS integrations in later phases.

## Process Architecture

One Python process, one event loop, on the main thread:

- The pygame UI loop runs as a coroutine that ticks the active face,
  flushes the SDL window, and `await asyncio.sleep(1/FPS)` each frame so
  other tasks make progress. SDL requires the main thread, so this loop
  must own it.
- The orchestrator (FSM) runs as a coroutine consuming events from a
  single `asyncio.Queue[Event]`.
- Uvicorn / FastAPI runs as a coroutine on the same loop.
- Input sources (keyboard, FastAPI, future wake-word, future API) push
  `Event`s into the queue. They never mutate FSM state directly.
- The orchestrator is the **only** writer to `state`.

The Gemma service is a separate process (outside this codebase) at
`http://localhost:8010`.

## States and Events

```python
class AppState(Enum):
    IDLE; LISTENING; THINKING; SPEAKING

class EventType(Enum):
    WAKE_DETECTED; UTTERANCE_END; REPLY_READY; PLAYBACK_DONE
    CANCEL; SHUTDOWN
```

Transition table (built once in `Orchestrator._build_table`):

| from \ event | result |
|---|---|
| `IDLE` + `WAKE_DETECTED` | → `LISTENING` |
| `LISTENING` + `UTTERANCE_END` | → `THINKING` (side effect: call Gemma, post `REPLY_READY`) |
| `THINKING` + `REPLY_READY` | → `SPEAKING` (side effect: play WAV, post `PLAYBACK_DONE`) |
| `SPEAKING` + `PLAYBACK_DONE` | → `LISTENING` (side effect: arm follow-up listen) |
| `*` + `CANCEL` | → `IDLE` (wildcard; cancels in-flight side effects) |
| `*` + `SHUTDOWN` | orchestrator returns from its `run()` loop |

Any other `(state, event)` pair is silently ignored at DEBUG.

## Layer breakdown and key files

```
learning-app/app/
  main.py              entrypoint: pygame init + asyncio.run + TaskGroup-ish wiring
  config.py            screen size, fps, ports, paths, env-backed knobs
  state.py             AppState enum
  events.py            Event dataclass + EventType enum
  orchestrator/
    fsm.py             Orchestrator with dispatch table and side-effect tasks
  ui/
    loop.py            pygame loop coroutine (ticks face, flips display, pushes to FbSink)
    faces/             Idle / Listening / Thinking / Speaking face renderers
  services/
    __init__.py        Services dataclass: gemma, piper, clips, db; .live() and .stubs()
    gemma.py           httpx.AsyncClient wrapper (stub in phase 3)
    piper.py           pygame.mixer playback facade for already-rendered WAVs (stub in phase 3)
    clips.py           pygame.mixer playback facade for pre-rendered ack/status WAVs (stub in phase 3)
    db.py              aiosqlite stub (no persistence yet)
  hardware/
    audio.py           pygame.mixer.init helper honouring SDL_AUDIODEV / WIDUSHI_OUTPUT_DEVICE
    mic.py             MicSource (sounddevice in phase 4; stub here)
    fb_sink.py         /dev/fb0 RGB565 sink for the SPI TFT (added in phase 8)
  input/
    sources.py         InputSource base class with .emit(Event)
    keyboard.py        pygame KEYDOWN -> events (manual debug walk)
    utterance.py       UtteranceStopSignal, FollowUpListenSignal, pcm16_mono_to_wav helper
  api/
    server.py          FastAPI app + uvicorn config (POST /events, GET /state, GET /health)
```

## Functional Requirements

- `Orchestrator(queue, services, *, followup_listen=None)` consumes events
  in a `while True: event = await queue.get()` loop. `SHUTDOWN` exits.
  `CANCEL` cancels in-flight side-effect tasks and drops to `IDLE`.
- Side effects are spawned with `asyncio.create_task` so the FSM loop
  stays responsive. Cancelling a side effect must be safe.
- `add_listener(fn)` lets the UI / tests observe transitions without
  reaching into private state.
- The keyboard source maps:
  - `SPACE` → `WAKE_DETECTED`
  - `ENTER` → request stop on the active recording, or emit
    `UTTERANCE_END` if no recording is active
  - `R` → `REPLY_READY`, `D` → `PLAYBACK_DONE`
  - `ESC` → `CANCEL` (twice in a row → `SHUTDOWN`)
- The FastAPI app exposes:
  - `GET /state` → current FSM state name
  - `GET /health` → liveness
  - `POST /events` → enqueue an `Event` (debug / API tests)
- Window-close = graceful `SHUTDOWN`. SIGINT/SIGTERM also enqueue
  `SHUTDOWN` (use `loop.add_signal_handler` with a fallback on platforms
  that don't support it).
- Faces are pure pygame draw calls (no image assets), 480×320, with light
  per-state animation:
  - **IDLE**: half-closed eyes, blink every 3–5 s, subtle breathing scale.
  - **LISTENING**: wide eyes, slight head tilt, animated mic-amplitude bars (teal).
  - **THINKING**: raised brows, eyes upper-left, orbital dots (amber).
  - **SPEAKING**: mouth open/close on a sine, eyes forward (coral).

## Configuration (env-backed in `config.py`)

Phase 3 ships at minimum:

- Screen: `SCREEN_WIDTH=480`, `SCREEN_HEIGHT=320`, `FPS=30`.
- API: `WIDUSHI_API_HOST` (default `0.0.0.0`), `WIDUSHI_API_PORT`
  (default `8020`).
- Logging: `WIDUSHI_LOG` (default `INFO`).
- Forward-declared placeholders later phases own:
  `GEMMA_URL`, `WAKE_WORD_MODEL`, `CLIPS_DIR`, etc.

## Headless dev image

- `learning-app/docker/Dockerfile.app` builds from `python:3.11-slim`,
  installs the SDL2 runtime libs (`libsdl2-2.0-0`, `libsdl2-mixer`,
  `libsdl2-ttf`, `libsdl2-image`, `libfreetype6`, `fonts-dejavu-core`,
  `libasound2`, `libportaudio2`), pip-installs the app deps, copies
  `learning-app/app` into `/app/app`, exposes `8020`.
- Default env in the image: `SDL_VIDEODRIVER=dummy`,
  `SDL_AUDIODRIVER=dummy`, `WIDUSHI_API_HOST=0.0.0.0`,
  `WIDUSHI_API_PORT=8020`.
- `learning-app/docker/docker-compose.yml` runs the image headless on a
  Mac for local FastAPI smoke tests; build context is the workspace root
  so future phases can package siblings (`openwakeword/`,
  `pre-generated-clips/`) into the image.

## Tests

`learning-app/tests/test_fsm.py` (pytest + pytest-asyncio):

- A full happy-path walk:
  `IDLE → LISTENING → THINKING → SPEAKING → LISTENING` using stub services.
- `CANCEL` from each non-IDLE state drops to `IDLE` and cancels any
  in-flight side-effect task.
- Listeners registered via `add_listener` see every transition.
- Stub `Services.stubs()` returns immediately (no Gemma, no audio
  hardware, no real model files).

## Validation

```bash
make -C learning-app lint
make -C learning-app test
make -C learning-app run-docker      # FastAPI on :8020 in headless container
curl localhost:8020/state            # → "IDLE"
curl -X POST localhost:8020/events -H 'content-type: application/json' \
  -d '{"type":"WAKE_DETECTED"}'
curl localhost:8020/state            # → "LISTENING"
```

## Non-Goals

- No real wake-word, mic, Gemma, TTS, or clip wiring (later phases).
- No SQLite persistence on disk (the `Database` class is a stub).
- No camera/vision pipelines.
- No streaming responses.
