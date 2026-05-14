# Phase 4 PRD: OpenWakeWord Integration

## Goal

Integrate the custom OpenWakeWord model for "Widushi" into the learning
app so that, while the FSM is in `IDLE`, hearing the wake word emits
`WAKE_DETECTED` and moves the FSM to `LISTENING`. While the FSM is in
any non-IDLE state, wake-word detection must be paused.

## Context

The phase 3 FSM accepts `(IDLE, WAKE_DETECTED) → LISTENING` and the
wildcard `CANCEL → IDLE` already returns to `IDLE`. This phase only
implements the wake-word input source and microphone capture path; it
must not mutate FSM state directly.

Speech-to-text and `UTTERANCE_END` are owned by phase 5.

## Functional Requirements

- Add a `WakeWordSource` in `learning-app/app/input/wakeword.py` as an
  `InputSource` subclass started by `app/main.py` alongside the keyboard
  source.
- Load the custom ONNX wake-word model from `WAKE_WORD_MODEL`, defaulting
  to `<workspace>/openwakeword/vDu_shee.onnx` on host runs and
  `/openwakeword/vDu_shee.onnx` in the container image.
- Keep wake-word detection enabled by default. Headless dev opts out
  with `WAKE_WORD_ENABLED=0`.
- Read mic audio as mono `int16` PCM, feed OpenWakeWord `1280`-sample
  frames (80 ms at 16 kHz). Resample if the mic isn't 16 kHz.
- Run OpenWakeWord inference off the event loop with
  `asyncio.to_thread(...)`.
- When the best score crosses `WAKE_WORD_THRESHOLD`, emit:

  ```python
  Event(EventType.WAKE_DETECTED,
        payload={"source": "openwakeword", "label": label, "score": score})
  ```

- Suppress repeated detections inside `WAKE_WORD_DEBOUNCE_S`.
- **FSM-aware gating (critical).** The source accepts an optional
  `state_getter: Callable[[], AppState]`. When provided and the current
  state is anything other than `AppState.IDLE`, the source must:
  - drop the incoming mic chunk,
  - clear the wake-word frame buffer,
  - skip detector inference, and
  - log once per state-change at DEBUG (no per-chunk spam).
  Without this gate, ambient noise or the user's continued talking during
  `THINKING`/`SPEAKING` re-fires `WAKE_DETECTED` (silently dropped by
  the FSM table) **and** also kicks off a phantom 15 s recording in
  phase 5's recorder whose `UTTERANCE_END` is also dropped — wasting
  Gemma latency and confusing the user. The follow-up signal check
  (phase 5) runs **before** the gate so post-`PLAYBACK_DONE` follow-up
  listens still work.
- If detection is disabled or the model file is missing, log clearly and
  park the source task forever without crashing the app.

## Configuration

Env-backed in `learning-app/app/config.py`:

- `WAKE_WORD_MODEL`: path to `vDu_shee.onnx`. Default
  `<workspace>/openwakeword/vDu_shee.onnx`.
- `WAKE_WORD_ENABLED`: bool opt-out. Default `true`.
- `WAKE_WORD_THRESHOLD`: detection threshold. Default `0.5`.
- `WAKE_WORD_DEBOUNCE_S`: post-detection debounce window. Default `2.0`.
- `WAKE_WORD_FRAME_MS`: mic chunk duration. Default `80`.

## Audio Capture

`MicSource` in `learning-app/app/hardware/mic.py`:

```python
async def frames(self) -> AsyncIterator[bytes]: ...
def drain(self) -> int: ...        # used by phase 7's clip echo cleanup
```

- `stub=True`: yield silence forever (used in tests and when
  `WAKE_WORD_ENABLED=0`).
- `stub=False`: capture mono `int16` PCM via `sounddevice.RawInputStream`,
  default `sample_rate=16000`, `chunk_ms=80`.
- Bounded queue between the sounddevice callback thread and the asyncio
  loop. **Drop** old audio if the queue is full — never block the
  real-time audio callback.
- `MIC_INPUT_DEVICE` from `WIDUSHI_INPUT_DEVICE` env var (substring of
  device name, integer index, or unset = ALSA `default`).

## Wake-Word Source Design

- `WakeWordDetector` protocol with
  `predict(pcm16: bytes) -> Mapping[str, float]`.
- `OpenWakeWordDetector` wrapper that imports `openwakeword.model.Model`
  lazily and constructs:

  ```python
  Model(wakeword_models=[str(model_path)], inference_framework="onnx")
  ```

- Dependency injection for `mic`, `detector_factory`, `clock`, and
  `state_getter` so tests run without OpenWakeWord, audio hardware, real
  model files, or a real orchestrator.
- Buffer incoming bytes until `1280 * 2` are available, then process one
  frame at a time.

## Dependencies

`learning-app/pyproject.toml` runtime: `numpy`, `onnxruntime`,
`openwakeword`, `sounddevice`. Container OS: `libasound2`,
`libportaudio2`. Build the image from the workspace root and copy
`openwakeword/` to `/openwakeword`; set
`WAKE_WORD_MODEL=/openwakeword/vDu_shee.onnx` in the image. The model
files do not need to move into `learning-app`.

The Dockerfile must also pre-fetch openWakeWord's shared feature models
during build (`from openwakeword.utils import download_models;
download_models(model_names=['vDu_shee'])`) so the Pi can start
detection without first-boot network access.

## Tests

`learning-app/tests/test_wakeword.py`. Use a `FakeMic` (yields
pre-supplied `bytes` chunks) and a `FakeDetector` (returns a queued list
of `Mapping[str, float]`) so no real audio hardware or model is needed:

- Above-threshold score emits exactly one `WAKE_DETECTED` with payload
  `{"source": "openwakeword", "label": ..., "score": ...}`.
- Repeated high scores inside `WAKE_WORD_DEBOUNCE_S` do not flood the
  queue.
- **FSM gate (added in this phase).** With
  `state_getter=lambda: AppState.THINKING`, no `WAKE_DETECTED` is
  emitted and the detector is never invoked, even with all-high scores.
- **FSM gate resumes.** With a `state_getter` whose first call returns a
  non-IDLE state and subsequent calls return `IDLE`, the detector is
  only invoked on chunks observed while state was `IDLE`.

(Recording / silence-timeout tests belong in phase 5.)

## Validation

```bash
make -C learning-app lint
make -C learning-app test
```

Manual smoke test on the host:

```bash
make -C learning-app run
# say "Widushi"
curl localhost:8020/state    # → "LISTENING"
```

For headless dev / CI: `WAKE_WORD_ENABLED=0 make run`.

## Non-Goals

- No speech-to-text, recording, or `UTTERANCE_END` emission (phase 5).
- No FSM transition table changes beyond what phase 3 documented.
- Wake-word detection must remain optional (`WAKE_WORD_ENABLED=0`) for
  headless dev / tests.
- Do not move generated wake-word model files into `learning-app`.
