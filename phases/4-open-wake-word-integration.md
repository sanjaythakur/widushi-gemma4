# Phase 4 PRD: OpenWakeWord Integration

## Goal

Integrate the custom OpenWakeWord model for "Widushi" into the learning app so that, while the app is in `IDLE`, hearing the wake word emits `WAKE_DETECTED` and moves the FSM to `LISTENING`.

## Context

The system already has a central asyncio FSM in `learning-app/app/orchestrator/fsm.py`. That FSM owns state changes and supports the following transition shape:

- `IDLE + WAKE_DETECTED -> LISTENING`
- `LISTENING + UTTERANCE_END -> THINKING`
- `THINKING + REPLY_READY -> SPEAKING`
- `SPEAKING + PLAYBACK_DONE -> LISTENING`
- `LISTENING + CANCEL -> IDLE` (wildcard `CANCEL` already returns to `IDLE`; the wake-word / utterance source is responsible for emitting `CANCEL` after `LISTENING_SILENCE_TIMEOUT_S` of silence so a follow-up turn that nobody answers drops back to `IDLE` instead of holding the mic open)

The `SPEAKING -> LISTENING` loopback lets the user ask follow-up questions in the same conversation without re-saying "Widushi". The silence timeout from `LISTENING` back to `IDLE` is what bounds that loop and forces the wake word again once the conversation is genuinely over.

This phase only needs to implement the real wake-word input source and microphone capture path. It must not mutate FSM state directly.

## Functional Requirements

- Add a real `WakeWordSource` in `learning-app/app/input/wakeword.py`.
- Load the custom ONNX wake-word model from `WAKE_WORD_MODEL`, defaulting to `../openwakeword/vDu_shee.onnx` on host runs and `/openwakeword/vDu_shee.onnx` in the container image.
- Keep wake-word detection enabled by default. Headless development can opt out with `WAKE_WORD_ENABLED=0`.
- Keep model artifacts in the sibling `openwakeword/` directory. Do not move them into `learning-app` just for packaging.
- When enabled, read microphone audio as mono `int16` PCM.
- Feed OpenWakeWord `16 kHz`, `1280` sample frames, equivalent to `80 ms` of audio.
- If microphone input is not already `16 kHz`, resample before inference.
- Run OpenWakeWord inference without blocking the asyncio event loop.
- When the best model score is at or above `WAKE_WORD_THRESHOLD`, emit:

```python
Event(
    EventType.WAKE_DETECTED,
    payload={
        "source": "openwakeword",
        "label": label,
        "score": score,
    },
)
```

- Suppress repeated detections for `WAKE_WORD_DEBOUNCE_S` seconds after a positive detection.
- If wake-word detection is disabled or the model file is missing, log clearly and park the source task without crashing the app.

## Configuration

Add these environment-backed settings in `learning-app/app/config.py`:

- `WAKE_WORD_MODEL`: path to the custom ONNX model. Default: `<workspace>/openwakeword/vDu_shee.onnx`.
- `WAKE_WORD_ENABLED`: boolean opt-out flag. Default: enabled.
- `WAKE_WORD_THRESHOLD`: detection threshold. Default: `0.5`.
- `WAKE_WORD_DEBOUNCE_S`: repeated-detection debounce window. Default: `2.0`.
- `WAKE_WORD_FRAME_MS`: mic chunk duration. Default: `80`.

## Audio Capture

Implement `MicSource` in `learning-app/app/hardware/mic.py` with one async contract:

```python
async def frames(self) -> AsyncIterator[bytes]:
    ...
```

The source should support:

- `stub=True`: yield silence forever for tests, CI, and wake-word-disabled development.
- `stub=False`: use `sounddevice.RawInputStream` to capture mono `int16` PCM.
- `sample_rate=16000` and `chunk_ms=80` by default.
- A small bounded queue between the sounddevice callback thread and the asyncio loop.
- Dropping old audio if the queue is full, rather than blocking the real-time audio callback.

## Wake-Word Source Design

`WakeWordSource` should remain an `InputSource` subclass and be started by `learning-app/app/main.py` like the other input sources. The source should use real microphone capture when enabled and silence/stub capture only when explicitly disabled or injected by tests.

Recommended structure:

- `WakeWordDetector` protocol with `predict(pcm16: bytes) -> Mapping[str, float]`.
- `OpenWakeWordDetector` wrapper that imports `openwakeword.model.Model` lazily and constructs:

```python
Model(
    wakeword_models=[str(model_path)],
    inference_framework="onnx",
)
```

- Dependency injection for `mic`, `detector_factory`, and `clock` so tests can run without OpenWakeWord, NumPy model files, or audio hardware.
- Buffer incoming audio bytes until at least `1280 * 2` bytes are available, then process one frame at a time.
- Call detector inference via `asyncio.to_thread(...)`.

## Dependencies

Add runtime dependencies in `learning-app/pyproject.toml`:

- `numpy`
- `onnxruntime`
- `openwakeword`
- `sounddevice`

Add container OS dependencies in `learning-app/docker/Dockerfile.app`:

- `libasound2`
- `libportaudio2`

Build the image from the workspace root so the sibling `openwakeword/` directory is in the Docker context, then copy it into `/openwakeword`. Set `WAKE_WORD_MODEL=/openwakeword/vDu_shee.onnx` in the image. The model files do not need to move into `learning-app`.

## Testing Requirements

Add focused tests in `learning-app/tests/test_wakeword.py`:

- A fake mic and fake detector test proving a score above threshold emits exactly one `WAKE_DETECTED`.
- A debounce test proving repeated high scores inside the debounce window do not flood the queue.
- Tests must not require real audio hardware, OpenWakeWord inference, or a real model file.

Keep the existing FSM tests as the end-to-end state transition proof.

## Validation

From `learning-app`, run:

```bash
make lint
make test
```

Expected result:

- Ruff passes for `app` and `tests`.
- Pytest passes all FSM and wake-word tests.

## Manual Smoke Test

Run:

```bash
make run
```

Then speak "Widushi" and verify:

```bash
curl localhost:8020/state
```

The state should move from `IDLE` to `LISTENING` after the wake word is detected.

For headless runs without microphone access:

```bash
WAKE_WORD_ENABLED=0 make run
```

## Non-Goals

- Do not implement speech-to-text or `UTTERANCE_END` detection in this phase.
- Do not change the FSM transition table beyond what this PRD documents (the `SPEAKING -> LISTENING` loopback and the `LISTENING` silence timeout to `IDLE` via `CANCEL`); any further transition changes belong in their own PRD.
- Do not make wake-word detection mandatory for headless development or tests; allow `WAKE_WORD_ENABLED=0`.
- Do not move generated wake-word model files into `learning-app`; package them from the sibling `openwakeword/` directory.
