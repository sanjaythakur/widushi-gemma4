# Phase 5 PRD: Gemma Audio Integration

## Goal

Connect the learning app to the local Gemma service so a full wake-word turn can answer a student's spoken question. After `WAKE_DETECTED`, the app should enter `LISTENING`, record the user's audio query, submit it to Gemma's native audio endpoint, and use the returned text as the tutor reply.

## Context

The Gemma service runs separately at `http://localhost:8010`. This phase uses:

- `POST /audio/listen`
- `Content-Type: multipart/form-data`
- required form file field: `audio`
- optional form field: `stream=false`
- expected JSON response field: `text`

Do not request Gemma/Piper TTS in this phase. The response should remain text-only and flow through the existing app-side `REPLY_READY -> SPEAKING` path. Real local speech output is handled in a later phase.

## Functional Requirements

- Preserve the existing FSM ownership model in `learning-app/app/orchestrator/fsm.py`.
- Keep the transition shape:
  - `IDLE + WAKE_DETECTED -> LISTENING`
  - `LISTENING + UTTERANCE_END -> THINKING`
  - `THINKING + REPLY_READY -> SPEAKING`
  - `SPEAKING + PLAYBACK_DONE -> LISTENING`
  - `LISTENING + CANCEL -> IDLE` (wildcard `CANCEL`; emitted by this phase when the silence timeout fires)
- Use one microphone owner for both wake-word detection and spoken-question recording.
- When OpenWakeWord detects "Widushi", emit `WAKE_DETECTED`, then continue consuming microphone frames as the user's query.
- After `PLAYBACK_DONE` returns the FSM to `LISTENING`, immediately reopen the microphone capture path and start a new recording attempt without waiting for another wake-word detection. The follow-up listen uses the same recording strategy and the same silence timeout as a post-wake listen.
- Record mono `int16` PCM at `16 kHz` and wrap it in a WAV container before sending to Gemma.
- End recording with a hybrid strategy:
  - automatically after trailing silence once speech has been heard;
  - automatically after a max recording duration;
  - cancel back to `IDLE` after a silence timeout if no speech is heard within `LISTENING_SILENCE_TIMEOUT_S` of entering `LISTENING` (this applies both to a post-wake listen and to a post-playback follow-up listen);
  - manually when ENTER or local API `UTTERANCE_END` requests a stop.
- If no voice is heard before `LISTENING_SILENCE_TIMEOUT_S`, emit `CANCEL` with `cancel_reason="no_voice_detected"` and do not call Gemma. The wildcard `CANCEL -> IDLE` handler in the orchestrator does the actual state drop, so this phase does not add a new `EventType`.
- Emit `UTTERANCE_END` with an audio payload:

```python
{
    "audio_bytes": wav_bytes,
    "filename": "question.wav",
    "content_type": "audio/wav",
    "sample_rate": 16000,
}
```

- In the THINKING side effect, prefer `audio_bytes` and call `GemmaClient.listen_audio(...)`.
- Preserve a text `prompt` fallback for tests and manual/debug API events.
- On Gemma failures, log the exception and cancel the turn back to `IDLE`.

## Configuration

Add environment-backed settings in `learning-app/app/config.py`:

- `GEMMA_URL`, default `http://localhost:8010`
- `GEMMA_TIMEOUT_S`, default `120`
- `LISTENING_SILENCE_RMS_THRESHOLD`, default `500`
- `LISTENING_TRAILING_SILENCE_S`, default `1.0`
- `LISTENING_MIN_RECORDING_S`, default `0.8`
- `LISTENING_MAX_RECORDING_S`, default `15.0`
- `LISTENING_SILENCE_TIMEOUT_S`, default `5.0` (applies on every `LISTENING` entry: after `WAKE_DETECTED` and after `PLAYBACK_DONE`)

## Implementation Notes

- Add a small shared manual-stop signal used by keyboard/API controls to stop the active recording without directly mutating FSM state.
- Implement WAV packaging with Python's standard `wave` module.
- Implement `GemmaClient.listen_audio(audio_bytes, filename, content_type)` using `httpx.AsyncClient`:

```python
await client.post(
    "/audio/listen",
    data={"stream": "false"},
    files={"audio": (filename, audio_bytes, content_type)},
)
```

- Runtime wiring should use live services configured from `GEMMA_URL`.
- Test wiring should keep `Services.stubs()` so tests do not require the Gemma service, audio hardware, or model files.

## Key Files

- `learning-app/app/input/wakeword.py` - owns wake detection and post-wake utterance recording.
- `learning-app/app/input/utterance.py` - WAV packaging and manual recording stop signal.
- `learning-app/app/services/gemma.py` - Gemma HTTP client and `/audio/listen` multipart upload.
- `learning-app/app/orchestrator/fsm.py` - THINKING side effect routes audio payloads to Gemma.
- `learning-app/app/input/keyboard.py` - ENTER requests active recording stop.
- `learning-app/app/api/server.py` - API `UTTERANCE_END` requests active recording stop.
- `learning-app/app/main.py` - wires live services and the shared stop signal.

## Testing Requirements

Add focused tests that do not require real hardware or a running Gemma service:

- Wake-word detection still emits `WAKE_DETECTED`.
- After wake detection, PCM frames are recorded and emitted as WAV bytes on `UTTERANCE_END`.
- Silence after entering `LISTENING` (both after wake detection and after `PLAYBACK_DONE`) emits `CANCEL` with `cancel_reason="no_voice_detected"` once `LISTENING_SILENCE_TIMEOUT_S` elapses, so the FSM returns to `IDLE`.
- Manual stop completes the active recording.
- `GemmaClient.listen_audio(...)` posts multipart audio to `/audio/listen` and extracts response `text`.
- The FSM audio path calls `listen_audio(...)` and emits `REPLY_READY` with the returned text.
- The old text-prompt fallback continues to work.

## Validation

From the workspace root:

```bash
make -C learning-app lint
make -C learning-app test
```

Expected result:

- Ruff passes for `app` and `tests`.
- Pytest passes all FSM, wake-word, and Gemma-client tests.

## Manual Smoke Test

1. Start the Gemma service and confirm `http://localhost:8010/docs#/tutor/listen_audio_listen_post` is reachable.
2. Start the learning app.
3. Say "Widushi".
4. Ask a short spoken question.
5. Confirm the app transitions through `LISTENING -> THINKING -> SPEAKING -> LISTENING` (the `SPEAKING -> LISTENING` loopback opens a follow-up turn without re-saying the wake word).
6. Confirm the text reply comes from Gemma's `/audio/listen` response.
7. Without saying anything during the follow-up `LISTENING`, confirm the app falls back to `IDLE` after `LISTENING_SILENCE_TIMEOUT_S` and now requires the wake word again.
8. Repeat with no speech after the wake word and confirm the app returns from `LISTENING` to `IDLE` without sending an audio request to Gemma.

For headless/manual fallback, trigger wake and stop events through the local app controls, but the production path should use real microphone audio after wake detection.

## Non-Goals

- Do not implement Gemma/Piper TTS output in this phase.
- Do not stream partial Gemma responses.
- Do not add vision, video, translation, or transcription routes.
- Do not let input sources mutate FSM state directly.
