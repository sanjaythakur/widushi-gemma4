# Phase 5 PRD: Gemma Audio Integration

## Goal

Connect the learning app to the local Gemma service so a full wake-word
turn can answer a student's spoken question. After `WAKE_DETECTED`, the
app records the user's audio query, submits it to Gemma's native audio
endpoint, and uses the returned text as the tutor reply.

This phase is text-only (no Piper output yet). Phase 6 extends the same
call to also fetch a Piper WAV in one round-trip.

## Gemma Service Contract

The Gemma service runs separately at `http://localhost:8010`:

- `POST /audio/listen`
- `Content-Type: multipart/form-data`
- required file field: `audio`
- form fields: `stream=false`
- expected JSON field: `text`

The text fallback path uses `POST /generate` with `{"prompt": ..., "stream": false}`
and the same `text` response field.

## Functional Requirements

- Preserve the FSM ownership model and transition table from phase 3.
- One mic owner: the wake-word source from phase 4 also drives recording
  after detection. This avoids two callers competing for sounddevice.
- After `WAKE_DETECTED`, the source continues consuming mic frames as
  the user's query.
- After `PLAYBACK_DONE` returns the FSM to `LISTENING`, the
  `_followup_listen_side_effect` raises a follow-up signal which the
  wake-word source picks up on the next mic chunk and starts recording
  immediately — no second wake word required. Follow-up
  `UTTERANCE_END` events tag `payload["followup"] = True`.
- Recording is mono `int16` PCM at 16 kHz, wrapped in a WAV container
  with the standard library `wave` module (`pcm16_mono_to_wav`).
- Recording end is a hybrid policy in `_record_utterance`:
  - **trailing-silence end**: stop after
    `LISTENING_TRAILING_SILENCE_S` of consecutive chunks below
    `LISTENING_SILENCE_RMS_THRESHOLD`, but only after at least
    `LISTENING_MIN_RECORDING_S` and once `voiced_seen=True`;
  - **max-duration end**: stop after `LISTENING_MAX_RECORDING_S`;
  - **no-voice cancel**: if no voice has been observed within
    `LISTENING_SILENCE_TIMEOUT_S` (applied on every `LISTENING` entry —
    both after `WAKE_DETECTED` and after `PLAYBACK_DONE`), emit
    `Event(EventType.CANCEL, payload={"cancel_reason": "no_voice_detected"})`
    and skip Gemma. The wildcard `CANCEL` handler in the orchestrator
    drops the FSM to `IDLE`;
  - **manual end**: ENTER on the keyboard or a `POST /events
    {"type":"UTTERANCE_END"}` request hits the shared
    `UtteranceStopSignal` and stops the recorder cleanly.
- **Voice-onset filter (critical for noisy rooms).** `voiced_seen` only
  flips `True` after `LISTENING_VOICE_ONSET_FRAMES` consecutive
  above-threshold chunks. A single 80 ms blip (fan, chair creak, the
  tail of `listen_start` echoing back through the speaker — phase 7) on
  its own must not flip `voiced_seen` and disarm the no-voice timeout.
  The current voiced-run counter resets on any silent chunk.
- Successful `UTTERANCE_END` payload:

  ```python
  {
      "audio_bytes": wav_bytes,
      "filename": "question.wav",
      "content_type": "audio/wav",
      "sample_rate": 16000,
      "followup": False | True,
  }
  ```

- The `THINKING` side effect prefers `audio_bytes` and calls
  `GemmaClient.listen_audio(...)`. A text `prompt` fallback remains for
  manual/API events.
- On Gemma failure, log the exception and post `CANCEL` so the FSM
  returns to `IDLE`.

## Configuration

Env-backed in `learning-app/app/config.py`:

- `GEMMA_URL`, default `http://localhost:8010`
- `GEMMA_TIMEOUT_S`, default `120`
- `LISTENING_SILENCE_RMS_THRESHOLD`, default `800`
  (mic-specific — see phase 8 for the calibration recipe)
- `LISTENING_VOICE_ONSET_FRAMES`, default `2` (set to `3` for very
  noisy rooms)
- `LISTENING_TRAILING_SILENCE_S`, default `1.5`
- `LISTENING_MIN_RECORDING_S`, default `0.8`
- `LISTENING_MAX_RECORDING_S`, default `15.0`
- `LISTENING_SILENCE_TIMEOUT_S`, default `5.0`
- `WIDUSHI_RMS_DEBUG`, default `false` — when set, the recorder logs
  `rms[NNN] t=…s rms=… thr=… silent|voiced run=… seen=…` per chunk at
  INFO. Used once after a mic / room change to pick a new
  `LISTENING_SILENCE_RMS_THRESHOLD`. Per-chunk logging is intentionally
  noisy; do **not** leave it on in production.

## Implementation Notes

- `learning-app/app/input/utterance.py` provides:
  - `UtteranceStopSignal`: `activate()`, `deactivate()`, `request_stop()`,
    `requested` property. The keyboard / API source calls
    `request_stop()`; `_record_utterance` polls `requested` per chunk.
  - `FollowUpListenSignal`: `request()` / `consume()` so the FSM's
    `SPEAKING → LISTENING` side effect can hand the wake-word source a
    one-shot bypass on the next mic chunk.
  - `pcm16_mono_to_wav(pcm: bytes, sample_rate: int) -> bytes`.
- `GemmaClient.listen_audio(audio_bytes, filename, content_type) -> str`
  uses `httpx.AsyncClient`:

  ```python
  await client.post(
      "/audio/listen",
      data={"stream": "false"},
      files={"audio": (filename, audio_bytes, content_type)},
  )
  ```

- `Services.live()` constructs `GemmaClient(url=GEMMA_URL,
  timeout_s=GEMMA_TIMEOUT_S)`. `Services.stubs()` keeps stubbed clients
  so unit tests don't need Gemma, audio hardware, or model files.

## Key Files

- `learning-app/app/input/wakeword.py` — owns wake detection and the
  post-wake `_record_utterance` (see phase 4 for the FSM-aware gate
  that lets follow-ups through but blocks phantom recordings).
- `learning-app/app/input/utterance.py` — WAV helper + signals.
- `learning-app/app/services/gemma.py` — HTTP client.
- `learning-app/app/orchestrator/fsm.py` — `_thinking_side_effect`
  routes audio payloads to Gemma and posts `REPLY_READY`.
- `learning-app/app/input/keyboard.py` — ENTER → request stop.
- `learning-app/app/api/server.py` — `POST /events
  {"type":"UTTERANCE_END"}` → request stop or enqueue.
- `learning-app/app/main.py` — wires `Services.live()`,
  `UtteranceStopSignal`, and `FollowUpListenSignal`.

## Tests

`learning-app/tests/test_gemma.py` and additions to
`learning-app/tests/test_wakeword.py`. All run without real audio,
Gemma, or model files:

- Wake → recording → `UTTERANCE_END` with WAV-shaped `audio_bytes`,
  `filename`, `content_type`, `sample_rate`.
- Initial silence after wake (within `LISTENING_SILENCE_TIMEOUT_S`)
  emits `CANCEL` with `cancel_reason="no_voice_detected"`.
- End-to-end through a real `Orchestrator`: `IDLE → LISTENING → IDLE`
  via the silence cancel.
- Follow-up loopback end-to-end:
  `IDLE → LISTENING → THINKING → SPEAKING → LISTENING → IDLE` driven by
  `FollowUpListenSignal` + silent recorder.
- Voice-onset filter: a single isolated above-threshold blip surrounded
  by silence must **not** flip `voiced_seen`; the no-voice timeout still
  fires `CANCEL`.
- Manual stop via `UtteranceStopSignal.request_stop()` finishes the
  active recording and emits `UTTERANCE_END`.
- `GemmaClient.listen_audio` posts multipart audio to `/audio/listen`
  and returns the response `text`.
- Text-prompt fallback still calls `/generate` and returns `text`.

## Validation

```bash
make -C learning-app lint
make -C learning-app test
```

Manual smoke test:

1. Start the Gemma service; confirm `http://localhost:8010/docs` is up.
2. `make -C learning-app run`.
3. Say "Widushi", ask a short spoken question. Confirm the FSM walks
   `LISTENING → THINKING → SPEAKING → LISTENING` and the reply text
   matches Gemma's response.
4. Say nothing during the follow-up `LISTENING`; confirm the FSM falls
   back to `IDLE` after `LISTENING_SILENCE_TIMEOUT_S`.
5. If recordings are landing at exactly `LISTENING_MAX_RECORDING_S`,
   the silence threshold is too low for the room — see phase 8 for the
   `WIDUSHI_RMS_DEBUG=1` calibration recipe.

## Non-Goals

- No Piper / TTS output (phase 6).
- No streaming partial Gemma responses.
- No vision, video, or transcription routes.
- Input sources still must not mutate FSM state directly.
