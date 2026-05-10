# Phase 6 PRD: Gemma-Service Piper TTS Output

## Goal

Use the existing Piper TTS support in the Gemma service at `http://localhost:8010` so a full wake-word turn produces both tutor text and spoken WAV output. After `LISTENING`, the FSM should stay in `THINKING` while `/audio/listen` processes the recorded question and renders TTS. Only after both text and audio are ready should the app enter `SPEAKING`, play the WAV, then return to `IDLE`.

## Gemma Service Contract

No `gemma-llama` code changes are required. The learning app should use the existing non-streaming audio endpoint:

- `POST /audio/listen`
- `Content-Type: multipart/form-data`
- required file field: `audio`
- form fields:
  - `stream=false`
  - `tts=true`
  - `voice=<configured voice>`, default `warm-academic`
- expected JSON fields:
  - `text`
  - `audio_url`
  - `audio_duration_ms`
  - `voice`
  - optional `audio_error`

Resolve `audio_url` against `GEMMA_URL`, then fetch the WAV bytes with `GET <audio_url>`. Treat `audio_error` or a missing `audio_url` as a failed turn and cancel back to `IDLE`.

`POST /tts/speak` is not needed for the main wake-word flow because `/audio/listen` can already return the answer text and Piper output in one request. It can remain a future utility for arbitrary text-only speech.

## Functional Requirements

- Preserve FSM ownership in `learning-app/app/orchestrator/fsm.py`.
- Keep the transition shape:
  - `IDLE + WAKE_DETECTED -> LISTENING`
  - `LISTENING + UTTERANCE_END -> THINKING`
  - `THINKING + REPLY_READY -> SPEAKING`
  - `SPEAKING + PLAYBACK_DONE -> LISTENING` (so the user can ask a follow-up without re-saying "Widushi"; the silence timeout in `LISTENING` falls back to `IDLE` via `CANCEL` if no follow-up arrives)
- In the THINKING side effect:
  - use audio payloads with a TTS-aware Gemma client method;
  - call `/audio/listen` with `tts=true`;
  - fetch the returned WAV before posting `REPLY_READY`;
  - post `REPLY_READY` with `text`, `audio_bytes`, `audio_duration_ms`, and `voice`.
- Preserve a text prompt fallback for manual/API tests by calling `/generate` with `tts=true` and fetching its returned `audio_url`.
- In the SPEAKING side effect:
  - do not synthesize text locally;
  - play the prepared `audio_bytes`;
  - emit `PLAYBACK_DONE` only after playback finishes.
- On Gemma, TTS, audio fetch, or playback failures, log the exception and emit `CANCEL`.

## Configuration

Add environment-backed settings in `learning-app/app/config.py`:

- `GEMMA_TTS_ENABLED`, default `true`
- `GEMMA_TTS_VOICE`, default `warm-academic`

Existing settings still apply:

- `GEMMA_URL`, default `http://localhost:8010`
- `GEMMA_TIMEOUT_S`, default `120`

## Key Implementation Notes

- Add a small reply data object in `learning-app/app/services/gemma.py`, for example:

```python
GemmaReply(
    text: str,
    audio_bytes: bytes | None,
    audio_duration_ms: float | None,
    voice: str | None,
)
```

- Add TTS-aware Gemma methods:
  - `listen_audio_with_tts(audio_bytes, filename, content_type) -> GemmaReply`
  - `complete_with_tts(prompt) -> GemmaReply`
- Keep older text-only methods if existing tests or debug tooling still use them.
- Implement URL resolution with a structured URL helper such as `urllib.parse.urljoin`.
- Keep `learning-app/app/services/piper.py` as the SPEAKING playback facade, but make it play already-rendered WAV bytes with `pygame.mixer.Sound(file=io.BytesIO(wav_bytes))`.
- Stub services should return valid WAV bytes and use deterministic short waits so unit tests do not require Gemma, Piper, or speakers.
- Stop active playback during cancellation or application shutdown when possible.

## Tests

Add focused tests that do not require a running Gemma service:

- Gemma client posts multipart audio to `/audio/listen` with `stream=false`, `tts=true`, and the configured `voice`.
- Gemma client resolves and fetches a relative `audio_url`.
- Gemma client raises on `audio_error` or missing `audio_url`.
- Text fallback posts `/generate` with `tts=true` and fetches its audio URL.
- FSM audio path uses the TTS-aware Gemma method and emits `REPLY_READY` containing both text and WAV bytes.
- SPEAKING passes existing WAV bytes to the playback facade and emits `PLAYBACK_DONE` after playback returns.
- TTS or playback failure emits `CANCEL` and returns the FSM to `IDLE`.

## Validation

From the workspace root:

```bash
make -C learning-app lint
make -C learning-app test
```

Manual smoke test:

1. Start the Gemma service and confirm `/health` reports `tts_ready=true`.
2. Start the learning app.
3. Say "Widushi".
4. Ask a short spoken question.
5. Confirm the state sequence is `LISTENING -> THINKING -> SPEAKING -> LISTENING`, then either a follow-up turn or a `LISTENING -> IDLE` silence-timeout fallback.
6. Confirm THINKING covers LLM plus Piper generation time, and SPEAKING covers actual audio playback time.

## Non-Goals

- Do not implement local Piper subprocess synthesis in the learning app.
- Do not switch to streaming NDJSON TTS in this phase.
- Do not modify `gemma-llama` unless the deployed service lacks the documented `/audio/listen` TTS fields.
