# Phase 6 PRD: Gemma-Service Piper TTS Output

## Goal

Use the existing Piper TTS support in the Gemma service so a full
wake-word turn produces both tutor text and spoken WAV output. The FSM
should stay in `THINKING` while `/audio/listen` processes the recorded
question **and** renders TTS. Only after both text and audio are ready
should the app enter `SPEAKING`, play the WAV, then return to
`LISTENING` for the follow-up turn.

## Gemma Service Contract (no service changes required)

The phase-1 Gemma service already returns Piper output from the same
endpoint phase 5 uses:

- `POST /audio/listen`
- `Content-Type: multipart/form-data`
- required file field: `audio`
- form fields:
  - `stream=false`
  - `tts=true`
  - `voice=<configured voice>` (default `warm-academic`)
- expected JSON fields: `text`, `audio_url`, `audio_duration_ms`,
  `voice`, optional `audio_error`.

Resolve `audio_url` against `GEMMA_URL` (`urllib.parse.urljoin`), then
`GET <audio_url>` for the WAV bytes. Treat `audio_error` or a missing
`audio_url` as a failed turn and `CANCEL`.

`POST /tts/speak` exists as a future utility for arbitrary text-only
speech but is not used in the wake-word flow.

## Functional Requirements

- Preserve the FSM table from phase 3.
- In the `THINKING` side effect:
  - call the TTS-aware Gemma method with the recorded `audio_bytes`;
  - fetch the returned WAV before posting `REPLY_READY`;
  - post `REPLY_READY` with `{"text", "audio_bytes",
    "audio_duration_ms", "voice"}`.
- Text-prompt fallback (manual / API events) calls `POST /generate` with
  `tts=true` and fetches its `audio_url` the same way.
- In the `SPEAKING` side effect:
  - **do not** synthesize text locally;
  - play the prepared `audio_bytes`;
  - emit `PLAYBACK_DONE` only after playback finishes.
- On Gemma failure, TTS failure, audio fetch failure, or playback
  failure: log the exception and emit `CANCEL`.
- Phase 7 plays acknowledgement clips concurrently with the Gemma
  request; `_thinking_side_effect` must wait for **both** the Gemma
  reply and the in-flight clip task before posting `REPLY_READY` so the
  reply audio doesn't talk over the clip.

## Configuration

Env-backed in `learning-app/app/config.py`:

- `GEMMA_TTS_ENABLED`, default `true`
- `GEMMA_TTS_VOICE`, default `warm-academic`

Existing `GEMMA_URL` and `GEMMA_TIMEOUT_S` from phase 5 still apply.

## Implementation Notes

- Add a small reply data object in `learning-app/app/services/gemma.py`:

  ```python
  @dataclass(frozen=True)
  class GemmaReply:
      text: str
      audio_bytes: bytes | None = None
      audio_duration_ms: float | None = None
      voice: str | None = None
  ```

- Add TTS-aware methods on `GemmaClient`:
  - `listen_audio_with_tts(audio_bytes, *, filename, content_type) -> GemmaReply`
  - `complete_with_tts(prompt) -> GemmaReply`
- Keep older text-only methods for any debug tooling.
- `learning-app/app/services/piper.py` is the playback facade for
  already-rendered WAVs:

  ```python
  pygame.mixer.Sound(file=io.BytesIO(wav_bytes)).play()
  ```

  with an `await asyncio.to_thread(...)` wait loop so the SPEAKING side
  effect can be cancelled cleanly. `aclose()` stops any active channel
  during shutdown / `CANCEL`.
- Stub services must return valid WAV bytes (e.g. 0.5 s of silence) and
  use deterministic short waits so unit tests don't need Gemma, Piper,
  or speakers.

## Tests

`learning-app/tests/test_gemma.py` and additions to `test_fsm.py`. None
of these may need a running Gemma service:

- Gemma client posts multipart audio to `/audio/listen` with
  `stream=false`, `tts=true`, and the configured `voice`.
- Gemma client resolves and fetches a relative `audio_url`.
- Gemma client raises on `audio_error` or a missing `audio_url`.
- Text-fallback `complete_with_tts` posts `/generate` with `tts=true`
  and fetches the returned audio.
- FSM audio path uses the TTS-aware method and emits `REPLY_READY`
  containing both text and WAV bytes.
- `SPEAKING` passes the WAV bytes to the playback facade and emits
  `PLAYBACK_DONE` only after playback returns.
- TTS or playback failure emits `CANCEL` and lands the FSM in `IDLE`.

## Validation

```bash
make -C learning-app lint
make -C learning-app test
```

Manual smoke test:

1. Start the Gemma service; confirm `/health` reports `tts_ready=true`.
2. `make -C learning-app run`.
3. Say "Widushi", ask a short spoken question.
4. Confirm the state sequence is
   `LISTENING → THINKING → SPEAKING → LISTENING`, then either a
   follow-up turn or a `LISTENING → IDLE` silence-timeout fallback.
5. Confirm `THINKING` covers LLM + Piper render time and `SPEAKING`
   covers actual audio playback time.

## Non-Goals

- No local Piper subprocess synthesis in the learning app.
- No streaming NDJSON TTS in this phase.
- No `gemma-llama` changes unless the deployed service is missing the
  documented `/audio/listen` TTS fields.
