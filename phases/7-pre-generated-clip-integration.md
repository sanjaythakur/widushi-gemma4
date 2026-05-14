# Phase 7 PRD: Pre-generated Clip Integration

## Goal

Use the pre-generated acknowledgement / status WAV clips from
[`pre-generated-clips/`](../pre-generated-clips/) to make Widushi feel
responsive during spoken turns — without adding new FSM states or
events.

## Clip Source

- Clips live at `pre-generated-clips/clips/<lang>/<clip_id>.wav`.
- Default language is `en`, configurable with `WIDUSHI_CLIPS_LANG`.
- Required clip IDs for this phase:
  - `listen_start` — "I'm listening."
  - `wait_thinking` — "Let me think." (first-turn waits)
  - `wait_checking` — "Let me check." (follow-up waits)

## User Experience

- On wake-word detection, before recording the user's question, play
  `listen_start` so the student knows the mic is hot.
- After a first-turn `UTTERANCE_END`, play `wait_thinking` while Gemma
  prepares the answer.
- After a follow-up `UTTERANCE_END`, play `wait_checking` while Gemma
  prepares the answer.
- Reply audio (phase 6) **must not overlap** the ack/status clip.
- Missing clip files must not break a turn — log at WARNING and
  continue silently. The conversation stays usable even with a
  half-shipped clip set.

## Architecture

- Add `ClipPlayer` in `learning-app/app/services/clips.py`:
  - shares pygame's mixer with phase-6's reply playback (`PiperClient`),
  - caches loaded `pygame.mixer.Sound` objects per clip id,
  - exposes `play(clip_id)`, `stop()`, `aclose()`,
  - supports `stub=True` for tests (returns after a short
    `asyncio.sleep`),
  - cancellation-safe: `asyncio.CancelledError` calls `stop()` on the
    active channel before re-raising.
- Add `clips: ClipPlayer` to the `Services` dataclass; wire it into
  both `Services.live()` and `Services.stubs()`.
- Configure clip paths in `learning-app/app/config.py`:

  ```python
  CLIPS_DIR = WORKSPACE_ROOT / "pre-generated-clips" / "clips"
  CLIPS_LANG = os.environ.get("WIDUSHI_CLIPS_LANG", "en")
  ```

  In the container `WORKSPACE_ROOT` resolves to `/`, so the runtime path
  the player looks at is `/pre-generated-clips/clips/<lang>/<id>.wav`.
  The Pi-deploy PRD ([phase 8](8-pi-deploy.md)) describes the bind
  mount the operator must add to the `docker run` command — without it
  every turn logs `clip not found` and runs in silence.

## Wake-word Flow Changes

- `WakeWordSource` accepts an optional `ClipPlayer`.
- After `WAKE_DETECTED` and before `_record_utterance`:
  1. Clear the wake-word frame buffer (stale pre-roll).
  2. `await clips.play("listen_start")`.
  3. Drain any mic frames captured during clip playback. The real
     `MicSource` queue is bounded (~640 ms); without a post-clip drain
     self-captured clip audio echoes back through the speaker into the
     mic and prefixes the user's question. `MicSource.drain()` returns
     the number of dropped chunks (no-op for `FakeMic` in tests).
- Emit `UTTERANCE_END` with `payload["followup"] = False` for the
  wake-word path.
- For the follow-up loopback (post-`PLAYBACK_DONE`): **do not** play
  `listen_start` (the student is mid-conversation). Begin recording
  immediately and emit `UTTERANCE_END` with `payload["followup"] = True`.
  This path must be checked **before** the FSM-aware gate from phase 4.

## THINKING Flow Changes

In `_thinking_side_effect`:

- Choose `wait_checking` when `payload["followup"]` is `True`, otherwise
  `wait_thinking`.
- Start clip playback as a concurrent `asyncio.Task`.
- Run the Gemma request in parallel.
- Wait for **both** to finish before posting `REPLY_READY`. This is what
  prevents the phase-6 reply audio from talking over the cue.
- On Gemma failure or side-effect cancellation: cancel the in-flight
  clip task, drain it, and post `CANCEL` (Gemma failure path).

## Container Packaging

- The clip set is **not** baked into the image. It is mounted at runtime
  from the host so swapping voices / languages does not require an
  image rebuild:

  ```bash
  -v "$PWD/pre-generated-clips/clips:/pre-generated-clips/clips:ro"
  ```

  See [phase 8](8-pi-deploy.md) for the full `docker run` command.

## Tests

`learning-app/tests/test_wakeword.py` and `test_fsm.py`. All run with
`stub=True` clip player and a `RecordingClips` test double that records
playback order:

- `listen_start` plays before wake-word recording starts; UTTERANCE_END
  payload tags `followup=False`.
- Follow-up recording (driven via `FollowUpListenSignal`) skips
  `listen_start` and tags `followup=True`.
- `THINKING` selects `wait_thinking` for first-turn utterances and
  `wait_checking` for follow-ups.
- `REPLY_READY` is only posted after both Gemma and the chosen clip
  finish.
- Gemma failure cancels any in-flight clip and drops the FSM to `IDLE`
  via `CANCEL`.
- Missing clip file logs a warning and returns without raising.

## Non-Goals

- No new FSM state, event, or face for the ack/status clips.
- No runtime clip generation. This phase only consumes WAV files
  produced by the [phase 2](2-pre-generated-clips.md) pipeline.
- No blocking Gemma until wait clips finish; the wait clip and Gemma
  request run concurrently to mask LLM latency.

## Validation

```bash
make -C learning-app lint
make -C learning-app test
```

Manual smoke test on the host:

```bash
WIDUSHI_CLIPS_LANG=en make -C learning-app run
# say "Widushi" — listen_start should play first
# ask a question — wait_thinking should play during Gemma latency
# answer should not overlap wait_thinking
```
