# Phase 7: Pre-generated Clip Integration PRD

## Goal

Use pre-generated acknowledgement/status WAV clips to make Widushi feel responsive during spoken turns without adding a new FSM state.

## Clip Source

- Clips live under `pre-generated-clips/clips/<lang>/<clip_id>.wav`.
- The default language is `en`, configurable with `WIDUSHI_CLIPS_LANG`.
- Required clip IDs for this phase:
  - `listen_start`
  - `wait_thinking`
  - `wait_checking`

## User Experience

- On wake-word detection, before recording the user's question, Widushi plays `listen_start` to indicate that it is listening.
- After a first-turn utterance ends, Widushi plays `wait_thinking` while Gemma prepares the answer.
- After a follow-up utterance ends, Widushi plays `wait_checking` while Gemma prepares the follow-up answer.
- Reply audio must not overlap the acknowledgement/status clip.
- Missing clip files must not break a conversation turn; log and continue silently from the user's perspective.

## Architecture

- Add a `ClipPlayer` service for pre-rendered WAV playback.
- `ClipPlayer` should share pygame's mixer with reply playback, cache loaded sounds, support cancellation, expose `play(clip_id)`, `stop()`, and `aclose()`, and support a stub mode for tests.
- Add `clips: ClipPlayer` to the shared `Services` bundle and wire it into live and stub services.
- Configure clip language in `app/config.py`:

```python
CLIPS_DIR = WORKSPACE_ROOT / "pre-generated-clips" / "clips"
CLIPS_LANG = os.environ.get("WIDUSHI_CLIPS_LANG", "en")
```

## Wake-word Flow

- `WakeWordSource` accepts an optional `ClipPlayer`.
- When wake-word detection emits `WAKE_DETECTED`, clear stale wake-word pre-roll, play `listen_start`, drain any mic frames captured during playback, then begin `_record_utterance`.
- Draining is needed because the real sounddevice mic queue is bounded and can otherwise retain self-captured clip audio.
- Emit `UTTERANCE_END` with `payload["followup"] = False` for the wake-word path.
- For the follow-up loopback path, do not play `listen_start`; begin recording immediately and emit `UTTERANCE_END` with `payload["followup"] = True`.

## THINKING Flow

- Keep the existing FSM states and transitions unchanged.
- In the `LISTENING -> THINKING` side effect:
  - Choose `wait_checking` when `payload["followup"]` is true.
  - Otherwise choose `wait_thinking`.
  - Start clip playback concurrently with the Gemma request.
  - Wait for both Gemma and the clip to finish before posting `REPLY_READY`, preventing reply audio from talking over the clip.
  - If Gemma fails or the side effect is cancelled, cancel any in-flight clip playback and post the existing `CANCEL` event on Gemma failure.

## Non-goals

- Do not add an `ACK_SPEAK` state, `ACK_DONE` event, or new face renderer.
- Do not block Gemma until wait clips finish; wait clips should mask Gemma latency by running concurrently.
- Do not generate clips at runtime; this phase only consumes already-generated WAV files.

## Validation

- Unit tests should cover:
  - `listen_start` plays before wake-word recording and tags `followup=False`.
  - Follow-up recording skips `listen_start` and tags `followup=True`.
  - THINKING selects `wait_thinking` for first-turn utterances.
  - THINKING selects `wait_checking` for follow-up utterances.
  - `REPLY_READY` waits for both Gemma and the selected clip.
  - Gemma failure cancels any in-flight clip and returns the FSM to `IDLE` via `CANCEL`.
- Run `pytest` and `ruff check` for `learning-app`.