## Phase 1: Pre-generated Clips

Build a self-contained `pre-generated-clips/` module that generates the
phase-1 acknowledgement/status clips offline.

### Output contract

- Write `22050 Hz`, mono, non-empty `.wav` files to:
  - `pre-generated-clips/clips/en/<clip_id>.wav`
  - `pre-generated-clips/clips/hi/<clip_id>.wav`
- Generation must be idempotent: skip files that already exist.
- English and Hindi may use different TTS engines, but must produce the
  same output format and file paths.

### Engines

- English: Piper with `en_US-lessac-medium`
- Hindi: AI4Bharat `indic-parler-tts`
  - Use Devanagari input, not romanized Hindi.
  - Use a feminine tutor-style voice prompt based on **Divya**.
  - Hindi lines should use feminine first-person agreement where the
    speaker is referring to herself.

### Developer workflow

- Primary entrypoint: `cd pre-generated-clips && make clips`
- Equivalent direct Docker flow:
  - `docker build -t piper-clips -f docker/Dockerfile.piper .`
  - `docker build -t indic-clips -f docker/Dockerfile.indic .`
  - `docker run --rm -v "$PWD:/work" piper-clips`
  - `docker run --rm -v "$PWD:/work" -e HF_TOKEN=<token> indic-clips`
- Hindi model is gated on HuggingFace, so `HF_TOKEN` is required for the
  Hindi pipeline.

### Clip list

| Clip ID | English | Hindi |
|---|---|---|
| `ack_okay` | Okay. | ठीक है। |
| `ack_got_it` | Got it. | समझ गई। |
| `ack_on_it` | On it. | हो जाता है। |
| `ack_sure` | Sure. | ज़रूर। |
| `greet_hello` | Hello. | नमस्ते। |
| `greet_hi` | Hi there. | हाँ, बोलो। |
| `wait_moment` | One moment. | एक सेकंड। |
| `wait_thinking` | Let me think. | सोच रही हूँ। |
| `wait_checking` | Let me check. | देख रही हूँ। |
| `done_here` | Here you go. | लो, ये रहा। |
| `done_set` | All set. | हो गया। |
| `error_repeat` | Sorry, could you say that again? | माफ़ी, दोबारा बोलोगे? |
| `error_unclear` | I didn't quite catch that. | समझ नहीं आई। |
| `error_unknown` | I'm not sure about that. | मुझे पता नहीं। |
| `listen_start` | I'm listening. | बोलो, मैं सुन रही हूँ। |
| `thanks` | Thank you. | शुक्रिया। |