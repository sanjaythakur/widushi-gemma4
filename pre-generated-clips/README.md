# Pre-generated TTS clips

This folder generates the phase-1 acknowledgement/status clips defined in
[`../phases/1.md`](../phases/1.md).

- Output: `22050 Hz` mono `.wav`
- Paths: `clips/en/<clip_id>.wav` and `clips/hi/<clip_id>.wav`
- Behavior: idempotent; existing files are skipped

## Engines

- English: Piper with `en_US-lessac-medium`
- Hindi: AI4Bharat `indic-parler-tts`

Hindi uses Devanagari input and a feminine tutor-style voice prompt based
on **Divya**. First-person Hindi lines should stay in feminine form
(`समझ गई`, `सोच रही हूँ`, `सुन रही हूँ`).

## Generate

Run from `pre-generated-clips/`.

```bash
make clips
```

If `HF_TOKEN` is not already exported, pass it for the Hindi pipeline:

```bash
HF_TOKEN=<token> make clips
```

Per-language commands:

```bash
make clips-en
HF_TOKEN=<token> make clips-hi
```

Regenerate from scratch:

```bash
make clean-clips && HF_TOKEN=<token> make clips
```

## Direct Docker

```bash
docker build -t piper-clips -f docker/Dockerfile.piper .
docker build -t indic-clips -f docker/Dockerfile.indic .

docker run --rm -v "$PWD:/work" piper-clips
docker run --rm -v "$PWD:/work" -e HF_TOKEN=<token> indic-clips
```

## Notes

- `ai4bharat/indic-parler-tts` is gated on HuggingFace, so Hindi
  generation requires accepting model access and providing `HF_TOKEN`.
- If the Hindi voice drifts less feminine than desired, first regenerate
  with a different `SEED`, then try swapping `Divya` to `Rani` in
  `scripts/generate_clips_hi.py`.
