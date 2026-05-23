# Pre-generated TTS clips

This folder generates two batches of `22050 Hz` mono WAV clips, both with
the same two Docker images:

1. **Phase-1 device clips** (`make clips`) -- the acknowledgement/status
   bank defined in [`../phases/1.md`](../phases/1.md). Output lands at
   `clips/en/<clip_id>.wav` and `clips/hi/<clip_id>.wav`.
2. **gemma-llama api-presets** (`make presets`) -- the learner-side audio
   the playground's "Load preset" dropdown plays into each mode endpoint.
   Output lands directly in `../gemma-llama/api-presets/<subdir>/*.wav`
   (the same folder Docker mounts read-only at `/api/presets` for the api
   container).

Both batches are idempotent: existing files with non-zero size are skipped.

## Engines

- English: Piper with `en_US-lessac-medium`
- Hindi: AI4Bharat `indic-parler-tts`

Hindi uses Devanagari input and a feminine tutor-style voice prompt based
on **Divya**. First-person Hindi lines should stay in feminine form
(`समझ गई`, `सोच रही हूँ`, `सुन रही हूँ`).

## Generate phase-1 device clips

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

## Generate gemma-llama api-presets

The api-preset audio is generated from
[`scripts/generate_presets_en.py`](scripts/generate_presets_en.py) (Piper,
6 clean US-English clips) and
[`scripts/generate_presets_hi.py`](scripts/generate_presets_hi.py) (Indic-
Parler-TTS, 10 Hindi / Hinglish / Indian-accented-English / one mumbled
clips). Both write directly into `../gemma-llama/api-presets/<subdir>/*.wav`
via a bind mount.

```bash
HF_TOKEN=<token> make presets
```

Per-language:

```bash
make presets-en
HF_TOKEN=<token> make presets-hi
```

To re-render everything from scratch:

```bash
make clean-presets && HF_TOKEN=<token> make presets
```

Override the destination (e.g. dry-run into a scratch dir before promoting):

```bash
make presets PRESETS_OUT_DIR=$PWD/_scratch
```

### Direct Docker (api-presets)

```bash
docker build -t piper-clips -f docker/Dockerfile.piper .
docker build -t indic-clips -f docker/Dockerfile.indic .

docker run --rm \
    -v "$PWD:/work" \
    -v "$PWD/../gemma-llama/api-presets:/out" \
    -e OUT_DIR=/out \
    --entrypoint python piper-clips \
    /work/scripts/generate_presets_en.py

docker run --rm \
    -v "$PWD:/work" \
    -v "$PWD/../gemma-llama/api-presets:/out" \
    -e OUT_DIR=/out \
    -e HF_TOKEN="$HF_TOKEN" \
    --entrypoint python indic-clips \
    /work/scripts/generate_presets_hi.py
```

## Direct Docker (phase-1 clips)

```bash
docker build -t piper-clips -f docker/Dockerfile.piper .
docker build -t indic-clips -f docker/Dockerfile.indic .

docker run --rm -v "$PWD:/work" piper-clips
docker run --rm -v "$PWD:/work" -e HF_TOKEN=<token> indic-clips
```

## Notes

- `ai4bharat/indic-parler-tts` is gated on HuggingFace, so the Hindi
  pipelines (`clips-hi` and `presets-hi`) require accepting model access
  and providing `HF_TOKEN`.
- If the Hindi voice drifts less feminine than desired, first regenerate
  with a different `SEED`, then try swapping `Divya` to `Rani` in
  `scripts/generate_clips_hi.py` and/or `scripts/generate_presets_hi.py`.
- The api-presets pipeline does **not** generate the four `vision-teach/*.jpg`
  images the playground also references -- those are real-world camera
  shots; drop your own JPEGs (or any common stock photos) at
  `../gemma-llama/api-presets/vision-teach/{milk,apple,book,cup}.jpg`.
