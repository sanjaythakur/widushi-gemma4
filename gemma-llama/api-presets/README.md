# api-presets

Drop preset media files in here. The api container mounts this folder
read-only at `/api/presets` (see `docker-compose.yml`) and the playground
reads `manifest.json` at boot to populate a "Load preset" dropdown above
each endpoint's form.

The 16 WAV files referenced by the shipped `manifest.json` can be
generated from the sibling [`pre-generated-clips/`](../../pre-generated-clips/)
sub-project in one command:

```bash
cd ../../pre-generated-clips && HF_TOKEN=hf_xxx make presets
```

That runs the Piper image for the clean US-English presets and the
Indic-Parler-TTS image for the Hindi / Hinglish / Indian-accented-English
/ mumbled-attempt presets, writing all 16 WAVs directly into the
sub-folders below via a bind mount. See
[`../../pre-generated-clips/README.md`](../../pre-generated-clips/README.md#generate-gemma-llama-api-presets)
for per-language targets, scratch-dir overrides, and direct `docker run`
recipes.

The four `vision-teach/*.jpg` images are **not** generated -- drop your
own JPEGs (or any stock photos) at the paths listed below.

## Folder layout

```
api-presets/
├── manifest.json            # describes every preset (committed)
├── free-convo/              # 4 WAVs for POST /free-convo/turn
├── voice-mirror-score/      # 4 WAVs for POST /voice-mirror/score
├── vision-teach/            # 4 image+audio pairs for POST /vision/teach-object
└── audio-listen/            # 4 WAVs for POST /audio/listen
```

`POST /voice-mirror/suggest` is text-only -- it has presets, but they only
populate `level` / `history` JSON fields, no media files are needed.

## Audio format

The api container's `ffmpeg` transcodes whatever you upload to mono 16 kHz
WAV before forwarding to Gemma, so any container works. **Mono 16 kHz WAV
is preferred** for the on-disk presets so they:

- match the server-side normalisation byte-for-byte (no surprises),
- play natively in every browser via `<audio>` (used for in-page preview),
- stay small (<= 1 MiB per ~30 s clip).

A quick recipe with `ffmpeg`:

```bash
ffmpeg -i source.m4a -ac 1 -ar 16000 -f wav out.wav
```

## Required files (rich preset set)

The shipped `manifest.json` references **15 audio files** + **4 images**.
Drop them into the directories listed below; the playground will skip any
preset whose `files:` declaration points at a missing path.

### `free-convo/` (4 audio files)

| Filename | Spoken content (any Indian language is fine if the meaning matches) |
| --- | --- |
| `teach-me-english.wav`           | Hindi: "Mujhe English sikha do" (or "sikhao") |
| `greeting-hindi.wav`             | Hindi: "Namaste, aap kaise ho?" |
| `i-want-to-learn-english.wav`    | English: "I want to learn English" |
| `hinglish-chitchat.wav`          | Hinglish: "Aaj mausam achha hai, kya scene hai?" |

### `voice-mirror-score/` (4 audio files)

| Filename | Spoken content | Paired `target_word` (in manifest) |
| --- | --- | --- |
| `apple-clean.wav`         | Clean US/UK English "apple"                 | `apple` |
| `apple-accented.wav`      | Hindi-accented "apple" (still intelligible) | `apple` |
| `apple-wrong-word.wav`    | Hindi: "angoor"                             | `apple` |
| `river-mumbled.wav`       | Mumbled / unclear "river"                   | `river` |

### `vision-teach/` (4 image + 4 audio = 8 files)

| Image filename     | Subject              | Audio filename            | Spoken guess |
| ---                | ---                  | ---                       | --- |
| `milk.jpg`         | Glass / bottle of milk | `milk-doodh.wav`        | Hindi: "doodh" |
| `apple.jpg`        | An apple             | `apple-guess.wav`         | English: "apple" |
| `book.jpg`         | A book               | `book-kitaab.wav`         | Hindi: "kitaab" |
| `cup.jpg`          | A cup or mug         | `cup-wrong-guess.wav`     | English: "plate" (deliberately wrong) |

Any common JPEG/PNG works (the api resizes everything to 896 px long-edge).
Keep each image &le; 10 MiB.

### `audio-listen/` (4 audio files)

| Filename | Spoken content |
| --- | --- |
| `what-time-is-it.wav`                | English: "What time is it?" |
| `roleplay-shopkeeper-greeting.wav`   | English: "Hello, I want to buy something" |
| `roleplay-doctor-headache.wav`       | English: "I have a headache" |
| `hinglish-question.wav`              | Hinglish: "Yeh kya hai in English?" |

## Editing the manifest

`manifest.json` is the single source of truth for what shows up in the
"Load preset" dropdowns. Each top-level key is an endpoint path
(`free-convo/turn`, `voice-mirror/suggest`, `voice-mirror/score`,
`vision/teach-object`, `audio/listen`); the value is a list of presets:

```json
{
  "id":    "<machine id, shown in the loaded-status line>",
  "label": "<human label shown in the dropdown>",
  "body":  { ... }       // JSON body fields (used by /voice-mirror/suggest)
  "form":  { ... }       // Form-data fields (target_word, max_tokens, ...)
  "files": { "<role>": "<relative path>" }
                          // role: "audio" or "image", path: under api-presets/
}
```

Restart the api container (or simply refresh the playground) after editing
the manifest -- the api reads it fresh on every `GET /presets` request.
