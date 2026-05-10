# Eval media assets

Tiny synthesized media files used by `eval/test_audio.py` and
`eval/test_video.py`. They are intentionally **synthetic** so the repo can
ship them without any licensing concerns -- they are pure ffmpeg `lavfi`
output (a sine tone for audio, ffmpeg's `testsrc` color bars for video) and
contain no creative authorship.

| File                  | Bytes  | Format                              |
| --------------------- | ------ | ----------------------------------- |
| `sample_question.wav` | ~160 KB | 5 s mono 16 kHz s16 PCM, 440 Hz tone |
| `sample_lab.mp4`      | ~53 KB  | 5 s 320x240@15 fps H.264 + 16 kHz mono AAC, 330 Hz tone over `testsrc` |

## What the eval tests actually check

Because there is **no real speech / lab footage** in these clips, the tests
deliberately do not check transcription or visual accuracy. They check the
codepath:

* HTTP 200 from every multimodal route
* Non-empty `text` in the response body
* Correct shape of the JSON response (e.g. `frames_used >= 1`,
  `text` is parseable JSON for `/video/analyze-process`)
* Latency stays within a generous budget

Drop your own real-world recordings (e.g. a 5-10 s spoken question, a short
hands-on clip) on top of these files to upgrade the eval to a semantic
assertion -- the dataset JSON files (`audio.json`, `video.json`) accept
optional `expected_keywords` arrays that the runner will substring-match
case-insensitively against the response when present.

## Regenerating

```bash
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" \
       -ac 1 -ar 16000 -sample_fmt s16 \
       eval/datasets/media/sample_question.wav

ffmpeg -y \
    -f lavfi -i "testsrc=size=320x240:rate=15:duration=5" \
    -f lavfi -i "sine=frequency=330:duration=5" \
    -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
    -c:a aac -b:a 64k -ac 1 -ar 16000 \
    -shortest -movflags +faststart \
    eval/datasets/media/sample_lab.mp4
```
