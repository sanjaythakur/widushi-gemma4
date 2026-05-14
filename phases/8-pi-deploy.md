# Phase 8 PRD: Raspberry Pi 5 Deploy

## Goal

Run the phase 3–7 learning app on a Raspberry Pi 5 with a 3.5" SPI TFT
panel, a USB mic, and a USB speaker, talking to the Gemma container on
port `8010`. Same image as the Mac headless dev container — only the
runtime flags change.

## Reference Hardware

- Raspberry Pi 5, 64-bit OS (`linux/arm64`).
- 3.5" SPI TFT, MHS35 (ILI9486), driven by the kernel's `fbtft` stack.
  The `mhs35` overlay needs `:rotate=90` (or `:rotate=270`) in
  `/boot/firmware/config.txt` so the framebuffer reports `480×320` and
  matches `config.SCREEN_SIZE`.
- USB mic and USB speaker (the Pi 5 has no on-board analog audio).

## Display: Framebuffer Sink

SDL2 has no `fbcon`/`fbdev` driver, and `fbtft` panels expose no
`/dev/dri`, so `KMSDRM` is also out. The app instead renders into an
off-screen surface (`SDL_VIDEODRIVER=dummy`) and copies each frame into
`/dev/fb0` itself.

`learning-app/app/hardware/fb_sink.py` — `FbSink`:

- `FbSink.create_from_env()` reads `WIDUSHI_FB_DEVICE` (e.g. `/dev/fb0`).
  Returns `None` when unset so Mac (cocoa) and headless dev runs use the
  normal `pygame.display.flip()` path with no code changes.
- Reads `width`, `height`, `bits_per_pixel` from
  `/sys/class/graphics/<fbN>/{virtual_size,bits_per_pixel}` (no
  `FBIOGET_VSCREENINFO` ioctls — sysfs is enough for fbtft panels with
  tight stride).
- Only supports 16 bpp panels. Per frame, takes the pygame RGB888
  surface, transposes to row-major (y, x, 3), packs into RGB565
  (`(r>>3)<<11 | (g>>2)<<5 | b>>3`), writes to an mmap'd `/dev/fb0`.
- Falls back to `pygame.transform.smoothscale` if the surface size
  doesn't match the panel — better fuzzy than crash if the overlay
  rotation is wrong.
- `close()` releases the mmap and closes the fd.

The UI loop calls `fb_sink.push(self._screen)` after
`pygame.display.flip()` each frame when the sink exists.

## Audio: ALSA Routing

USB devices typically split across two ALSA cards:

```text
$ aplay  -l
card 1: UACDemoV10 [UACDemoV1.0], device 0: USB Audio
$ arecord -l
card 0: Device [USB PnP Sound Device], device 0: USB Audio
```

Default ALSA `default` PCM points at `hw:0,0`, which is the capture-only
mic — opening it for playback fails. `learning-app/docker/asound.conf`
ships a routing file that pins `default` capture to the USB mic
(card 0) and `default` playback to the USB speaker (card 1):

```
pcm.!default { type asym  playback.pcm "speaker"  capture.pcm "mic" }
ctl.!default { type hw    card 1 }
pcm.speaker  { type plug  slave { pcm "hw:1,0" } }
pcm.mic      { type plug  slave { pcm "hw:0,0" } }
```

The Dockerfile copies this to `/etc/asound.conf` so pygame's mixer
(SDL `default`) and `sounddevice` (`device=None`) both land on the
right hardware. If the Pi enumerates the cards in a different order,
either edit `asound.conf` and rebuild, or override at runtime with
`-e WIDUSHI_INPUT_DEVICE` / `-e WIDUSHI_OUTPUT_DEVICE` /
`-e SDL_AUDIODEV=plughw:1,0`.

`make -C learning-app install-asound` drops the same file onto the
host's `/etc/asound.conf` so `aplay`, `arecord`, and `speaker-test`
follow the same routing for off-container debugging.

## Image (recap of phases 3–7)

`learning-app/docker/Dockerfile.app` from `python:3.11-slim`:

- OS deps: `ca-certificates`, `libasound2`, `libportaudio2`,
  `libsdl2-2.0-0`, `libsdl2-mixer-2.0-0`, `libsdl2-ttf-2.0-0`,
  `libsdl2-image-2.0-0`, `libfreetype6`, `fonts-dejavu-core`.
- pip deps: `pygame>=2.5`, `fastapi>=0.110`, `uvicorn[standard]>=0.27`,
  `httpx>=0.27`, `aiosqlite>=0.19`, `pydantic>=2`, `numpy>=1.26`,
  `onnxruntime>=1.17`, `openwakeword>=0.6`, `sounddevice>=0.4`.
- Pre-fetch openWakeWord shared feature models during build so first
  boot has no network dependency.
- `COPY learning-app/docker/asound.conf /etc/asound.conf`.
- `COPY learning-app/app /app/app`.
- `COPY openwakeword /openwakeword`. Set
  `WAKE_WORD_MODEL=/openwakeword/vDu_shee.onnx`.
- Default env: `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`,
  `WIDUSHI_API_HOST=0.0.0.0`, `WIDUSHI_API_PORT=8020`. The Pi runtime
  flips `SDL_AUDIODRIVER` to `alsa` and adds `WIDUSHI_FB_DEVICE`.

Build from the workspace root so the sibling `openwakeword/` is in the
Docker context:

```bash
docker build -t widushi-app -f learning-app/docker/Dockerfile.app .
```

## Networking to the Gemma Container

`gemma-llama-api` publishes port `8010` on the host. Inside a bridged
container `localhost` points at the container itself, so the default
`GEMMA_URL=http://localhost:8010` fails with
`httpx.ConnectError: All connection attempts failed`. Two ways to fix:

- **`--network host` (recommended on Pi).** The app shares the host
  network namespace, so `localhost:8010` resolves to the Gemma
  container's published port. The `-p 8020:8020` flag becomes a no-op
  (Docker prints a warning) and is dropped.
- **Bridge networking.** Add `--add-host=host.docker.internal:host-gateway`
  and `-e GEMMA_URL=http://host.docker.internal:8010`. This is what
  `learning-app/docker/docker-compose.yml` uses for Mac dev.

## Pi `docker run`

Run from the workspace root so the clip bind-mount path resolves:

```bash
docker run --rm \
  --network host \
  --device /dev/fb0 --device /dev/snd \
  --group-add audio \
  -e SDL_VIDEODRIVER=dummy \
  -e SDL_AUDIODRIVER=alsa \
  -e WIDUSHI_FB_DEVICE=/dev/fb0 \
  -e GEMMA_URL=http://localhost:8010 \
  -v "$PWD/pre-generated-clips/clips:/pre-generated-clips/clips:ro" \
  widushi-app
```

What each flag does:

| Flag | Why |
|---|---|
| `--network host` | Reach `gemma-llama-api`'s host-published `:8010`. |
| `--device /dev/fb0` | SPI TFT framebuffer the `FbSink` writes to. |
| `--device /dev/snd` | USB mic + USB speaker. |
| `--group-add audio` | Container user must be in `audio` to open ALSA. |
| `SDL_VIDEODRIVER=dummy` | SDL renders off-screen; `FbSink` does the visible blit. |
| `SDL_AUDIODRIVER=alsa` | Pygame mixer / Piper playback go through ALSA → `/etc/asound.conf`. |
| `WIDUSHI_FB_DEVICE=/dev/fb0` | Activates `FbSink.create_from_env()`. |
| `GEMMA_URL=http://localhost:8010` | Required because the image default `localhost` only worked with bridge networking + `host.docker.internal`. |
| `-v …/pre-generated-clips/clips:/pre-generated-clips/clips:ro` | Phase 7 clips. Without this, every wake-word turn logs `clip not found: /pre-generated-clips/clips/en/listen_start.wav`. |

## Per-subsystem Smoke Tests

Verify each subsystem **before** running the full app — saves time
debugging interactions:

```bash
# Display: panel briefly flashes, then goes black.
cat /dev/urandom > /dev/fb0; sleep 0.5; dd if=/dev/zero of=/dev/fb0 bs=1M count=1

# Speaker (uses the asound.conf default = USB speaker).
speaker-test -c2 -twav

# Mic.
arecord -d 3 -fS16_LE -r16000 -c1 /tmp/mic.wav
aplay /tmp/mic.wav

# Gemma reachable from the host.
curl http://localhost:8010/health
```

## RMS Calibration After a Mic / Room Change

The room noise floor is mic-specific. The default
`LISTENING_SILENCE_RMS_THRESHOLD=800` is tuned for a quiet room with the
original USB mic. **Symptom of a wrong threshold:** every recording
captures exactly `LISTENING_MAX_RECORDING_S` seconds (e.g. `15.04s`)
because trailing silence never fires. Recalibrate:

```bash
docker run --rm \
  --network host --device /dev/fb0 --device /dev/snd --group-add audio \
  -e SDL_VIDEODRIVER=dummy -e SDL_AUDIODRIVER=alsa \
  -e WIDUSHI_FB_DEVICE=/dev/fb0 -e GEMMA_URL=http://localhost:8010 \
  -e WIDUSHI_RMS_DEBUG=1 \
  -v "$PWD/pre-generated-clips/clips:/pre-generated-clips/clips:ro" \
  widushi-app
```

Per 80 ms recorder chunk this prints:

```text
INFO app.input.wakeword: rms[042] t=3.36s rms=124 thr=800 silent run=0 seen=True
```

Say "Widushi", stay quiet ~2 s, speak normally for a few seconds, stay
quiet again. Read the `rms=` column. Set
`-e LISTENING_SILENCE_RMS_THRESHOLD=<value>` comfortably above the
quiet-room peaks but below soft speech, then drop `-e WIDUSHI_RMS_DEBUG=1`.
For very noisy rooms also bump `-e LISTENING_VOICE_ONSET_FRAMES=3`.
For environments with longer between-phrase pauses, bump
`-e LISTENING_TRAILING_SILENCE_S=2.0`.

Re-run this whenever the input device, room, or background AC changes.

## Operational Checklist

Before declaring a Pi deployment good:

- [ ] `aplay -l` and `arecord -l` agree with the card numbers in
      `learning-app/docker/asound.conf`.
- [ ] `cat /sys/class/graphics/fb0/virtual_size` reports `480,320`.
- [ ] `curl http://localhost:8010/health` returns OK and
      `tts_ready=true`.
- [ ] `pre-generated-clips/clips/en/{listen_start,wait_thinking,wait_checking}.wav`
      exist on the host.
- [ ] `gemma-llama-api` container is healthy and exposes `:8010`.
- [ ] `widushi-app` container starts without `fbcon not available`,
      without `cannot find card '0'`, and without `httpx.ConnectError`.
- [ ] Recording terminates before `LISTENING_MAX_RECORDING_S` (i.e. the
      RMS threshold is calibrated for this mic / room).
- [ ] Logs don't show `clip not found` warnings.
- [ ] Logs don't show `wake word detected` lines while UI is in
      `THINKING`/`SPEAKING` (phase-4 FSM gate working).

## Non-Goals

- No multi-Pi orchestration / fleet management.
- No HTTPS / external internet exposure of `:8020`.
- No on-device model fine-tuning. The Pi consumes models built upstream
  (`gemma-llama` GGUF, openWakeWord ONNX, Piper voice JSON+ONNX).
