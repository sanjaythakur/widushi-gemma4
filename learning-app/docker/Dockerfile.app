# Image for the Widushi main process (pygame + asyncio + FastAPI).
#
# Base is the same `python:3.11-slim` used by the pre-generated-clips
# images. The image is multi-arch on Docker Hub (linux/amd64 and
# linux/arm64), so the same Dockerfile builds on a Mac M-series host
# and on a Raspberry Pi 5.
#
# Build (from the workspace root):
#   docker build -t widushi-app -f learning-app/docker/Dockerfile.app .
#
# Run (Mac, headless, FastAPI only):
#   docker run --rm -p 8020:8020 \
#     -e SDL_VIDEODRIVER=dummy -e SDL_AUDIODRIVER=dummy widushi-app
#
# Run (Raspberry Pi, framebuffer):
#   docker run --rm --device /dev/fb0 --device /dev/snd \
#     -e SDL_VIDEODRIVER=fbcon widushi-app

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# SDL2 runtime libs for pygame. We keep the list intentionally small —
# fonts come from the system DejaVu set so we skip libsdl2-ttf-extras.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libasound2 \
        libportaudio2 \
        libsdl2-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libsdl2-image-2.0-0 \
        libfreetype6 \
        fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches across code changes.
COPY learning-app/pyproject.toml ./
RUN pip install \
        "pygame>=2.5" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "httpx>=0.27" \
        "aiosqlite>=0.19" \
        "pydantic>=2" \
        "numpy>=1.26" \
        "onnxruntime>=1.17" \
        "openwakeword>=0.6" \
        "sounddevice>=0.4"

# openWakeWord's PyPI package does not ship the shared feature extractor
# models it loads at runtime (for example melspectrogram.onnx). Download them
# during image build so the Pi can start wake-word detection without a first-run
# network dependency. Passing our custom model name avoids downloading the
# stock wake-word models; the shared feature/VAD models are always fetched.
RUN python -c "from openwakeword.utils import download_models; download_models(model_names=['vDu_shee'])"

# Application source.
COPY learning-app/app /app/app
COPY openwakeword /openwakeword

EXPOSE 8020

# Default display/audio drivers are headless. For local compose, wake-word
# detection is explicitly disabled; Pi runs can use the packaged model below.
ENV SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    WIDUSHI_API_HOST=0.0.0.0 \
    WIDUSHI_API_PORT=8020 \
    WAKE_WORD_MODEL=/openwakeword/vDu_shee.onnx

CMD ["python", "-m", "app.main"]
