"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .llama_adapter import LlamaAdapter
from .model_config import load_model_config
from .routers import (
    audio,
    chat,
    classify,
    extract,
    free_convo,
    generate,
    summarize,
    video,
    vision,
    voice_mirror,
)
from .schemas import HealthResponse
from .tts import PiperEngine
from .tts.router import router as tts_router
from .tts.storage import TTSStorage

logger = logging.getLogger("gemma_llama")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)

    cfg = load_model_config(settings.model_config_path)
    logger.info("loaded model config: %s (%s)", cfg.display_name, settings.model_config_path)

    adapter = LlamaAdapter(
        base_url=settings.llama_server_url,
        timeout=settings.llm_timeout_seconds,
        retries=settings.llm_retries,
    )
    await adapter.start()

    # ------------------------------------------------------------------
    # TTS subsystem -- best-effort start. A missing binary or empty
    # voices dir leaves the engine in ready=False, which the router
    # helpers honour by returning ``audio_error`` instead of failing.
    # ------------------------------------------------------------------
    tts_engine = PiperEngine(
        binary_path=settings.piper_binary,
        voices_dir=Path(settings.tts_voices_dir),
        default_voice=settings.tts_default_voice,
        max_concurrency=settings.tts_max_concurrency,
        enabled=settings.tts_enabled,
    )
    await tts_engine.start()

    tts_storage = TTSStorage(
        output_dir=Path(settings.tts_output_dir),
        ttl_seconds=settings.tts_output_ttl_seconds,
    )
    if settings.tts_enabled:
        tts_storage.start_janitor()

    app.state.settings = settings
    app.state.model_config = cfg
    app.state.adapter = adapter
    app.state.tts_engine = tts_engine
    app.state.tts_storage = tts_storage
    try:
        yield
    finally:
        await tts_storage.stop_janitor()
        await tts_engine.stop()
        await adapter.stop()


app = FastAPI(
    title="gemma-llama",
    version="0.1.0",
    description=(
        "Local Gemma 4 inference service backed by llama.cpp. Exposes text, "
        "vision, audio, and video endpoints designed as a personalized "
        "educational tutor running on a Raspberry Pi 5."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_and_logging(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception(
            "%s %s -> exception (%.1f ms)", request.method, request.url.path, elapsed_ms
        )
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# --- core endpoints --------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def playground(request: Request):
    cfg = request.app.state.model_config
    engine: PiperEngine | None = getattr(request.app.state, "tts_engine", None)
    return _templates.TemplateResponse(
        request,
        "playground.html",
        {
            "model_display_name": cfg.display_name,
            "model_short_name": cfg.short_name,
            "modalities": cfg.modalities.model_dump(),
            "tts_ready": bool(engine and engine.ready),
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    adapter: LlamaAdapter = request.app.state.adapter
    cfg = request.app.state.model_config
    engine: PiperEngine | None = getattr(request.app.state, "tts_engine", None)
    reachable = await adapter.health_check()
    body = HealthResponse(
        status="ok" if reachable else "degraded",
        model=cfg.short_name,
        llama_server_reachable=reachable,
        tts_ready=bool(engine and engine.ready),
    )
    return JSONResponse(
        status_code=200 if reachable else 503,
        content=body.model_dump(),
    )


# --- routers ---------------------------------------------------------------

app.include_router(generate.router, tags=["inference"])
app.include_router(chat.router, tags=["inference"])
app.include_router(classify.router, tags=["tasks"])
app.include_router(extract.router, tags=["tasks"])
app.include_router(summarize.router, tags=["tasks"])
app.include_router(vision.router, tags=["tutor"])
app.include_router(audio.router, tags=["tutor"])
app.include_router(video.router, tags=["tutor"])
app.include_router(free_convo.router, tags=["modes"])
app.include_router(voice_mirror.router, tags=["modes"])
app.include_router(tts_router, tags=["tts"])
