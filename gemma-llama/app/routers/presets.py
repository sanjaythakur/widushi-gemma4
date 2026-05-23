"""Playground preset router.

Reads ``/api/presets/manifest.json`` (mounted from ``./api-presets`` on the
host via ``docker-compose.yml``) and serves it alongside the raw media files
it references. The playground HTML calls these routes to populate a "Load
preset" dropdown per endpoint and to fetch the WAV/JPG bytes when the user
picks one.

Two routes:

* ``GET /presets`` -- returns the parsed manifest plus an ``available``
  boolean. When the manifest is missing the playground simply hides the
  preset dropdowns; everything else still works.
* ``GET /presets/files/{relpath}`` -- streams a single preset file. The
  ``relpath`` is whatever the manifest declared; we resolve it inside the
  mount root and refuse anything that escapes.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _presets_root() -> Path:
    """Resolve the on-disk root for playground presets.

    Defaults to ``/api/presets`` (the mount target in ``docker-compose.yml``).
    Override with ``API_PRESETS_DIR`` for dev runs outside Docker.
    """
    return Path(os.environ.get("API_PRESETS_DIR", "/api/presets")).resolve()


def _manifest_path() -> Path:
    return _presets_root() / "manifest.json"


@router.get("/presets", tags=["playground"], include_in_schema=False)
async def get_presets():
    """Return the parsed preset manifest, or ``{available: false}`` if absent.

    Failure modes (missing dir, missing/invalid JSON) are logged but never
    raise -- the playground UI degrades gracefully into "no presets" mode.
    """
    path = _manifest_path()
    if not path.exists():
        return JSONResponse({"available": False, "reason": f"{path} does not exist"})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("preset manifest unreadable at %s: %s", path, exc)
        return JSONResponse({"available": False, "reason": f"manifest unreadable: {exc}"})
    if not isinstance(data, dict):
        return JSONResponse({"available": False, "reason": "manifest must be a JSON object"})
    return JSONResponse({"available": True, "presets": data})


@router.get("/presets/files/{relpath:path}", tags=["playground"], include_in_schema=False)
async def get_preset_file(relpath: str):
    """Serve a single preset asset (audio, image, ...) from the mount root.

    The path is resolved against ``_presets_root()`` and any attempt to
    escape via ``..`` returns ``400``. We pick the MIME type via
    :mod:`mimetypes`; unknown extensions fall back to
    ``application/octet-stream`` (the browser still plays WAV/JPEG/PNG).
    """
    root = _presets_root()
    candidate = (root / relpath).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path escapes presets root") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"preset not found: {relpath}")
    mime, _ = mimetypes.guess_type(candidate.name)
    return FileResponse(
        path=str(candidate),
        media_type=mime or "application/octet-stream",
        filename=candidate.name,
    )
