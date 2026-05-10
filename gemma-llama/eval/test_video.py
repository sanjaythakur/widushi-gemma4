"""Live video-route eval tests.

Skipped automatically when:
    * the gemma-llama API is not reachable, or
    * the active model does not advertise ``video: true`` (the route returns
      409, treated as a skip).

Bundled media is the synthetic ``sample_lab.mp4`` (ffmpeg ``testsrc`` color
bars + 330 Hz sine, see ``eval/datasets/media/README.md``). Assertions
target codepath correctness (HTTP 200, frames sampled, JSON shape) rather
than visual / audio comprehension. Drop a real clip and add
``expected_keywords`` to ``eval/datasets/video.json`` to upgrade.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from .client import APIClient

_DATASETS = Path(__file__).parent / "datasets"


def _load() -> list[dict]:
    return json.loads((_DATASETS / "video.json").read_text(encoding="utf-8"))


def _check_video_capable(exc: httpx.HTTPStatusError) -> None:
    if exc.response.status_code == 409:
        pytest.skip(
            f"Active model does not support video (server said 409: {exc.response.text})"
        )
    raise exc


@pytest.mark.asyncio
@pytest.mark.parametrize("item", _load(), ids=lambda i: Path(i["file"]).name)
async def test_video_analyze_process(client: APIClient, item: dict) -> None:
    media = _DATASETS / item["file"]
    if not media.exists():
        pytest.skip(f"missing bundled asset: {media}")

    n_frames = int(item.get("n_frames", 6))
    try:
        resp = await client.video_analyze(
            media,
            task=item.get("task"),
            n_frames=n_frames,
            include_audio=bool(item.get("include_audio", True)),
        )
    except httpx.HTTPStatusError as exc:
        _check_video_capable(exc)
        raise

    text = (resp.get("text") or "").strip()
    assert text, f"video returned empty text: {resp}"
    assert resp.get("model")
    assert isinstance(resp.get("inference_time_ms"), (int, float))
    assert resp.get("frames_used", 0) >= max(1, n_frames - 1), (
        f"frames_used={resp.get('frames_used')} too low for n_frames={n_frames}"
    )
    # We always request audio in the bundled dataset; if the model config
    # disables audio at runtime the server flips include_audio off silently.
    assert isinstance(resp.get("audio_used"), bool)

    expected_keys = item.get("expected_keys") or []
    if expected_keys:
        # JSON-format enforcement is on for non-streaming /video/analyze-process,
        # so the model SHOULD return a parseable object. We assert that here.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            pytest.fail(f"video response is not valid JSON: {exc}\ntext={text!r}")
        missing = [k for k in expected_keys if k not in parsed]
        assert not missing, (
            f"video response missing expected keys {missing!r}; got {list(parsed)!r}"
        )
