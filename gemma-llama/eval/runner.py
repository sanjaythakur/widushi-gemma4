"""CLI evaluation runner: ``python -m eval.runner --all|--task X``."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import APIClient
from .metrics import (
    classification_metrics,
    extraction_metrics,
    keyword_overlap,
    latency_stats,
    summarization_metrics,
)

_DATASETS = Path(__file__).parent / "datasets"
_REPORTS = Path(__file__).parent / "reports"


def _load_dataset(name: str) -> list[dict[str, Any]]:
    path = _DATASETS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


# ---------------------------------------------------------------------------
# Per-task runners
# ---------------------------------------------------------------------------


async def run_classify(client: APIClient) -> dict[str, Any]:
    items = _load_dataset("classification")
    preds: list[list[str]] = []
    expected: list[list[str]] = []
    latencies: list[float] = []
    raw_results: list[dict[str, Any]] = []
    for item in items:
        resp = await client.classify(item["text"], item["labels"], item.get("multi_label", False))
        preds.append(_normalise_labels(resp.get("labels")))
        expected.append(_normalise_labels(item["expected"]))
        latencies.append(float(resp.get("inference_time_ms", 0.0)))
        raw_results.append({"item": item, "response": resp})
    return {
        "task": "classify",
        "n": len(items),
        "metrics": classification_metrics(preds, expected),
        "latency_ms": latency_stats(latencies),
        "results": raw_results,
    }


async def run_extract(client: APIClient) -> dict[str, Any]:
    items = _load_dataset("extraction")
    preds: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    latencies: list[float] = []
    raw_results: list[dict[str, Any]] = []
    for item in items:
        resp = await client.extract(item["text"], item["schema"])
        data = resp.get("data") or {}
        preds.append(data)
        expected.append(item["expected"])
        latencies.append(float(resp.get("inference_time_ms", 0.0)))
        raw_results.append({"item": item, "response": resp})
    return {
        "task": "extract",
        "n": len(items),
        "metrics": extraction_metrics(preds, expected),
        "latency_ms": latency_stats(latencies),
        "results": raw_results,
    }


async def run_summarize(client: APIClient) -> dict[str, Any]:
    items = _load_dataset("summarization")
    preds: list[str] = []
    refs: list[str] = []
    latencies: list[float] = []
    raw_results: list[dict[str, Any]] = []
    for item in items:
        resp = await client.summarize(
            item["text"], style=item.get("style"), max_sentences=item.get("max_sentences", 3)
        )
        preds.append(resp.get("summary", ""))
        refs.append(item["reference"])
        latencies.append(float(resp.get("inference_time_ms", 0.0)))
        raw_results.append({"item": item, "response": resp})
    return {
        "task": "summarize",
        "n": len(items),
        "metrics": summarization_metrics(preds, refs),
        "latency_ms": latency_stats(latencies),
        "results": raw_results,
    }


async def run_audio(client: APIClient) -> dict[str, Any]:
    items = _load_dataset("audio")
    latencies: list[float] = []
    raw_results: list[dict[str, Any]] = []
    nonempty = 0
    keyword_hits = 0
    keyword_total = 0
    for item in items:
        path = _DATASETS / item["file"]
        if not path.exists():
            raw_results.append({"item": item, "skipped": f"missing media: {path}"})
            continue
        mode = item["mode"]
        if mode == "listen":
            resp = await client.audio_listen(path)
        elif mode == "transcribe":
            resp = await client.audio_transcribe(path)
        elif mode == "translate":
            resp = await client.audio_translate(
                path,
                target_language=item.get("target_language") or "English",
                source_language=item.get("source_language"),
            )
        else:
            raw_results.append({"item": item, "skipped": f"unknown mode: {mode}"})
            continue
        text = (resp.get("text") or "").strip()
        if text:
            nonempty += 1
        keywords = item.get("expected_keywords") or []
        if keywords:
            keyword_total += 1
            if keyword_overlap(text, keywords) >= 0.5:
                keyword_hits += 1
        latencies.append(float(resp.get("inference_time_ms", 0.0)))
        raw_results.append({"item": item, "response": resp})
    n = max(1, len(items))
    metrics = {
        "non_empty_rate": round(nonempty / n, 4),
        "keyword_pass_rate": (
            round(keyword_hits / keyword_total, 4) if keyword_total else None
        ),
    }
    return {
        "task": "audio",
        "n": len(items),
        "metrics": metrics,
        "latency_ms": latency_stats(latencies),
        "results": raw_results,
    }


async def run_video(client: APIClient) -> dict[str, Any]:
    items = _load_dataset("video")
    latencies: list[float] = []
    raw_results: list[dict[str, Any]] = []
    nonempty = 0
    json_valid = 0
    key_match = 0
    key_total = 0
    for item in items:
        path = _DATASETS / item["file"]
        if not path.exists():
            raw_results.append({"item": item, "skipped": f"missing media: {path}"})
            continue
        resp = await client.video_analyze(
            path,
            task=item.get("task"),
            n_frames=int(item.get("n_frames", 6)),
            include_audio=bool(item.get("include_audio", True)),
        )
        text = (resp.get("text") or "").strip()
        if text:
            nonempty += 1
        try:
            parsed = json.loads(text)
            json_valid += 1
            expected = item.get("expected_keys") or []
            if expected:
                key_total += 1
                if all(k in parsed for k in expected):
                    key_match += 1
        except (json.JSONDecodeError, TypeError):
            pass
        latencies.append(float(resp.get("inference_time_ms", 0.0)))
        raw_results.append({"item": item, "response": resp})
    n = max(1, len(items))
    metrics = {
        "non_empty_rate": round(nonempty / n, 4),
        "json_valid_rate": round(json_valid / n, 4),
        "expected_keys_rate": (
            round(key_match / key_total, 4) if key_total else None
        ),
    }
    return {
        "task": "video",
        "n": len(items),
        "metrics": metrics,
        "latency_ms": latency_stats(latencies),
        "results": raw_results,
    }


_TASKS = {
    "classify": run_classify,
    "extract": run_extract,
    "summarize": run_summarize,
    "audio": run_audio,
    "video": run_video,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> int:
    tasks = list(_TASKS) if args.all else [args.task]
    _REPORTS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }
    async with APIClient() as client:
        health = await client.health()
        summary["health"] = health
        if not health.get("llama_server_reachable"):
            print("WARN: llama_server_reachable=false; results may be empty.", file=sys.stderr)
        for task_name in tasks:
            print(f"\n=== running task: {task_name} ===")
            result = await _TASKS[task_name](client)
            summary["tasks"][task_name] = {
                "metrics": result["metrics"],
                "latency_ms": result["latency_ms"],
                "n": result["n"],
            }
            print(json.dumps(summary["tasks"][task_name], indent=2))

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = _REPORTS / f"report-{task_name}-{ts}.json"
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"-> wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run gemma-llama evaluations.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Run every task.")
    g.add_argument("--task", choices=sorted(_TASKS), help="Run a single task.")
    args = p.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
