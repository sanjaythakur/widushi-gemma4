"""Aggregate metrics for the evaluation harness."""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable

from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def latency_stats(values_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(v) for v in values_ms)
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def pct(p: float) -> float:
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
        return values[idx]

    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "p99": round(pct(99), 2),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classification_metrics(
    predictions: list[list[str]], expected: list[list[str]]
) -> dict[str, float]:
    """Multi-label aware accuracy + micro P/R + macro F1."""
    if not predictions:
        return {"accuracy": 0.0, "micro_precision": 0.0, "micro_recall": 0.0, "macro_f1": 0.0}

    exact = sum(1 for p, e in zip(predictions, expected) if set(p) == set(e))
    accuracy = exact / len(predictions)

    tp = fp = fn = 0
    label_stats: dict[str, Counter] = {}
    for p, e in zip(predictions, expected):
        ps, es = set(p), set(e)
        tp += len(ps & es)
        fp += len(ps - es)
        fn += len(es - ps)
        for lbl in ps | es:
            d = label_stats.setdefault(lbl, Counter())
            if lbl in ps and lbl in es:
                d["tp"] += 1
            elif lbl in ps:
                d["fp"] += 1
            else:
                d["fn"] += 1

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0

    f1s = []
    for d in label_stats.values():
        p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    macro_f1 = statistics.fmean(f1s) if f1s else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "macro_f1": round(macro_f1, 4),
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extraction_metrics(
    predictions: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, float]:
    if not predictions:
        return {"json_valid_rate": 0.0, "field_accuracy": 0.0}

    valid_count = 0
    field_total = 0
    field_correct = 0
    for p, e in zip(predictions, expected):
        is_valid = isinstance(p, dict) and "_error" not in p
        if is_valid:
            valid_count += 1
            for k, v in e.items():
                field_total += 1
                if str(p.get(k)).strip().lower() == str(v).strip().lower():
                    field_correct += 1
        else:
            field_total += len(e)

    return {
        "json_valid_rate": round(valid_count / len(predictions), 4),
        "field_accuracy": round(field_correct / field_total, 4) if field_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------


def keyword_overlap(text: str, keywords: Iterable[str]) -> float:
    """Fraction of ``keywords`` that appear (case-insensitive) in ``text``.

    Used by the audio/video evaluators where we only have soft expectations
    on what the model should mention. Returns 0.0 when ``keywords`` is empty
    so the caller can short-circuit on "no expectations".
    """
    keys = [k.strip().lower() for k in keywords if k and k.strip()]
    if not keys:
        return 0.0
    haystack = (text or "").lower()
    hits = sum(1 for k in keys if k in haystack)
    return hits / len(keys)


def summarization_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = _ROUGE.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": round(statistics.fmean(r1), 4),
        "rouge2": round(statistics.fmean(r2), 4),
        "rougeL": round(statistics.fmean(rl), 4),
    }
