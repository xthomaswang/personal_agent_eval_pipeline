"""Decode-rate and chunk-gap metrics, plus median/min/max helpers."""

from __future__ import annotations

import statistics
from typing import Any, Callable

from .client import RequestRecord


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def decode_metrics(rec: RequestRecord, count_tokens: Callable[[str], int] | None = None) -> dict[str, Any]:
    """Decode rate = (completion_tokens - 1) / (t_last - t_first).

    Uses server usage when present, else the local tokenizer over the generated text
    (an estimate: tool-call markup re-rendering is not counted)."""
    times = [c.t for c in rec.chunks]
    gaps = [b - a for a, b in zip(times, times[1:])]
    n = rec.completion_tokens
    source = "usage"
    if n is None and count_tokens is not None:
        text = rec.reasoning + rec.content + "".join(c.get("name", "") + c.get("arguments", "") for c in rec.tool_calls)
        n, source = count_tokens(text), "tokenizer"
    elif n is None:
        source = None
    span = (rec.t_last - rec.ttft) if rec.ttft is not None and rec.t_last is not None else None
    rate = (n - 1) / span if n and n > 1 and span and span > 0 else None
    return {
        "completion_tokens": n,
        "tokens_source": source,
        "decode_tps": rate,
        "n_token_chunks": len(times),
        "tokens_per_chunk": (n / len(times)) if n and times else None,
        "gap_p50_ms": None if not gaps else 1000 * percentile(gaps, 0.5),
        "gap_p99_ms": None if not gaps else 1000 * percentile(gaps, 0.99),
    }


def mmm(xs: list[float | None]) -> dict[str, float | None]:
    v = [x for x in xs if x is not None]
    if not v:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
