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


def decode_metrics(rec: RequestRecord, count_tokens: Callable[[str], int | None] | None = None) -> dict[str, Any]:
    """Decode rate = (completion_tokens - 1) / (t_last - t_first).

    Uses server usage when present, else the local tokenizer over the generated text
    (an estimate: tool-call markup re-rendering is not counted). A tokenizer that cannot
    count (it returns None, e.g. the model's own tokenizer failed to load) leaves the count
    empty rather than falling back to another model's tokenizer."""
    times = [c.t for c in rec.chunks]
    gaps = [b - a for a, b in zip(times, times[1:])]
    n = rec.completion_tokens
    source: str | None = "usage"
    if n is None and count_tokens is not None:
        text = rec.reasoning + rec.content + "".join(c.get("name", "") + c.get("arguments", "") for c in rec.tool_calls)
        n = count_tokens(text)
        source = "tokenizer" if n is not None else None
    elif n is None:
        source = None
    span = (rec.t_last - rec.ttft) if rec.ttft is not None and rec.t_last is not None else None
    rate = (n - 1) / span if n and n > 1 and span and span > 0 else None
    return {
        "completion_tokens": n,
        "tokens_source": source,
        "decode_s": span,
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


CAP_TRUNCATED = (
    "incomplete_tool_call",  # oMLX: the token cap cut a tool call; the stream ends with this error code
    "incomplete tool call; increase the output budget",  # lithos-metal: the same case, by message
)


def cap_truncated(m: dict) -> bool:
    """The stream produced tokens and ended at the token cap inside a tool call, which the server
    flagged as a stream error (oMLX and lithos-metal). llama.cpp reports the same case as
    finish_reason=length. A stream that fails before its first token (no TTFT) is not one."""
    err = m.get("error") or ""
    return (not m["ok"] and m.get("ttft_s") is not None and m.get("e2e_s") is not None
            and any(c in err for c in CAP_TRUNCATED))


def server_timing(usage: dict | None) -> tuple[float | None, float | None]:
    """oMLX's own (time_to_first_token, generation_duration) in seconds from a stream's usage; None
    for a server that does not report them."""
    def num(k: str) -> float | None:
        v = (usage or {}).get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return num("time_to_first_token"), num("generation_duration")


def direct_tool_call(m: dict) -> bool:
    """The response opens with a tool call (no content or reasoning first). oMLX sends a tool call as
    one delta only after it has generated all of it, so the client's first visible chunk comes at the
    end: its TTFT includes the call's whole decode, and its decode_s is about 0."""
    return m.get("first_kind") == "tool_call"


def ttft(m: dict) -> tuple[float | None, str | None]:
    """(TTFT, source) of one request. "client": the first visible chunk, as METHODOLOGY defines it.
    For a direct tool call the server's time_to_first_token ("server"); without one the client's
    value is kept but marked "client-incl-decode", so it is never published as a plain TTFT."""
    t = m.get("ttft_s")
    if t is None:
        return None, None
    if direct_tool_call(m):
        st = m.get("server_ttft_s")
        return (float(st), "server") if st is not None else (t, "client-incl-decode")
    return t, "client"


def decode_time(m: dict) -> float | None:
    """Decode time for a decode rate: the server's generation_duration for a direct tool call (whose
    client-side span is ~0), else t_last - ttft; None when there is none."""
    if direct_tool_call(m):
        g = m.get("server_generation_s")
        if g is not None and g > 0:
            return float(g)
    d = m.get("decode_s")
    return d if d is not None and d > 0 else None


def latency_sample(m: dict) -> bool:
    """TTFT and e2e are complete measurements: a successful request that streamed a token, or a
    cap-truncated stream (no usage, so no cached or prompt tokens). A successful stream without a
    token-bearing chunk has no TTFT and is not one."""
    timed = m.get("ttft_s") is not None and m.get("e2e_s") is not None
    return bool(m["ok"] and timed) or cap_truncated(m)


def cap_hit(m: dict) -> str | None:
    """How a request ended at its output cap: "length" (finish_reason) or "incomplete_tool_call"
    (a cap-truncated tool call); None otherwise."""
    if m["ok"] and m.get("finish_reason") == "length":
        return "length"
    if cap_truncated(m):
        return "incomplete_tool_call"
    return None
