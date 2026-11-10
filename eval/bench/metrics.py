"""The fixed metrics block of a run: what a person feels, with the same keys in every cell.

    block(calls, events=None, wall_s=None) -> dict with exactly the keys in KEYS

*calls* are the recording proxy's rows (``calls.jsonl``). *events* are the gateway's event stream of the same run, when
the run went through a gateway (the product path). A figure that the given records cannot give is ``None``, never 0:
a direct run has no event stream, so it has no first sign, no failed tool calls and no approvals to count.

The block exists because the figures were recorded and never shown (cached tokens), or not countable at all (repeated
tool calls): a build that broke the prompt prefix, or one that read the same page four times, changed no summary.
Only the standard library is used, so a script run with the system Python can import this file.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

KEYS = (
    "wall_s",  # the run's wall time, seconds
    "first_sign_s",  # from the first turn's start to the first tool card or reply a person could see (events)
    "model_calls",  # generation requests, main and aux
    "failed_model_calls",  # of those: an error, a status other than 200, or a stream cut before its end
    "tool_calls",  # tool calls started (events), or tool calls in the main replies (calls)
    "repeat_calls",  # tool calls with the same tool and arguments as an earlier call of the run
    "failed_tools",  # tool calls that did not end in success (events)
    "approvals_asked",  # approval cards (events)
    "checkpoints",  # work checkpoints: the run noted where it stood and its oldest tool results left the context (events)
    "prompt_first",  # prompt tokens of the first main call, of the median one and of the largest
    "prompt_median",
    "prompt_max",
    "cached_share",  # of all main prompt tokens, the share the server took from its cache
    "reasoning_tokens",  # thinking tokens, when the server reports them
    "reasoning_chars",  # thinking characters seen by the proxy
)
SIGN_EVENTS = ("tool.started", "assistant.message")
GENERATION_ROLES = ("main", "aux")


def _stamped(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (e for e in events if isinstance(e.get("ts"), (int, float))),
        key=lambda e: e["ts"],
    )


def first_sign_s(events: Sequence[dict[str, Any]]) -> float | None:
    """Seconds from the first turn's start to the first thing a person could see of it: a tool card or a reply."""
    stamped = _stamped(events)
    start = next((e["ts"] for e in stamped if e.get("type") == "turn.started"), None)
    if start is None:
        return None
    sign = next(
        (e["ts"] for e in stamped if e["ts"] >= start and e.get("type") in SIGN_EVENTS),
        None,
    )
    return None if sign is None else round(sign - start, 1)


def event_repeats(events: Iterable[dict[str, Any]]) -> int:
    """Tool calls with the same tool and the same arguments as an earlier call of the run (from the event stream)."""
    seen: set[str] = set()
    repeats = 0
    for event in events:
        if event.get("type") != "tool.started":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        key = json.dumps(
            [data.get("name"), data.get("arguments")],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        repeats += key in seen
        seen.add(key)
    return repeats


def call_repeats(calls: Iterable[dict[str, Any]]) -> int | None:
    """The same count from the proxy's tool-call hashes; None when a reply with tool calls has no hashes (an old record)."""
    seen: set[str] = set()
    repeats = 0
    for row in calls:
        if row.get("role") != "main" or not row.get("tool_calls"):
            continue
        hashes = row.get("tool_call_sha256")
        if not isinstance(hashes, list) or len(hashes) != row["tool_calls"]:
            return None
        for h in hashes:
            repeats += h in seen
            seen.add(h)
    return repeats


def failed_tools(events: Iterable[dict[str, Any]]) -> int:
    return sum(1 for e in events if e.get("type") == "tool.finished"
               and isinstance(e.get("data"), dict) and e["data"].get("status") not in (None, "success"))  # fmt: skip


def checkpoints(events: Iterable[dict[str, Any]]) -> int:
    """Notices of kind ``checkpoint``. Each one is one more model request (the notes) and a smaller prompt after it."""
    return sum(1 for e in events if e.get("type") == "notice"
               and isinstance(e.get("data"), dict) and e["data"].get("kind") == "checkpoint")  # fmt: skip


def _failed(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("error"))
        or row.get("status") not in (None, 200)
        or row.get("stream_complete") is False
    )


def _reasoning_tokens(row: dict[str, Any]) -> int | None:
    raw = row.get("usage_raw") if isinstance(row.get("usage_raw"), dict) else {}
    details = (
        raw.get("completion_tokens_details")
        if isinstance(raw.get("completion_tokens_details"), dict)
        else {}
    )
    value = details.get("reasoning_tokens", raw.get("reasoning_tokens"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def block(
    calls: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]] | None = None,
    wall_s: float | None = None,
) -> dict[str, Any]:
    """The fixed block (see the module's text). *wall_s* is the caller's measure; without it the records' span is used."""
    ordered = sorted(calls, key=lambda r: r.get("call_index") or 0)
    gen = [r for r in ordered if r.get("role") in GENERATION_ROLES]
    main = [r for r in gen if r.get("role") == "main"]
    prompts = [
        r["usage"]["prompt_tokens"]
        for r in main
        if isinstance(r.get("usage"), dict)
        and isinstance(r["usage"].get("prompt_tokens"), int)
    ]
    cached = sum(
        (r["usage"].get("cached_tokens") or 0)
        for r in main
        if isinstance(r.get("usage"), dict)
        and isinstance(r["usage"].get("prompt_tokens"), int)
    )
    reasoning = [t for t in (_reasoning_tokens(r) for r in gen) if t is not None]
    chars = [
        r["output_reasoning_chars"]
        for r in gen
        if isinstance(r.get("output_reasoning_chars"), int)
    ]
    if wall_s is None:
        if events:
            stamped = _stamped(events)
            start = next(
                (e["ts"] for e in stamped if e.get("type") == "turn.started"),
                stamped[0]["ts"] if stamped else None,
            )
            wall_s = None if start is None else stamped[-1]["ts"] - start
        else:
            spans = [
                (r["ts_start"], r["ts_end"])
                for r in gen
                if isinstance(r.get("ts_start"), (int, float))
                and isinstance(r.get("ts_end"), (int, float))
            ]
            wall_s = (
                max(e for _, e in spans) - min(s for s, _ in spans) if spans else None
            )
    has_events = events is not None
    out = {
        "wall_s": None if wall_s is None else round(wall_s, 1),
        "first_sign_s": first_sign_s(events) if has_events else None,
        "model_calls": len(gen),
        "failed_model_calls": sum(1 for r in gen if _failed(r)),
        "tool_calls": sum(1 for e in events if e.get("type") == "tool.started")
        if has_events
        else sum(r.get("tool_calls") or 0 for r in main),
        "repeat_calls": event_repeats(events) if has_events else call_repeats(main),
        "failed_tools": failed_tools(events) if has_events else None,
        "approvals_asked": sum(
            1 for e in events if e.get("type") == "approval.requested"
        )
        if has_events
        else None,
        "checkpoints": checkpoints(events) if has_events else None,
        "prompt_first": prompts[0] if prompts else None,
        "prompt_median": statistics.median(prompts) if prompts else None,
        "prompt_max": max(prompts) if prompts else None,
        "cached_share": round(cached / sum(prompts), 3)
        if prompts and sum(prompts)
        else None,
        "reasoning_tokens": sum(reasoning) if reasoning else None,
        "reasoning_chars": sum(chars) if chars else None,
    }
    assert tuple(out) == KEYS
    return out


def line(b: dict[str, Any]) -> str:
    """The block as one line of a summary (a figure that is None is left out, and so is a run with no checkpoint)."""

    def part(text: str, *keys: str) -> str | None:
        return None if any(b.get(k) is None for k in keys) else text.format(**b)

    parts = [
        part("wall {wall_s} s", "wall_s"),
        part("first sign {first_sign_s} s", "first_sign_s"),
        part(
            "{model_calls} model calls ({failed_model_calls} failed)",
            "model_calls",
            "failed_model_calls",
        ),
        part("{tool_calls} tool calls", "tool_calls"),
        part("{repeat_calls} repeated", "repeat_calls"),
        part("{failed_tools} failed tools", "failed_tools"),
        part("{approvals_asked} approvals", "approvals_asked"),
        part("{checkpoints} checkpoints", "checkpoints") if b.get("checkpoints") else None,
        part(
            "prompt {prompt_first} / {prompt_median:.0f} / {prompt_max} (first / median / max)",
            "prompt_first",
            "prompt_median",
            "prompt_max",
        ),
        None if b.get("cached_share") is None else f"cached {b['cached_share']:.0%}",
        part("{reasoning_tokens} thinking tokens", "reasoning_tokens"),
    ]
    return " · ".join(p for p in parts if p)
