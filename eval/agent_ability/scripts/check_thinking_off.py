"""Thinking-off check over a smoke run: did any model call on either endpoint return reasoning?

Usage: python3 -I check_thinking_off.py <harness-smoke run dir>
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
stats: dict = collections.defaultdict(
    lambda: {"calls": 0, "rewritten": 0, "reasoning_calls": 0, "reasoning_chars": 0, "reasoning_tokens": 0,
             "modes": collections.Counter(), "sampling": collections.Counter(), "errors": 0}
)
for f in sorted(root.rglob("calls.jsonl")):
    harness = f.relative_to(root).parts[0]
    for line in f.open():
        r = json.loads(line)
        if r.get("handled_locally"):
            continue
        s = stats[(harness, r.get("endpoint"))]
        s["calls"] += 1
        s["rewritten"] += bool(r.get("thinking_rewritten"))
        chars = r.get("output_reasoning_chars") or 0
        s["reasoning_chars"] += chars
        s["reasoning_calls"] += chars > 0
        usage = r.get("usage") or {}
        s["reasoning_tokens"] += ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        s["modes"][r.get("thinking_mode")] += 1
        s["sampling"][json.dumps(r.get("sampling_forwarded"), sort_keys=True)] += 1
        s["errors"] += bool(r.get("error"))

ok = True
for (harness, endpoint), s in sorted(stats.items()):
    print(f"{harness:12} {endpoint:18} calls={s['calls']:4} rewritten={s['rewritten']:4} "
          f"reasoning_calls={s['reasoning_calls']} reasoning_chars={s['reasoning_chars']} "
          f"reasoning_tokens={s['reasoning_tokens']} errors={s['errors']} modes={dict(s['modes'])}")
    for sampling, n in s["sampling"].items():
        print(f"{'':31} sampling {sampling} x{n}")
    ok &= s["reasoning_calls"] == 0 and s["reasoning_tokens"] == 0
print("thinking off on every call:", ok)
