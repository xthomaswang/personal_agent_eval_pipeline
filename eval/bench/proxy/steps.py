"""Step counts of a run, computed from the proxy's calls.jsonl.

Definitions (DESIGN 2.2): a step is a generation request (/v1/messages or
/v1/chat/completions) that started between t0 and t_end, has role=main and
succeeded (HTTP 200 and a complete response). The proxy's step budget uses
the same definition, so these helpers and --max-steps always agree.

All times are time.monotonic() seconds, as in the records' mono_* fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .request_info import GENERATION_ENDPOINTS


def load_calls(path: Path) -> list[dict[str, Any]]:
    """Read calls.jsonl (a run directory or the file itself)."""
    path = Path(path)
    if path.is_dir():
        path = path / "calls.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_generation(record: dict[str, Any]) -> bool:
    return record.get("endpoint") in GENERATION_ENDPOINTS


def is_ok(record: dict[str, Any]) -> bool:
    """HTTP 200, no error, and for a stream the end marker was seen."""
    return record.get("status") == 200 and record.get("error") is None and record.get("stream_complete") is not False


def counts_as_step(record: dict[str, Any]) -> bool:
    """Whether a call counts toward n_llm (ignoring the time window)."""
    return is_generation(record) and record.get("role") == "main" and is_ok(record)


def _started_in(record: dict[str, Any], t0: float | None, t_end: float | None) -> bool:
    start = record.get("mono_start")
    if start is None:
        return False
    return (t0 is None or start >= t0) and (t_end is None or start <= t_end)


def step_counts(records: Iterable[dict[str, Any]], t0: float | None = None, t_end: float | None = None) -> dict[str, int]:
    """n_llm and its companions for the generation requests of a run.

    * ``n_llm``: role=main, succeeded, started in [t0, t_end].
    * ``n_llm_aux``: role=aux, succeeded, started in [t0, t_end].
    * ``n_llm_err``: failed (not HTTP 200 or incomplete), any role, started in
      [t0, t_end]; this includes calls refused by the step budget.
    * ``n_llm_post_submit``: like n_llm but started after t_end.
    * ``n_tool``: tool calls in the responses of the n_llm requests.
    * ``images_new_sum``: images seen for the first time, over the generation
      requests started in [t0, t_end].

    Together n_llm + n_llm_aux + n_llm_err cover every generation request in
    the window exactly once.
    """
    counts = {"n_llm": 0, "n_llm_aux": 0, "n_llm_err": 0, "n_llm_post_submit": 0, "n_tool": 0, "images_new_sum": 0}
    for record in records:
        if not is_generation(record):
            continue
        if _started_in(record, t0, t_end):
            counts["images_new_sum"] += record.get("images_new") or 0
            if not is_ok(record):
                counts["n_llm_err"] += 1
            elif record.get("role") == "main":
                counts["n_llm"] += 1
                counts["n_tool"] += record.get("tool_calls") or 0
            else:
                counts["n_llm_aux"] += 1
        elif t_end is not None and (record.get("mono_start") or 0) > t_end and counts_as_step(record):
            counts["n_llm_post_submit"] += 1
    return counts


def new_image_long_sides(records: Iterable[dict[str, Any]], t0: float | None = None, t_end: float | None = None) -> list[int | None]:
    """Long sides of every image first seen in the window (for the GUI check:
    at least one, and all equal to 1280)."""
    sides: list[int | None] = []
    for record in records:
        if is_generation(record) and _started_in(record, t0, t_end):
            sides.extend(record.get("images_new_long_side") or [])
    return sides
