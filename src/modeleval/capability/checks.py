"""Per-run checks of the proxy's calls.jsonl and the verdict mapping (DESIGN §6.4).

Two kinds of finding are kept apart on purpose:

- **Pipeline mismatches** make a run `invalid` (excluded from the score, re-run once): a call went to
  another model than the target's `served_id`, or the proxy forwarded other sampling or thinking
  settings than the registry and the arm ask for. These say nothing about the model.
- **Model or server faults** are that model's failures: the run stands and is flagged, and the flags
  add up to a per-model anomaly count. HTTP >= 500, reasoning text while thinking is off, upstream
  stream errors, a template error in the server log, a harness error.

The verdict says what happened to the run (harness-smoke's vocabulary plus `invalid` and
`server_lost`); `solved` is the score. They differ on purpose for a run that hit the wall clock:
its verdict is `timeout`, but the checker still ran, and if it passed the task counts as solved
(DESIGN §6.4). `solved_within_1200` keeps the reference-equivalent time budget visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GENERATION_ENDPOINTS = frozenset({"chat_completions", "messages"})
THINKING_ENDPOINTS = frozenset({"chat_completions", "messages", "count_tokens"})
# Top-level OpenAI thinking controls the proxy removes in off/on modes (bench/proxy/thinking.py)
OPENAI_THINKING_EXTRA = ("reasoning_effort", "reasoning", "thinking_budget")
BUDGET_ERROR = "bench_budget_exhausted"
UPSTREAM_ERROR_TYPES = frozenset({"upstream_unreachable", "upstream_timeout"})
# What a run outcome can be; only the first four are scored (valid)
VALID_VERDICTS = ("pass", "fail", "timeout", "harness_error")
VERDICTS = (*VALID_VERDICTS, "invalid", "server_lost")
SOLVED_WITHIN_S = 1200
EXAMPLES = 3


def load_calls(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(records, unreadable lines). A proxy killed mid-write can leave a truncated last line."""
    if not path.exists():
        return [], 0
    records, bad = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad += 1
    return records, bad


def thinking_ok(protocol: str | None, view: Any, mode: str) -> bool:
    """Whether a forwarded thinking view (record `thinking_forwarded`) is what `--thinking mode` writes."""
    if protocol not in ("openai", "anthropic"):
        return True
    if not isinstance(view, dict):
        return False
    kwargs = view.get("chat_template_kwargs")
    if not isinstance(kwargs, dict) or "reasoning_effort" in kwargs:
        return False
    want = {"off": False, "on": True}[mode]
    if kwargs.get("enable_thinking") is not want:
        return False
    if protocol == "openai":
        return not any(key in view for key in OPENAI_THINKING_EXTRA)
    return view.get("thinking") == {"type": "disabled" if mode == "off" else "enabled"} and "output_config" not in view


def check_calls(
    records: list[dict[str, Any]], *, served_id: str, sampling: dict[str, Any], thinking: str, unreadable: int = 0
) -> dict[str, Any]:
    """Counts of mismatches and faults over one run's calls, plus the model time.

    Calls the proxy answered itself (step budget refusals, Claude Code's /api/hello) never reached the
    server, so only their count is kept.
    """
    out: dict[str, Any] = {
        "calls": len(records),
        "generation_calls": 0,
        "model_mismatch": 0,
        "sampling_mismatch": 0,
        "thinking_mismatch": 0,
        "http_5xx": 0,
        "reasoning_leak": 0,
        "upstream_error": 0,
        "budget_refusals": 0,
        "unreadable_lines": unreadable,
        "models_seen": [],
        "model_s": 0.0,
        "examples": [],
    }
    models: set[str] = set()

    def note(kind: str, record: dict[str, Any], detail: str) -> None:
        out[kind] += 1
        if len(out["examples"]) < EXAMPLES:
            out["examples"].append(f"call {record.get('call_index')}: {kind}: {detail}")

    for record in records:
        endpoint = record.get("endpoint")
        error = record.get("error") if isinstance(record.get("error"), dict) else None
        if record.get("budget_exhausted") or (error and error.get("type") == BUDGET_ERROR):
            out["budget_refusals"] += 1
            continue
        if record.get("handled_locally"):
            continue
        status = record.get("status")
        # Only generation calls count: Hermes probes /api/v1/models, /api/tags, /v1/props, /props and
        # /version at every start and oMLX answers them 404, which says nothing about the model.
        if endpoint in GENERATION_ENDPOINTS:
            if isinstance(status, int) and status >= 500:
                note("http_5xx", record, f"HTTP {status}")
            if error and (error.get("source") == "upstream" or error.get("type") in UPSTREAM_ERROR_TYPES):
                note("upstream_error", record, str(error.get("type")))
        if endpoint in THINKING_ENDPOINTS and not thinking_ok(record.get("protocol"), record.get("thinking_forwarded"), thinking):
            note("thinking_mismatch", record, json.dumps(record.get("thinking_forwarded"), sort_keys=True))
        if endpoint not in GENERATION_ENDPOINTS:
            continue
        out["generation_calls"] += 1
        out["model_s"] += float(record.get("duration_s") or 0.0)
        model = record.get("model")
        if model is not None:
            models.add(str(model))
        if model != served_id and (model is not None or status == 200):
            note("model_mismatch", record, f"model {model!r}")
        if record.get("sampling_forwarded") != sampling:
            note("sampling_mismatch", record, json.dumps(record.get("sampling_forwarded"), sort_keys=True))
        if thinking == "off" and (record.get("output_reasoning_chars") or 0) > 0:
            note("reasoning_leak", record, f"{record.get('output_reasoning_chars')} reasoning chars")
    out["models_seen"] = sorted(models)
    out["model_s"] = round(out["model_s"], 3)
    return out


def invalid_reasons(calls: dict[str, Any]) -> list[str]:
    return [name for name in ("model", "sampling", "thinking") if calls[f"{name}_mismatch"]]


def no_model_call(record: dict[str, Any], calls: dict[str, Any]) -> bool:
    """A harness error before any generation request reached the proxy (the proxy did not start, the
    binary or its config is broken): the model never had a turn, so it is the pipeline's fault."""
    return record.get("verdict") == "harness_error" and not calls.get("generation_calls")


def outcome(
    record: dict[str, Any],
    calls: dict[str, Any],
    *,
    max_steps: int,
    server_lost: bool = False,
    template_errors: int = 0,
) -> dict[str, Any]:
    """Verdict, score and flags of one run, from smoke.runner.run_one's record and check_calls().

    - server_lost: the server died during the run (the orchestrator relaunches; the run is repeated);
    - invalid: a pipeline mismatch, or a harness error before any model call (excluded, re-run once);
    - timeout: bench.harness killed the harness at the wall clock, or localmelo stopped at its own
      `--timeout` (its JSON run record says status "timeout", exit code 0);
    - pass / fail / harness_error: harness-smoke's verdict otherwise. A harness_error after model calls
      is terminal: a harness that crashes on a model's output is that model's failure, and it is flagged.
    """
    report = record.get("harness_report") or {}
    run_record = report.get("run_record") if isinstance(report.get("run_record"), dict) else {}
    melo_status = run_record.get("status") if record.get("harness") == "localmelo" else None
    reasons = invalid_reasons(calls) + (["no_model_call"] if no_model_call(record, calls) else [])
    smoke_verdict = record.get("verdict")
    if server_lost:
        verdict = "server_lost"
    elif reasons:
        verdict = "invalid"
    elif smoke_verdict in ("pass", "fail") and melo_status == "timeout":
        verdict = "timeout"
    else:
        verdict = smoke_verdict
    valid = verdict in VALID_VERDICTS
    check_passed = bool((record.get("check") or {}).get("passed"))
    n_llm = record.get("n_llm") or 0
    solved = valid and check_passed and n_llm <= max_steps
    wall = record.get("wall_s")
    anomalies = [
        name for name in ("http_5xx", "reasoning_leak", "upstream_error") if calls.get(name)
    ]
    if template_errors:
        anomalies.append("template_error")
    if verdict == "harness_error":
        anomalies.append("harness_error")
    return {
        "verdict": verdict,
        "smoke_verdict": smoke_verdict,
        "valid": valid,
        "solved": solved,
        "solved_within_1200": solved and wall is not None and wall <= SOLVED_WITHIN_S,
        "invalid_reasons": reasons,
        "anomalies": anomalies,
        "model_s": calls.get("model_s"),
        "localmelo_status": melo_status,
        "template_errors": template_errors,
    }


def native_turns(harness: str, report: dict[str, Any]) -> int | None:
    """Each harness's own step count (Claude Code num_turns, Hermes API calls, localmelo steps)."""
    key = {"claude_code": "num_turns", "hermes": "api_call_count", "localmelo": "steps"}.get(harness)
    value = report.get(key) if key else None
    return value if isinstance(value, int) else None
