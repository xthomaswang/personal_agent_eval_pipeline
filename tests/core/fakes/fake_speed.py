"""Fake modeleval.speed with the docs/CONTRACTS.md interface, steered by a JSON control file.

$FAKE_SPEED_CONTROL (JSON): {"functional_pass": true, "units": [...], "disturb": {"m3-16k": 1},
  "invalid": {"m1": 1}, "incomplete": {"m4-1": 1}, "functional_kill": 1, "sleep": 0.4}
  disturb[u] = n: the first n attempts of u (in this process) touch $FAKE_QUIET_DISTURB while they
  run, so the fake monitor marks their window DISTURBED; invalid[u] = n: the first n attempts report
  cold_cached_tokens 4096; incomplete[u] = n: the first n attempts miss a latency sample (valid
  false, invalid_kind "incomplete"); functional_kill = n: the first n functional blocks SIGKILL the
  server and fail with a ConnectError, as a server lost mid-block. Every call is appended to
  $FAKE_SPEED_LOG.
$FAKE_SPEED_RECORD=1: also record functional and attempts into units.json with the real
modeleval.speed writers (record_functional, record_unit, finalize), as the real module does.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ALL_UNITS = ["g5+m3-64k", "m3-16k", "m3-32k", "m3-110k", "m3-think-16k", "m1", "m4-1", "m4-2", "m4-3"]
_seen: dict[str, int] = {}


@dataclass
class SpeedContext:
    model: object
    base_url: str
    server_pid: int
    run_dir: Path
    tokenizer_dir: Path
    unverified_dir: Path


@dataclass
class UnitResult:
    unit: str
    attempt: int
    t_start: float
    t_end: float
    raw_file: Path
    metrics: dict
    valid: bool
    note: str


@dataclass
class FunctionalResult:
    ok: bool
    stop_reason: str | None
    items: dict = field(default_factory=dict)
    files: list = field(default_factory=list)


def _real():
    if os.environ.get("FAKE_SPEED_RECORD") == "1":
        from modeleval import speed
        return speed
    return None


if _real() is not None:
    finalize = _real().finalize


def _control() -> dict:
    p = os.environ.get("FAKE_SPEED_CONTROL")
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


def _log(*parts: object) -> None:
    p = os.environ.get("FAKE_SPEED_LOG")
    if p:
        with open(p, "a") as f:
            f.write(" ".join(map(str, parts)) + "\n")


def timed_units(model) -> list[str]:
    units = _control().get("units") or ALL_UNITS
    return [u for u in units if u != "m3-think-16k" or model.spec != "none"]


async def run_functional(ctx: SpeedContext) -> FunctionalResult:
    async with httpx.AsyncClient() as c:   # the server must be up and serving
        r = await c.get(f"{ctx.base_url}/models")
    _log("functional", ctx.server_pid, r.status_code)
    n = _seen["functional"] = _seen.get("functional", 0) + 1
    if n <= _control().get("functional_kill", 0):
        os.kill(ctx.server_pid, signal.SIGKILL)
        return FunctionalResult(False, "error: ConnectError: All connection attempts failed", {})
    ok = bool(_control().get("functional_pass", True))
    fr = FunctionalResult(ok, None if ok else "G1 failed", {"G1": {"result": "PASS" if ok else "FAIL",
                                                                  "detail": "no tool call"}})
    if _real() is not None:
        _real().record_functional(ctx, fr)
    return fr


async def run_unit(ctx: SpeedContext, unit: str, attempt: int) -> UnitResult:
    ctl = _control()
    n = _seen[unit] = _seen.get(unit, 0) + 1
    disturb = n <= ctl.get("disturb", {}).get(unit, 0)
    flag = Path(os.environ["FAKE_QUIET_DISTURB"])
    t0 = time.time()
    if disturb:
        flag.touch()
    await asyncio.sleep(ctl.get("sleep", 0.4))
    flag.unlink(missing_ok=True)
    t1 = time.time()
    raw = Path(ctx.run_dir) / "speed" / f"{unit}-{attempt}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"type": "header", "unit": unit, "attempt": attempt}) + "\n")
    metrics = {"ttft_s": 1.0}
    if unit in ("m1", "m4-1", "m4-2", "m4-3"):
        metrics["cold_cached_tokens"] = 4096 if n <= ctl.get("invalid", {}).get(unit, 0) else 0
    valid, note = True, ""
    if n <= ctl.get("incomplete", {}).get(unit, 0):
        valid, note = False, f"{unit}: steps [7] have no latency sample"
        metrics["invalid_kind"] = "incomplete"
    _log("unit", unit, attempt, ctx.server_pid)
    res = UnitResult(unit, attempt, t0, t1, raw, metrics, valid, note)
    if _real() is not None:
        _real().record_unit(ctx, res)
    return res
