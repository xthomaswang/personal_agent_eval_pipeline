"""RESULTS.md (DESIGN §7-8), rebuilt by `modeleval report` from the verified runs under the repository's
docs/benchmark/<family>/<quant>/backend/<run-id>/.

Rows come from the newest complete run per model and comparability key. A table never mixes machines
(the results path's first level) or protocol versions; speed tables also never mix macOS or Xcode builds,
and capability tables never mix harness epochs. Rows that do not share a key go into separate tables, each with its own reference row (or a
note that the reference must be re-run for that key). Harness epochs declared equivalent in pins.toml
(`[[epoch.equivalent]]`, DESIGN §6.5) are one epoch here: a capability table is keyed and titled by the
canonical id, and the footer names the equivalence once.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import paths, registry, speed, stats
from .capability import epoch as harness_epoch

OLD_CAMPAIGN = "../qwen3.8-27b_20261017-2258_benchmark"
BAND = {"m1": 0.20, "m3": 0.20, "m4": 0.09}  # measured cross-night spread of the old campaign (DESIGN §7)
HARNESSES = ("claude_code", "hermes", "localmelo")  # bench.harness BUILDERS names, in run order
HARNESS_NAMES = {"claude_code": "Claude Code", "hermes": "Hermes", "localmelo": "localmelo"}
SETS = ("gate", "ranking")
SET_TASKS = {"gate": 22, "ranking": 22}  # DESIGN §6.3: tasks per set, each run through the three harnesses
CONTEXTS = (("m3-16k", "16K"), ("m3-32k", "32K"), ("g5+m3-64k", "64K"), ("m3-110k", "110K"))
GIB = 2**30
MINOR_MARK = "‡"   # a unit kept with a minor disturbance flag (DESIGN §5.4); "†" is the TTFT-includes-decode mark
MINOR_NOTE = (
    "‡ Kept with a minor disturbance flag (DESIGN §5.4). Following the project owner's instruction of 2026-10-26 "
    "(small disturbances have little effect: skip them, no re-run), only a major disturbance repeats a unit: foreign CPU "
    ">= 300 % or foreign GPU time >= 500 ms per 5 s sample (10 %) for at least 6 consecutive samples (30 s) or in at "
    "least 25 % of the unit's samples, the lock, a peer's server port, power, a battery below 20 %, memory pressure "
    "from foreign growth, or a gap in the monitor. A minor flag (one foreign process above 25 % CPU, foreign CPU above "
    "150 % or foreign GPU time above 250 ms in two consecutive samples, including a shorter burst above the major "
    "lines, a build or download tool) keeps the result: in the old campaign's M8 one busy core moved "
    "M1 / M3 / M4 by +2.9 % / -0.9 % / +2.0 %, inside the ~10 % cross-run error the owner accepts.")


# ======================================================================= discovery
@dataclass
class Run:
    model_key: str
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    model: dict[str, Any]
    units: dict[str, Any] | None
    cap: dict[str, Any] | None
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def protocol(self) -> Any:
        for d in (self.manifest, self.units or {}, self.cap or {}):
            if d.get("protocol_version") is not None:
                return d["protocol_version"]
        return None

    @property
    def builds(self) -> tuple[str | None, str | None]:
        """(macOS build, Xcode build): part of the speed comparability key. From the manifest (run.py's
        `machine`), else units.json, else the first agentbench raw header of the speed leg. A run whose
        kept attempts ran on launches with different builds gets a "mixed ..." key of its own, so it
        never shares a table with either build."""
        used = self.attempt_builds()
        if len(used) > 1:
            return ("mixed " + "/".join(sorted(str(m) for m, _ in used)),
                    "mixed " + "/".join(sorted({str(x) for _, x in used})))
        m = self.manifest
        mac, xc = None, None
        for d in (m.get("machine"), m.get("system"), m):
            if isinstance(d, dict):
                mac = mac or _first(_dig(d, "sw_vers", "BuildVersion"), d.get("macos_build"))
                xc = xc or _first(d.get("xcode_build"), _xcode_build(d.get("xcodebuild")))
        usys = (self.units or {}).get("system") or {}
        mac, xc = mac or usys.get("macos_build"), xc or usys.get("xcode_build")
        if not (mac and xc):
            hsys = _raw_header_system(self.run_dir)
            mac, xc = mac or hsys.get("macos_build"), xc or hsys.get("xcode_build")
        return mac, xc

    def attempt_builds(self) -> set[tuple[Any, Any]]:
        """The (macOS, Xcode) builds of the launches the kept speed attempts ran on (run.py records
        `builds` per launch; older manifests have none)."""
        by_launch = {ln.get("launch"): ln.get("builds") for ln in self.manifest.get("launches") or []
                     if isinstance(ln, dict) and isinstance(ln.get("builds"), dict)}
        out = set()
        for unit in ((self.units or {}).get("units") or {}):
            kept, _ = unit_state(self, unit)
            b = by_launch.get((kept or {}).get("launch"))
            if b and b.get("macos"):
                out.add((b.get("macos"), b.get("xcode")))
        return out

    @property
    def epoch(self) -> Any:
        """The harness epoch the run recorded (compare through canonical_epoch, never directly)."""
        e = _first(self.manifest.get("epoch"), _dig(self.manifest, "pins", "epoch"), (self.cap or {}).get("epoch"))
        return e.get("id") if isinstance(e, dict) else e

    @property
    def order(self) -> tuple[str, str]:
        return (str(_first(self.manifest.get("started"), self.manifest.get("created"), "")), self.run_id)

    @property
    def machine(self) -> str:
        """The results path's machine level (paths.results_dir)."""
        return paths.run_parts(self.run_dir)[0]

    @property
    def is_reference(self) -> bool:
        return bool(self.model.get("reference"))

    @property
    def name(self) -> str:
        return str(self.model.get("display") or self.model_key)


def _first(*xs: Any) -> Any:
    return next((x for x in xs if x not in (None, "", {})), None)


def _dig(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _xcode_build(x: Any) -> str | None:
    if isinstance(x, dict):
        x = x.get("default")
    if not isinstance(x, str):
        return None
    import re

    m = re.search(r"Build version\s+(\S+)", x)
    return m.group(1) if m else None


def _raw_header_system(run_dir: Path) -> dict[str, Any]:
    for f in sorted((Path(run_dir) / "speed").glob("*.jsonl")):
        try:
            with f.open(encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
        except (OSError, json.JSONDecodeError):
            continue
        if first.get("type") == "header":
            return speed._system_from_header([first])
    return {}


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def cap_records(doc: Any) -> list[dict[str, Any]]:
    """capability/results.json: a list of records, or {"records" | "results" | "rows": [...]}."""
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    if isinstance(doc, dict):
        for k in ("records", "results", "rows"):
            if isinstance(doc.get(k), list):
                return [r for r in doc[k] if isinstance(r, dict)]
    return []


def run_dirs(results_root: Path) -> list[Path]:
    """This pipeline's run directories under results_root (docs/benchmark): .../backend/<run-id>/ (paths.
    results_dir) with a manifest.json or speed/units.json. Other suites and other pipelines' rows
    (summary.json only) are not runs of ours."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    found = {m.parent for m in root.rglob("manifest.json")}
    found |= {u.parent.parent for u in root.rglob("units.json") if u.parent.name == "speed"}
    return sorted(d for d in found if d.parent.name == paths.SUITE and d.parent != root
                  and not any(x.startswith(("_", ".")) for x in d.relative_to(root).parts))


def discover(results_root: Path) -> list[Run]:
    runs = []
    for rdir in run_dirs(results_root):
        manifest = _json(rdir / "manifest.json") or {}
        units = _json(speed.units_path(rdir))
        cap = _json(rdir / "capability" / "results.json")
        model = _first(*(d for d in (manifest.get("model"), (units or {}).get("model_info"))
                         if isinstance(d, dict))) or {}
        key = str(model.get("key") or (units or {}).get("model") or "/".join(rdir.parts[-4:-2]))
        model = {"key": key, **model}
        runs.append(Run(key, rdir.name, rdir, manifest, dict(model), units,
                        cap if isinstance(cap, dict) else ({"records": cap} if isinstance(cap, list) else None),
                        cap_records(cap)))
    return runs


def epoch_map() -> dict[str, str]:
    """Declared-equivalent harness epoch -> its canonical id (pins.toml `[[epoch.equivalent]]`, DESIGN §6.5)."""
    return harness_epoch.canonical_map(registry.load_pins())


def canonical_epoch(epoch: Any, mapping: dict[str, str] | None = None) -> Any:
    """The id a run's harness epoch is compared and shown under: the oldest of its declared chain."""
    if epoch is None:
        return None
    return (epoch_map() if mapping is None else mapping).get(str(epoch), epoch)


# ======================================================================= completeness and selection
def unit_state(run: Run, unit: str) -> tuple[dict | None, str]:
    """(kept attempt, state): "ok" (clean, or kept with a minor flag: `minor_kept`); "disturbed xN"
    (every allowed attempt had a major disturbance) or "invalid" (a cold request hit the cache), where
    the metric stays empty; or "missing" (not finished)."""
    rec = speed.unit_record({"units": dict(((run.units or {}).get("units") or {}))}, unit)
    kept = speed.kept_attempt(rec, run.run_dir)
    if kept:
        return kept, "ok"
    v = rec.get("verdict")
    if isinstance(v, str) and v not in speed.KEPT_VERDICTS:
        return None, v
    if len(rec["attempts"]) >= 3:
        return None, f"disturbed x{len(rec['attempts'])}"
    return None, "missing"


def functional(run: Run) -> dict[str, Any] | None:
    return speed.functional_result(run.units)


def speed_complete(run: Run) -> bool:
    flag = _dig(run.manifest, "legs", "speed", "complete")
    if isinstance(flag, bool):
        return flag
    f = functional(run)
    if f is None or f.get("ok") is None:   # None: the server was lost in the block, not a verdict
        return False
    if not f.get("ok"):
        return True  # a failed gate is a complete (short) result
    model = SimpleNamespace(spec=run.model.get("spec", "mtp"))
    return all(unit_state(run, u)[1] != "missing" for u in speed.timed_units(model))


def cap_complete(run: Run, set_name: str) -> bool:
    """Explicit flags win (results.json `complete_sets`, or the manifest's capability leg); a finished
    leg (`finished: true`) is complete for its sets, even with pairs excluded as invalid twice; a leg
    that yielded counts per set: every planned (harness, task) pair, or the set's fixed size
    (DESIGN §6.3), needs a valid verdict."""
    doc = run.cap or {}
    flag = _first(_dig(doc, "complete_sets", set_name), _dig(run.manifest, "legs", "capability", "complete"))
    if isinstance(flag, bool):
        return flag
    if doc.get("finished") is True and set_name in (doc.get("sets") or SETS):
        return any(stats.rec_set(r) == set_name for r in run.records)
    have = {(stats.rec_harness(r), stats.rec_task(r)) for r in stats.final_records(run.records, set_name)}
    planned = [p for p in (doc.get("planned") or []) if isinstance(p, dict) and stats.rec_set(p) == set_name]
    if planned:
        return all((stats.rec_harness(p), stats.rec_task(p)) in have for p in planned)
    return len(have) >= SET_TASKS[set_name] * len(HARNESSES)


def newest(runs: list[Run]) -> list[Run]:
    """The newest run per model."""
    best: dict[str, Run] = {}
    for r in runs:
        if r.model_key not in best or r.order > best[r.model_key].order:
            best[r.model_key] = r
    return sorted(best.values(), key=lambda r: (not r.is_reference, r.name))


def speed_groups(runs: list[Run]) -> list[tuple[tuple, list[Run]]]:
    groups: dict[tuple, list[Run]] = {}
    for r in runs:
        if r.units is not None and speed_complete(r):
            groups.setdefault((r.machine, r.protocol, *r.builds), []).append(r)
    out = [(k, newest(v)) for k, v in groups.items()]
    return sorted(out, key=lambda kv: (not any(r.is_reference for r in kv[1]), str(kv[0])))


def cap_groups(runs: list[Run], mapping: dict[str, str] | None = None) -> list[tuple[tuple, list[Run]]]:
    """Capability tables by (machine, protocol, canonical harness epoch, set)."""
    mapping = epoch_map() if mapping is None else mapping
    groups: dict[tuple, list[Run]] = {}
    for r in runs:
        for s in SETS:
            if r.records and cap_complete(r, s):
                groups.setdefault((r.machine, r.protocol, canonical_epoch(r.epoch, mapping), s), []).append(r)
    out = [(k, newest(v)) for k, v in groups.items()]
    return sorted(out, key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]), SETS.index(kv[0][3])))


# ======================================================================= formatting
def _f(x: Any, nd: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _gib(b: Any) -> str:
    return "-" if b is None else f"{b / GIB:.1f}"


def _ratio(new: float | None, ref: float | None, band: float) -> str:
    """new / ref; a ratio inside the cross-night band is marked as no difference."""
    if new is None or not ref:
        return ""
    r = new / ref
    return f" ({r:.2f}x{' ≈' if abs(r - 1) <= band else ''})"


def table(head: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
                     + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def _row_name(run: Run) -> str:
    return f"{run.name}{' (reference)' if run.is_reference else ''}"


def _run_ref(run: Run) -> str:
    return f"`{run.model_key}/{run.run_id}`"


# ======================================================================= speed tables
def _metric(run: Run, unit: str, *path: str) -> tuple[Any, str]:
    kept, state = unit_state(run, unit)
    if kept is None:
        return None, state
    return _dig(kept.get("metrics"), *path), state


def minor_kept(run: Run, unit: str) -> bool:
    """The unit's kept attempt was judged minor: kept with its flags, not repeated (DESIGN §5.4)."""
    kept, _ = unit_state(run, unit)
    return bool(kept) and (kept.get("verdict") == "minor" or kept.get("severity") == "minor")


def _minor_mark(run: Run, *units: str) -> str:
    return MINOR_MARK if any(minor_kept(run, u) for u in units) else ""


def _cell(run: Run, unit: str, path: tuple[str, ...], nd: int = 1, ref: Run | None = None, band: float = 0.0) -> str:
    v, state = _metric(run, unit, *path)
    if v is None:
        return state if state != "missing" else "-"  # "disturbed xN" or "invalid" (a cold request hit the cache)
    out = _f(v, nd) + _minor_mark(run, unit)
    if ref is not None and ref is not run:
        rv, _ = _metric(ref, unit, *path)
        out += _ratio(v, rv, band)
    return out


def m4_summary(run: Run) -> dict[str, Any]:
    """Median over the kept M4 runs (DESIGN §5.1: 3 runs, median reported)."""
    ms = [unit_state(run, f"m4-{i}") for i in (1, 2, 3)]
    kept = [k["metrics"]["m4"] for k, _ in ms if k]

    def med(key: str) -> float | None:
        v = [m[key] for m in kept if m.get(key) is not None]
        return statistics.median(v) if v else None

    return {"runs": len(kept), "states": [s for _, s in ms], "headline_s": med("headline_s"),
            "ttft_sum_s": med("ttft_sum_s"), "raw_total_s": med("raw_total_s"),
            "mean_completion_tokens": med("mean_completion_tokens"),
            "cap_hits": [m.get("cap_hits") for m in kept], "steps": [m.get("steps") for m in kept]}


def memory(run: Run) -> dict[str, Any]:
    mem = (run.units or {}).get("memory")
    return mem if mem else speed.memory_summary(run.run_dir, run.units)


def _decode_mark(run: Run, unit: str, key: str) -> str:
    """"†" when the unit's kept attempt has a TTFT that includes a tool call's decode (no server TTFT)."""
    v, _ = _metric(run, unit, key, "ttft_includes_decode")
    return "†" if v else ""


def _gate_cell(f: dict[str, Any] | None, gid: str) -> str:
    if f is None:
        return "-"
    it = (f.get("items") or {}).get(gid)
    if not it:
        return "not run" if f.get("stop_reason") else "-"
    return str(it.get("result"))


def speed_section(key: tuple, rows: list[Run]) -> str:
    machine, proto, mac, xc = key
    ref = next((r for r in rows if r.is_reference), None)
    out = [f"## Speed on {machine} (protocol {proto}, macOS build {mac or '?'}, Xcode build {xc or '?'})"]
    if str(mac).startswith("mixed"):
        out.append("These runs' kept attempts ran on different builds (a resume across an update): they are "
                   "compared with nothing; re-run them on one build.")
    if ref is None:
        out.append("No reference run with this protocol version and these builds: re-run the reference speed leg "
                   "(DESIGN §2) before comparing these rows.")
    ok_rows = [r for r in rows if (functional(r) or {}).get("ok")]
    # functional block and M7
    frows = []
    for r in rows:
        f = functional(r)
        m7 = (f or {}).get("m7") or {}
        vision = r.model.get("vision")
        m7cell = (f"error after {m7.get('trials')} trials" if m7.get("error") else
                  f"{m7.get('passed_cases')}/{m7.get('scored_cases')}"
                  + (f" ({len(m7['incomplete'])} incomplete)" if m7.get("incomplete") else "")) if m7 else "-"
        frows.append([_row_name(r), _run_ref(r), _gate_cell(f, "G0"), _gate_cell(f, "G1"), _gate_cell(f, "G3"),
                      _gate_cell(f, "G4"),
                      (f"{_gate_cell(f, 'G2')} / {_gate_cell(f, 'G2b')}" if vision else "vision: none"),
                      m7cell, ", ".join(m7.get("failing") or []) or "-",
                      "server lost: " + str(f.get("server_lost")) if f and f.get("server_lost") else
                      "stopped: " + str(f.get("stop_reason")) if f and not f.get("ok") else "ok"])
    out.append("### Functional gate and M7\n\n" + table(
        ["Model", "Run", "G0", "G1", "G3", "G4", "G2 / G2b", "M7 cases passed", "M7 failing cases", "Run state"], frows))
    if not ok_rows:
        return "\n\n".join(out)
    # latency and memory
    lrows = []
    for r in ok_rows:
        mem = memory(r)
        press = mem.get("max_pressure")
        swap = mem.get("swapouts")
        lrows.append([_row_name(r), _cell(r, "g5+m3-64k", ("g5", "ttft_s"), 1),
                      f"{_gib(mem.get('after_load'))} / {_gib(mem.get('g5_peak'))} / {_gib(mem.get('m3_110k_peak'))} / "
                      f"{_gib(mem.get('peak'))}",
                      _gib(mem.get("peak_system")),
                      f"{_f(press, 0)} / {_f(swap, 0)}",
                      _cell(r, "m1", ("m1", "warm_ttft_s", "median"), 3, ref, BAND["m1"]) + _decode_mark(r, "m1", "m1"),
                      _cell(r, "m1", ("m1", "cold_ttft_s", "median"), 2)])
    marks = [r for r in ok_rows if _decode_mark(r, "m1", "m1")]
    out.append("### Latency and memory\n\n" + table(
        ["Model", "G5 64K cold TTFT s", "Server footprint GiB: after load / G5 peak / 110K peak / overall peak",
         "System delta peak GiB", "Max pressure / swap-outs", "M1 warm TTFT s", "M1 cold TTFT s"], lrows)
               + ("\n\n† A trial opened with a tool call and the server reported no time to first token: that "
                  "TTFT includes the call's decode." if marks else ""))
    # decode curve
    drows, srows = [], []
    for r in ok_rows:
        cells = [_cell(r, u, ("m3", "decode_tps", "median"), 1, ref, BAND["m3"]) for u, _ in CONTEXTS]
        secs = [_cell(r, u, ("m3", "s_per_100"), 2) for u, _ in CONTEXTS]
        if r.model.get("spec") == "none":
            think = "= 16K (no spec)"
            think_s = "-"
        else:
            think_ref = ref if ref and ref.model.get("spec") != "none" else None
            think = _cell(r, "m3-think-16k", ("m3", "decode_tps", "median"), 1, think_ref, BAND["m3"])
            think_s = _cell(r, "m3-think-16k", ("m3", "s_per_100"), 2)
        drows.append([_row_name(r)] + cells + [think])
        srows.append([_row_name(r)] + secs + [think_s])
    head = ["Model"] + [c for _, c in CONTEXTS] + ["thinking 16K"]
    out.append("### Decode (M3, thinking off, cap 150, median of 3 trials)\n\nTokens per second by context; "
               "the ratio is to the reference.\n\n" + table(head, drows))
    out.append("Seconds per 100 output tokens:\n\n" + table(head, srows))
    # M4
    mrows, notes = [], []
    ref_m4 = m4_summary(ref) if ref else None
    for r in ok_rows:
        s = m4_summary(r)
        disturbed = [x for x in s["states"] if x.startswith("disturbed")]
        head_cell = _f(s["headline_s"], 1)
        if ref_m4 and r is not ref:
            head_cell += _ratio(s["headline_s"], ref_m4["headline_s"], BAND["m4"])
        caps = ", ".join(f"{c}/{n}" for c, n in zip(s["cap_hits"], s["steps"])) or "-"
        mark = "".join(_decode_mark(r, f"m4-{i}", "m4") for i in (1, 2, 3))[:1]
        if s["runs"] and s["headline_s"] is None:
            notes.append(f"{r.name}: no kept M4 run has a decode time (no step with server usage and a decode "
                         "span), so R and the headline are empty.")
        if mark:
            notes.append(f"{r.name}: † some steps opened with a tool call and the server reported no time to first "
                         "token; their TTFT includes the call's decode.")
        mrows.append([_row_name(r), f"{s['runs']}/3" + _minor_mark(r, "m4-1", "m4-2", "m4-3")
                      + (f" ({', '.join(disturbed)})" if disturbed else ""), head_cell + mark,
                      _f(s["ttft_sum_s"], 1) + mark, _f(s["raw_total_s"], 1), _f(s["mean_completion_tokens"], 1), caps])
    out.append("### M4 20-step trajectory (median of the kept runs)\n\n" + table(
        ["Model", "Runs kept", "Headline s", "TTFT sum s", "Raw total s", "Mean completion tok/step", "Cap hits"], mrows)
        + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else ""))
    flagged = [(r, [u for u in speed.timed_units(SimpleNamespace(spec=r.model.get("spec", "mtp"))) if minor_kept(r, u)])
               for r in ok_rows]
    flagged = [(r, units) for r, units in flagged if units]
    if flagged:
        out.append(MINOR_NOTE + " Units kept with a minor flag: "
                   + "; ".join(f"{r.name}: {', '.join(units)}" for r, units in flagged) + ".")
    return "\n\n".join(out)


# ======================================================================= capability tables
def _pair(tts: dict[str, Any], key: str) -> str:
    ratio = tts[f"{key}_ratio"]
    return f"{_f(tts[f'{key}_new'], 0)} / {_f(tts[f'{key}_ref'], 0)}" + (f" ({ratio:.2f}x)" if ratio else "")


def capability_section(key: tuple, rows: list[Run]) -> str:
    machine, proto, epoch, set_name = key
    ref = next((r for r in rows if r.is_reference), None)
    n_tasks = SET_TASKS[set_name]
    out = [f"## Capability on {machine}: {set_name} set (protocol {proto}, harness epoch {epoch or '?'})"]
    if ref is None:
        out.append("No reference run in this epoch: the comparison needs the reference's capability leg on the same "
                   "harness epoch (DESIGN §7).")
    present = {str(stats.rec_harness(x)) for r in rows for x in stats.final_records(r.records, set_name)}
    harnesses = [h for h in HARNESSES if h in present or not present] + sorted(present - set(HARNESSES))
    hrows = []
    for r in rows:
        hc = stats.harness_counts(r.records, set_name)
        recs = stats.final_records(r.records, set_name)
        kinds: dict[str, int] = {}
        for x in recs:
            for k in stats.rec_anomalies(x):
                kinds[k] = kinds.get(k, 0) + 1
        cells = []
        for h in harnesses:
            d = hc.get(h)
            if not d:
                cells.append("-")
                continue
            lo, hi = d["wilson"]
            cells.append(f"{d['solved']}/{d['runs']} ({lo * 100:.0f}-{hi * 100:.0f} %)" if lo is not None else "-")
        solved = sum(d["solved"] for d in hc.values())
        fast = sum(d["fast"] for d in hc.values())
        runs_n = sum(d["runs"] for d in hc.values())
        anomalies = sum(d["anomalies"] for d in hc.values())
        kinds_s = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        hrows.append([_row_name(r), _run_ref(r)] + cells + [f"{solved}/{runs_n}", f"{fast}/{runs_n}",
                                                            f"{anomalies}" + (f" ({kinds_s})" if kinds_s else "")])
    head = (["Model", "Run"] + [HARNESS_NAMES.get(h, h) for h in harnesses]
            + ["Runs solved", "Solved within 1200 s", "Anomalies"])
    out.append("### Solved per harness\n\nCounts of runs; Wilson 95 % intervals are for display only (the three "
               "harnesses are correlated, so the decision below is per task).\n\n" + table(head, hrows))
    if ref is None:
        return "\n\n".join(out)
    crows, drows, trows = [], [], []
    for r in rows:
        if r is ref:
            continue
        c = stats.compare(r.records, ref.records, set_name=set_name)
        md = c["mdd"]
        if set_name == "gate":
            share = md["gate_margin"] / max(c["tasks"], 1) * 100
            mdd_s = f"rule: net loss >= {md['gate_margin']} of {c['tasks']} tasks ({share:.0f} %)"
        else:
            at = md["margin_at_discordant"]
            mdd_s = (f"net >= {at} of {md['discordant']} differing tasks" if at is not None
                     else f"none with {md['discordant']} differing tasks (needs >= {md['min_discordant']})")
            mdd_s += f"; all {md['n_tasks']} differing: net >= {md['margin_all_tasks']}"
        harness_net = ", ".join(f"{h} {v['net']:+d}" for h, v in (c["per_harness"] or {}).items())
        decision = f"**{c['decision']}**" + (f" (provisional: {len(c['excluded'])} pairs excluded)"
                                             if c["provisional"] else "")
        crows.append([_row_name(r), f"{c['tasks']} ({c['pairs']} pairs)", f"{c['score_new']} vs {c['score_ref']}",
                      str(c["wins"]), str(c["losses"]), str(c["ties"]), f"{c['p']:.3f}", harness_net or "-", decision,
                      mdd_s])
        diffs = [f"{t} {d:+d}" for t, d in c["diffs"].items() if d]
        drows.append(f"- {r.name}: " + (", ".join(diffs) if diffs else "no task differs"))
        if c["excluded"]:
            drows.append(f"  - excluded pairs (scored on neither side): {'; '.join(c['excluded'])}")
        for side, tasks in (("only the new model", c["only_new"]), ("only the reference", c["only_ref"])):
            if tasks:
                drows.append(f"  - tasks with valid pairs for {side}: {', '.join(tasks)}")
        tts = stats.time_to_solution(r.records, ref.records, set_name)
        trows.append([_row_name(r), str(tts["pairs"]), _pair(tts, "wall"), _pair(tts, "model"),
                      f"{_f(tts['steps_new'], 1)} / {_f(tts['steps_ref'], 1)}"])
    if crows:
        c0 = stats.decide(set_name, 0, 0, 1.0, None)[1]
        out.append(f"### Against the reference ({ref.name})\n\nPer task, a model's score is the number of harnesses "
                   f"that solved it (0-3); {n_tasks} tasks. Only the (harness, task) pairs with a valid result for both models "
                   f"are scored; a decision with excluded pairs is provisional. Decision rule, fixed in advance: {c0}.\n\n" + table(
                       ["Model", "Tasks", "Score new vs ref", "Ahead", "Behind", "Tied", "Sign-test p",
                        "Net per harness", "Decision", "Minimum detectable difference"], crows))
        out.append("Per-task differences (new minus reference, harnesses solved):\n\n" + "\n".join(drows))
        out.append("### Time to solution (pairs both models solved)\n\n" + table(
            ["Model", "Pairs", "Wall s new / ref", "Model s new / ref", "Steps per pair new / ref"], trows))
    return "\n\n".join(out)


# ======================================================================= footer
def equivalence_note(shown: set, raw: dict | None = None) -> str:
    """The one footnote on declared-equivalent harness epochs whose canonical id has a table; "" if none."""
    raw = registry.load_pins() if raw is None else raw
    decls = {d["new"]: d for d in harness_epoch.equivalences(raw)}
    mapping = harness_epoch.canonical_map(raw)
    named = [d for new, d in decls.items() if mapping[new] in {str(e) for e in shown}]
    if not named:
        return ""
    parts = [f"{d['new']} counts as {mapping[d['new']]} ({d['reason']})" for d in named]
    return ("- Declared-equivalent harness epochs (pins.toml `[[epoch.equivalent]]`, DESIGN §6.5; `modeleval "
            "selftest` checks each against git): " + "; ".join(parts) + ". Their runs share the tables of the "
            "canonical id named in each heading.")


def footer(runs: list[Run], protocols: set, epochs: set) -> str:
    lines = ["## Setting and notes", ""]
    models: dict[str, dict[str, Any]] = {}
    for r in runs:
        models.setdefault(r.model_key, r.model)
    for key, m in sorted(models.items(), key=lambda kv: (not kv[1].get("reference"), kv[0])):
        s = m.get("sampling") or {}
        src = m.get("source") or {}
        made = src.get("made_by") or ""
        repo = src.get("hf_repo") or ""
        rev = str(src.get("revision") or "")[:10]
        origin = ", ".join(x for x in (f"{repo}@{rev}" if repo else "", made) if x)
        lines.append(f"- **{m.get('display') or key}**: quantization {m.get('quant') or '?'}"
                     + (f" ({origin})" if origin else "") + f"; speculative decoding {m.get('spec') or '?'}; "
                     f"sampling temperature {s.get('temperature', '?')}, top_p {s.get('top_p', '?')}, "
                     f"top_k {s.get('top_k', '?')}" + ("" if m.get("vision") else "; vision: none"))
        for note in m.get("notes") or []:
            lines.append(f"  - {note}")
    lines += [
        "- Thinking is off (`enable_thinking: false`, no `reasoning_effort`) in every request of both legs, except the "
        "M3 thinking cell of models with speculative decoding, which sends `enable_thinking: true` only, so each "
        "template renders its own default effort. Without speculative decoding the thinking cell would repeat the 16K "
        "decode rate and is not run.",
        "- Text-only: every screenshot is replaced by its text equivalent of equal token size (Qwen3.8 tokenizer), "
        "for every model.",
        "- M4 headline = sum of TTFT over the 21 steps + 21 x 100 / R, where R is the run's decode rate (completion "
        "tokens / decode time over steps with server usage). The raw total mixes byte-identical prefill with each "
        "model's own output length, so a terser model would win it regardless of speed. Cap hits count "
        "`finish_reason=length` and streams cut inside a tool call (`incomplete_tool_call`), which still count as "
        "latency samples.",
        "- Decode tok/s counts each model's own tokens. M3's output is a JSON array, where token counts are about "
        "equal across tokenizers; seconds per 100 tokens are given for the same reason. Prompt sizes are Qwen3.8 "
        "tokens (\"16K\" = 16,000); every result records the server's own `prompt_tokens`.",
        "- Memory: server footprint of the oMLX process tree (`footprint`, RSS fallback); the system delta is "
        "wired + compressed + file-backed memory against the pre-launch baseline. File-backed pages are the page "
        "cache: mmapped weights appear in the delta only if they were not already cached before the launch, so the "
        "delta is an indication, not a measure of mmapped memory.",
        f"- Cross-night band (old campaign): M1 and M3 up to ~{BAND['m3'] * 100:.0f} %, M4 ~{BAND['m4'] * 100:.0f} %. "
        "A ratio marked ≈ lies inside it and counts as no difference.",
        "- Capability: solved = the checker passes within the 30-step budget (also after a wall timeout); runs "
        "excluded only for pipeline mismatches (model id, forwarded sampling or thinking). Anomalies (5xx, a "
        "reasoning leak with thinking off, a template fallback) are attributed to the model, never excluded, and "
        "counted per model; they do not change the checker's verdict.",
        f"- Protocol version(s): {', '.join(str(p) for p in sorted(protocols, key=str)) or '-'} (current: "
        f"{paths.PROTOCOL_VERSION}); harness epoch(s): {', '.join(str(e) for e in sorted(epochs, key=str)) or '-'}.",
        *([note] if (note := equivalence_note(epochs)) else []),
        f"- Earlier measurements: the Qwen3.8-27B serving campaign (`{OLD_CAMPAIGN}`, 2026-10-17..22) is not "
        "comparable with these tables: it used image-mode M1/M4, temperature 0 for some rows, three M4 runs with "
        "whole-run exclusion of cut steps, and an older localmelo (`5c009a8-dirty`).",
    ]
    return "\n".join(lines)


# ======================================================================= entry points
def build_report(results_root: Path = paths.RESULTS) -> str:
    runs = discover(results_root)
    parts = ["# Results", "",
             "Generated by `uv run modeleval report` from the runs under `docs/benchmark/<machine>/<family>/<quant>/backend/` "
             "(at the repository root); do not edit by hand. Method: "
             "[DESIGN.md](DESIGN.md)."]
    sg = speed_groups(runs)
    cg = cap_groups(runs)
    if not sg and not cg:
        parts.append("\nNo complete run yet.")
    for key, rows in sg:
        parts.append("\n" + speed_section(key, rows))
    for key, rows in cg:
        parts.append("\n" + capability_section(key, rows))
    used = {(r.model_key, r.run_id) for _, rows in sg + cg for r in rows}
    shown = [r for r in runs if (r.model_key, r.run_id) in used]
    protocols = {k[1] for k, _ in sg} | {k[1] for k, _ in cg}
    epochs = {k[2] for k, _ in cg}
    parts.append("\n" + footer(shown, protocols, epochs))
    return "\n".join(parts).rstrip() + "\n"


def write(results_root: Path = paths.RESULTS, out: Path = paths.REPO / "RESULTS.md") -> Path:
    out.write_text(build_report(results_root), encoding="utf-8")
    return out
