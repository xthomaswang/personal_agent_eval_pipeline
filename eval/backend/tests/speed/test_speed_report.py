"""report.py on synthetic result trees written through speed.py's own recorders and a synthetic
capability results.json."""

from __future__ import annotations

import json
from pathlib import Path

from speedkit import NANBEIGE, g5_line, header, m1_lines, m3_lines, m4_lines, make_model

from modeleval import report, speed

GIB = 2**30
GATE_TASKS = [f"aider/ex{i:02d}" for i in range(12)] + [f"mcpmark/fs/t{i}" for i in range(10)]
RANK_TASKS = [f"aider/rank{i:02d}" for i in range(22)]
HARNESSES = ("claude_code", "hermes", "localmelo")


def speed_run(root: Path, run_id: str, decode: float = 30.0, m1_warm: float = 0.30, m4_ttft: float = 1.0,
              build: str = "26B5101f", protocol: int = 1, epoch: str = "e1", skip: tuple[str, ...] = (),
              disturbed: tuple[str, ...] = (), minor: tuple[str, ...] = (), stop: str | None = None,
              **model_kw) -> speed.SpeedContext:
    model = make_model(**model_kw)
    run_dir = root / "mac" / model.key / "oq5" / "backend" / run_id   # <machine>/<family>/<quant>/backend/<run-id>
    ctx = speed.SpeedContext(model, "http://127.0.0.1:8790/v1", 1, run_dir, root / "tok", root / "_unverified")
    (run_dir / "speed").mkdir(parents=True)
    model_doc = speed.model_info(model)
    (run_dir / "manifest.json").write_text(json.dumps({"model": model_doc, "protocol_version": protocol, "epoch": epoch,
                                                       "started": run_id}))
    system = speed._system_from_header([header(build=build)])
    speed.record_functional(ctx, speed.FunctionalResult(
        stop is None, stop, {"G0": {"result": "PASS"}, "G1": {"result": "PASS"}, "G3": {"result": "PASS"},
                             "G4": {"result": "PASS"}},
        {"scored_cases": 22, "passed_cases": 21, "failing": ["m7-18-parallel"]}, [], 200.0, 250.0), system)
    if stop:
        return ctx
    lines = {"g5+m3-64k": [g5_line(t_send=300.0)] + m3_lines(decodes=(decode * 0.8,) * 3, ctx=64000),
             "m3-16k": m3_lines(decodes=(decode,) * 3), "m3-32k": m3_lines(decodes=(decode * 0.9,) * 3, ctx=32000),
             "m3-110k": m3_lines(decodes=(decode * 0.6,) * 3, ctx=110000),
             "m3-think-16k": m3_lines(decodes=(decode * 1.5,) * 3, cap=1000, thinking=True),
             "m1": m1_lines(warm=m1_warm), "m4-1": m4_lines(ttft=m4_ttft), "m4-2": m4_lines(ttft=m4_ttft * 1.1),
             "m4-3": m4_lines(ttft=m4_ttft * 0.9)}
    t = 300.0
    for unit in speed.timed_units(model):
        if unit in skip:
            continue
        metrics, valid, notes = speed.unit_metrics(unit, lines[unit])
        tries = 3 if unit in disturbed else 1
        for attempt in range(1, tries + 1):
            raw = run_dir / "speed" / f"{unit}-{attempt}.jsonl"
            raw.write_text("{}\n")
            speed.record_unit(ctx, speed.UnitResult(unit, attempt, t, t + 60, raw, metrics, valid, ""))
            clean = unit not in disturbed
            verdict = {"clean": clean, "reasons": [] if clean else ["mds"]}
            if unit in minor:   # kept with its flag (DESIGN §5.4), never repeated
                verdict = {"clean": False, "severity": "minor", "reasons": ["busy: mds(1) 40%"],
                           "minor_reasons": ["busy: mds(1) 40%"], "major_reasons": []}
            speed.record_verdict(run_dir, unit, attempt, verdict)
            t += 70
    (run_dir / "speed" / "memory.tsv").write_text(
        "t\ttree_bytes\tsys_delta_bytes\tpressure\tswapouts_delta\n"
        f"150\t{20 * GIB}\t{21 * GIB}\t1\t0\n320\t{30 * GIB}\t{35 * GIB}\t1\t0\n900\t{25 * GIB}\t{26 * GIB}\t1\t0\n")
    speed.finalize(run_dir)
    return ctx


def cap_run(root: Path, model_key: str, run_id: str, solved: dict[str, dict[str, bool]], epoch: str = "e1",
            sets: tuple[str, ...] = ("gate",), wall: float = 100.0, reference: bool = False, display: str = "",
            drop: int = 0) -> Path:
    run_dir = root / "mac" / model_key / "oq5" / "backend" / run_id
    (run_dir / "capability").mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        mpath.write_text(json.dumps({"model": {"key": model_key, "display": display or model_key, "reference": reference,
                                               "spec": "mtp", "sampling": {}}, "protocol_version": 1,
                                     "epoch": epoch, "started": run_id}))
    recs = []
    for s in sets:
        for task in (GATE_TASKS if s == "gate" else RANK_TASKS):
            for h in HARNESSES:
                ok = solved.get(task, {}).get(h, True)
                recs.append({"harness": h, "task": task, "set": s, "verdict": "pass" if ok else "fail",
                             "check": {"passed": ok}, "wall_s": wall, "model_s": wall * 0.8, "steps": 5,
                             "anomalies": ["5xx"] if (not ok and h == "hermes") else []})
    recs = recs[: len(recs) - drop] if drop else recs
    (run_dir / "capability" / "results.json").write_text(json.dumps({"epoch": epoch, "records": recs}))
    return run_dir


def test_empty_results_give_a_placeholder(tmp_path):
    md = report.build_report(tmp_path / "results")
    assert "No complete run yet." in md and "Earlier measurements" in md and md.startswith("# Results")


def test_speed_table_with_reference_ratios_and_bands(tmp_path):
    root = tmp_path / "results"
    speed_run(root, "20261027-010000", decode=30.0, m1_warm=0.30)
    speed_run(root, "20261027-030000", decode=20.0, m1_warm=0.33, m4_ttft=2.0, **NANBEIGE)
    md = report.build_report(root)
    assert md.count("## Speed on mac (protocol 1, macOS build 26B5101f, Xcode build 27A9275)") == 1
    assert "Qwen3.8-27B oQ5 + MTP (reference)" in md and "Nanbeige4.3-33B oQ5" in md
    assert "| Nanbeige4.3-33B oQ5 | 20.0 (0.67x) | 18.0 (0.67x) | 16.0 (0.67x) | 12.0 (0.67x) | = 16K (no spec) |" in md
    assert "0.330 (1.10x ≈)" in md  # M1 warm inside the 20 % band
    assert "45.0" in md  # reference thinking decode, 1.5 x 30
    assert "3.33" in md  # seconds per 100 tokens at 16K for the reference
    # memory: after load 20 GiB (sample before the functional block), G5 peak 30, overall 30 GiB
    assert "20.0 / 30.0 / - / 30.0" in md or "20.0 / 30.0 / 25.0 / 30.0" in md
    assert "m7-18-parallel" in md and "21/22" in md
    assert "vision: none" in md
    assert "/Users/" not in md and str(tmp_path) not in md
    assert report.MINOR_MARK not in md and "minor disturbance flag" not in md   # nothing was kept with a flag


def test_units_kept_with_a_minor_flag_are_marked(tmp_path):
    root = tmp_path / "results"
    speed_run(root, "20261027-010000", decode=30.0)
    speed_run(root, "20261027-030000", decode=20.0, minor=("m3-16k", "m1", "m4-2"), **NANBEIGE)
    md = report.build_report(root)
    assert "| Nanbeige4.3-33B oQ5 | 20.0‡ (0.67x) | 18.0 (0.67x) | 16.0 (0.67x) | 12.0 (0.67x) | = 16K (no spec) |" in md
    assert "0.330‡" not in md and "0.300‡ (1.00x ≈)" in md          # M1 warm TTFT
    assert "| 3/3‡ |" in md and md.count("| 3/3 |") == 1               # M4 runs kept; the reference has no flag
    assert md.count(report.MINOR_NOTE) == 1
    assert "Units kept with a minor flag: Nanbeige4.3-33B oQ5: m3-16k, m1, m4-2." in md
    nan = next(r for r in report.discover(root) if not r.is_reference)
    assert report.minor_kept(nan, "m1") and not report.minor_kept(nan, "m3-32k")
    assert report.unit_state(nan, "m3-16k")[1] == "ok"


def test_m4_headline_median_and_ratio(tmp_path):
    root = tmp_path / "results"
    ref = speed_run(root, "20261027-010000", m4_ttft=1.0)
    speed_run(root, "20261027-030000", m4_ttft=2.0, **NANBEIGE)
    s = report.m4_summary(report.discover(root)[1] if report.discover(root)[1].is_reference else report.discover(root)[0])
    assert s["runs"] == 3 and s["headline_s"] == speed.m4_metrics(m4_lines(ttft=1.0))[0]["headline_s"]
    md = report.build_report(root)
    head_new = speed.m4_metrics(m4_lines(ttft=2.0))[0]["headline_s"]
    head_ref = speed.m4_metrics(m4_lines(ttft=1.0))[0]["headline_s"]
    assert f"{head_new:.1f} ({head_new / head_ref:.2f}x)" in md
    assert "0/21, 0/21, 0/21" in md
    assert ref.run_dir.exists()


def test_builds_and_protocols_never_share_a_table(tmp_path):
    root = tmp_path / "results"
    speed_run(root, "20261027-010000")
    speed_run(root, "20261027-030000", build="26C1001a", **NANBEIGE)
    md = report.build_report(root)
    assert "macOS build 26B5101f" in md and "macOS build 26C1001a" in md
    assert "No reference run with this protocol version and these builds" in md
    root2 = tmp_path / "results2"
    speed_run(root2, "20261027-010000")
    speed_run(root2, "20261027-030000", protocol=2, **NANBEIGE)
    md2 = report.build_report(root2)
    assert "## Speed on mac (protocol 1," in md2 and "## Speed on mac (protocol 2," in md2


def test_newest_complete_run_is_used(tmp_path):
    root = tmp_path / "results"
    speed_run(root, "20261027-010000", decode=30.0)
    speed_run(root, "20261028-010000", decode=40.0, skip=("m4-3",))  # newer but incomplete
    md = report.build_report(root)
    assert "20261027-010000" in md and "20261028-010000" not in md
    speed_run(root, "20261029-010000", decode=50.0)
    md = report.build_report(root)
    assert "20261029-010000" in md and "| 50.0 |" in md


def test_disturbed_units_and_failed_gates(tmp_path):
    root = tmp_path / "results"
    speed_run(root, "20261027-010000", disturbed=("m3-110k",))
    speed_run(root, "20261027-030000", stop="G3.tool_use failed", **NANBEIGE)
    md = report.build_report(root)
    assert "disturbed x3" in md
    assert "stopped: G3.tool_use failed" in md
    # run.py stops retrying a unit whose cold request hit the cache: the cell says so
    ref_dir = root / "mac" / "qwen-ref" / "oq5" / "backend" / "20261027-010000"
    units = json.loads((ref_dir / "speed/units.json").read_text())
    units["units"]["m1"] = {"verdict": "invalid", "attempts": [
        {"attempt": 1, "verdict": "invalid", "kept": False, "valid": False, "raw_file": "x", "metrics": {}}]}
    (ref_dir / "speed/units.json").write_text(json.dumps(units))
    latency = report.build_report(root).split("### Latency and memory")[1].split("### Decode")[0]
    assert "| invalid |" in latency
    latency = md[md.index("### Latency and memory"):md.index("### Decode")]
    assert "Nanbeige" not in latency


def test_capability_gate_decision_and_pairs(tmp_path):
    root = tmp_path / "results"
    cap_run(root, "qwen-ref", "20261027-010000", {}, reference=True, display="Qwen ref", wall=100.0)
    lost = {t: {"claude_code": False, "hermes": False} for t in GATE_TASKS[:5]}
    cap_run(root, "nanbeige", "20261027-030000", lost, display="Nanbeige", wall=200.0)
    md = report.build_report(root)
    assert "## Capability on mac: gate set (protocol 1, harness epoch e1)" in md
    row = next(ln for ln in md.splitlines() if ln.startswith("| Nanbeige |") and "worse" in ln)
    assert "| 22 (66 pairs) | 56 vs 66 | 0 | 5 | 17 |" in row and "**worse**" in row and "claude_code -5" in row
    assert "provisional" not in row
    assert "net loss >= 5 of 22 tasks (23 %)" in row
    assert "aider/ex00 -2" in md
    assert "| Nanbeige | 56 | 11200 / 5600 (2.00x) |" in md  # 56 pairs both solved, wall 200 vs 100 s
    assert "Claude Code" in md and "17/22" in md and "5 (5xx 5)" in md


def test_capability_excluded_pairs_make_the_decision_provisional(tmp_path):
    root = tmp_path / "results"
    cap_run(root, "qwen-ref", "20261027-010000", {}, reference=True, display="Qwen ref")
    run = cap_run(root, "nanbeige", "20261027-030000", {}, display="Nanbeige")
    doc = json.loads((run / "capability/results.json").read_text())
    for r in doc["records"][:2]:   # two pairs invalid twice: excluded, never scored as unsolved
        r["verdict"], r["valid"] = "invalid", False
    doc.update(finished=True, sets=["gate"])   # a finished leg is complete even with excluded pairs
    (run / "capability/results.json").write_text(json.dumps(doc))
    md = report.build_report(root)
    row = next(ln for ln in md.splitlines() if ln.startswith("| Nanbeige |") and "pairs)" in ln)
    assert "(64 pairs)" in row and "| 0 | 0 |" in row and "provisional: 2 pairs excluded" in row
    assert "excluded pairs (scored on neither side)" in md


def test_capability_ranking_set_and_epochs(tmp_path):
    root = tmp_path / "results"
    cap_run(root, "qwen-ref", "20261027-010000", {t: {h: False for h in HARNESSES} for t in RANK_TASKS[:6]},
            sets=("gate", "ranking"), reference=True, display="Qwen ref")
    cap_run(root, "nanbeige", "20261027-030000", {}, sets=("gate", "ranking"), display="Nanbeige")
    cap_run(root, "flash", "20261027-050000", {}, sets=("gate",), epoch="e2", display="Flash")
    md = report.build_report(root)
    assert "## Capability on mac: ranking set (protocol 1, harness epoch e1)" in md
    rank = md[md.index("## Capability on mac: ranking set"):]
    assert "**better**" in rank and "0.031" in rank
    assert "## Capability on mac: gate set (protocol 1, harness epoch e2)" in md
    e2 = md[md.index("harness epoch e2"):]
    assert "No reference run in this epoch" in e2[:400]


def test_incomplete_capability_sets_are_left_out(tmp_path):
    root = tmp_path / "results"
    cap_run(root, "qwen-ref", "20261027-010000", {}, reference=True, drop=3)
    assert "## Capability" not in report.build_report(root)
    run = cap_run(root, "qwen-ref", "20261028-010000", {}, reference=True, drop=3)
    doc = json.loads((run / "capability/results.json").read_text())
    doc["complete_sets"] = {"gate": True}
    (run / "capability/results.json").write_text(json.dumps(doc))
    assert "## Capability on mac: gate set" in report.build_report(root)


def test_write_results_md(tmp_path):
    out = report.write(tmp_path / "none", tmp_path / "RESULTS.md")
    assert out.read_text().startswith("# Results")


def test_reads_run_py_and_runner_files(tmp_path):
    """The files as run.py and the capability runner write them: units.json with verdict strings,
    `kept` and ~ paths, functional {"passed", "result"}, manifest `machine`, results.json `rows`."""
    root = tmp_path / "results"
    for key, ref, dec in (("qwen-ref", True, 30.0), ("nanbeige", False, 20.0)):
        run_dir = root / "mac" / key / "oq5" / "backend" / "20261027-010000"
        (run_dir / "speed").mkdir(parents=True)
        (run_dir / "capability").mkdir()
        spec = "mtp" if ref else "none"
        model = make_model(**({} if ref else NANBEIGE))
        units: dict = {"run_id": "20261027-010000", "model": key,
                       "functional": {"passed": True, "launch": 1, "time": "t", "result": {
                           "ok": True, "stop_reason": None, "items": {"G0": {"result": "PASS"}}, "m7": None,
                           "files": [], "t_start": 1.0, "t_end": 2.0}}, "units": {}}
        lines = {"g5+m3-64k": [g5_line()] + m3_lines(decodes=(dec,) * 3, ctx=64000), "m1": m1_lines(),
                 "m3-think-16k": m3_lines(decodes=(dec,) * 3, cap=1000, thinking=True)}
        for unit in speed.timed_units(model):
            metrics, valid, note = speed.unit_metrics(unit, lines.get(unit) or (
                m4_lines() if unit.startswith("m4") else m3_lines(decodes=(dec,) * 3)))
            raw = run_dir / "speed" / f"{unit}-2.jsonl"
            raw.write_text("{}\n")
            units["units"][unit] = {"verdict": "clean", "attempts": [
                {"attempt": 1, "launch": 1, "t_start": 1, "t_end": 2, "verdict": "disturbed", "reasons": ["mds"],
                 "valid": True, "note": "", "raw_file": "~/elsewhere/_unverified/x.jsonl", "metrics": metrics,
                 "kept": False},
                {"attempt": 2, "launch": 1, "t_start": 3, "t_end": 4, "verdict": "clean", "reasons": [],
                 "valid": valid, "note": "", "raw_file": str(raw), "metrics": metrics, "kept": True}]}
        (run_dir / "speed" / "units.json").write_text(json.dumps(units))
        (run_dir / "manifest.json").write_text(json.dumps({
            "protocol_version": 1, "run_id": "20261027-010000", "status": "complete", "started": "2026-10-27T01:00",
            "model": {**speed.model_info(model), "spec": spec}, "epoch": "ep-1",
            "machine": {"sw_vers": {"ProductVersion": "27.2", "BuildVersion": "26B5101f"},
                        "xcodebuild": "Xcode 27.1; Build version 27A9275"}}))
        rows = [{"set": "gate", "harness": h, "task": t, "kind": "aider", "scored": True, "valid": True,
                 "verdict": "pass", "solved": True, "solved_within_1200": True, "anomalies": [], "wall_s": 50.0,
                 "model_s": 40.0, "n_llm": 5} for t in GATE_TASKS[:4] for h in HARNESSES]
        (run_dir / "capability" / "results.json").write_text(json.dumps(
            {"schema_version": 1, "protocol_version": 1, "model": key, "epoch": "ep-1", "sets": ["gate"],
             "rows": rows, "finished": True}))
    md = report.build_report(root)
    assert "## Speed on mac (protocol 1, macOS build 26B5101f, Xcode build 27A9275)" in md
    assert "| Nanbeige4.3-33B oQ5 | 20.0 (0.67x) |" in md
    assert "## Capability on mac: gate set (protocol 1, harness epoch ep-1)" in md  # finished: complete despite 4 tasks
    assert "**not worse**" in md and "12/12" in md
