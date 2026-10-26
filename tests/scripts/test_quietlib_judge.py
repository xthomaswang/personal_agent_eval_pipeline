"""judge_window and window_flags (DESIGN 5.4) on hand-made monitor TSVs.

Timeline used throughout: one sample every 5 s at T0 + 5 i (i = 0..20). The unit runs from
T0 + 50 to T0 + 70, so its window [T0 + 45, T0 + 75] holds samples 9..15; sample 8 is the one
just before, sample 16 the one just after."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

T0 = 1_800_000_000.0
T_START, T_END = T0 + 50, T0 + 70
SYSTEM_PY = Path("/usr/bin/python3")
REPO = Path(__file__).resolve().parents[2]


def row(q, i: int, **kw) -> dict:
    r = {"iso": "", "epoch": T0 + 5 * i, "state": "QUIET", "load1": 1.5, "foreign_cpu": 20.0, "gpu_util": 0,
         "gpu_mem_gb": 0.9, "battery": 100, "thermal_pressure": 0, "pressure": 1, "swapouts_delta": 0,
         "foreign_gpu_ms": 5.0, "foreign_rss_delta_mb": 0.0, "reasons": ""}
    r.update(kw)
    return r


def write(q, path: Path, special: dict[int, dict] | None = None, n: int = 21, skip: tuple = (),
          header: bool = True) -> Path:
    special = special or {}
    with path.open("a", encoding="utf-8") as f:
        if header:
            f.write(q.monitor_header(5.0, [100], frozenset(), 16384))
        for i in range(n):
            if i not in skip:
                f.write(q.format_row(row(q, i, **special.get(i, {}))))
    return path


def judge(q, tmp_path, special=None, **kw) -> dict:
    return q.judge_window(write(q, tmp_path / "m.tsv", special, **kw), T_START, T_END)


def test_quiet_window_is_clean(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {12: {"foreign_cpu": 40.0}})
    assert v == {"clean": True, "reasons": [], "peak_foreign_cpu": 40.0, "samples": 7, "disturbed_samples": 0}


@pytest.mark.parametrize("busy, clean", [
    ({11: "foo(42) 40%", 12: "foo(42) 30%"}, False),          # same pid twice inside
    ({11: "foo(42) 40%", 12: "bar(43) 30%"}, True),           # two different pids
    ({12: "foo(42) 90%"}, True),                              # one sample: a SPIKE
    ({8: "foo(42) 40%", 9: "foo(42) 40%"}, False),            # persistence may use the sample just before
    ({7: "foo(42) 40%", 8: "foo(42) 40%"}, True),             # both before the window
    ({15: "foo(42) 40%", 16: "foo(42) 40%"}, True),           # the sample after never counts
    ({11: "a b (Renderer)(42) 40%", 12: "a b (Renderer)(42) 40%"}, False),   # names with spaces
])
def test_same_foreign_process_in_two_consecutive_samples(quietlib, tmp_path, busy, clean):
    v = judge(quietlib, tmp_path, {i: {"reasons": f"busy: {b}"} for i, b in busy.items()})
    assert v["clean"] is clean
    if not clean:
        assert "busy in 2 consecutive samples" in v["reasons"][1] and "(42)" in v["reasons"][1]


@pytest.mark.parametrize("cpu, clean", [
    ({11: 160.0, 12: 155.0}, False), ({12: 400.0}, True), ({11: 160.0, 13: 160.0}, True),
    ({8: 200.0, 9: 151.0}, False), ({11: 150.0, 12: 150.0}, True),
])
def test_foreign_cpu_total_in_two_consecutive_samples(quietlib, tmp_path, cpu, clean):
    v = judge(quietlib, tmp_path, {i: {"foreign_cpu": c} for i, c in cpu.items()})
    assert v["clean"] is clean
    assert v["peak_foreign_cpu"] == max(c for i, c in cpu.items() if 9 <= i <= 15)


@pytest.mark.parametrize("gpu, floor, clean", [
    ({11: 150.0, 12: 400.0}, None, False), ({11: 150.0}, None, True), ({11: 90.0, 12: 90.0}, None, True),
    ({11: 90.0, 12: 90.0}, 50.0, False), ({11: 150.0, 12: None}, None, True),
])
def test_foreign_gpu_time_in_two_consecutive_samples(quietlib, tmp_path, gpu, floor, clean):
    path = write(quietlib, tmp_path / "m.tsv", {i: {"foreign_gpu_ms": g} for i, g in gpu.items()})
    kw = {} if floor is None else {"gpu_floor_ms": floor}
    v = quietlib.judge_window(path, T_START, T_END, **kw)
    assert v["clean"] is clean
    if not clean:
        assert "foreign GPU" in v["reasons"][1]


def test_gpu_floor_default(quietlib):
    assert quietlib.GPU_FLOOR_MS == 100.0


@pytest.mark.parametrize("reason, clean", [
    ("lock not held by us", False), ("port 8500 listening", False), ("port 8642 listening", False),
    ("power: Battery Power; powermode 2 (High Power)", False), ("battery: 15% < 20%", False),
    ("tools: git clone https://example.invalid/x", False),
    ("thermal: CPU_Speed_Limit = 80", True),                 # logged, not judged
    ("allowed thermal: CPU_Speed_Limit = 80", True), ("allowed power: AC Power; powermode 1 (Low Power)", True),
    ("memory pressure 2 (warn)", True),                      # judged per window with the RSS rule
])
def test_instant_events_count_at_once(quietlib, tmp_path, reason, clean):
    v = judge(quietlib, tmp_path, {12: {"reasons": reason}})
    assert v["clean"] is clean
    if not clean:
        assert v["reasons"][1].endswith(reason.split("; ")[0])   # reasons are "; "-separated segments


def test_battery_column_counts_without_a_reason(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {13: {"battery": 19}})
    assert not v["clean"] and "battery: 19% < 20%" in v["reasons"][1]
    sub = tmp_path / "no-battery"
    sub.mkdir()
    assert judge(quietlib, sub, {13: {"battery": -1}})["clean"]


@pytest.mark.parametrize("special, clean", [
    ({12: {"pressure": 2}}, True),                                                   # server-made pressure
    ({10: {"foreign_rss_delta_mb": 600.0}, 11: {"foreign_rss_delta_mb": 600.0}, 12: {"pressure": 2}}, False),
    ({10: {"foreign_rss_delta_mb": 600.0}, 11: {"foreign_rss_delta_mb": 600.0}, 12: {"pressure": 4}}, False),
    ({12: {"pressure": 2}, 13: {"foreign_rss_delta_mb": 900.0}, 14: {"foreign_rss_delta_mb": 900.0}}, True),
    ({8: {"foreign_rss_delta_mb": 3000.0}, 12: {"pressure": 2}}, True),             # growth before the window
    ({10: {"foreign_rss_delta_mb": 2000.0}, 11: {"foreign_rss_delta_mb": -1500.0}, 12: {"pressure": 2}}, True),
    ({10: {"foreign_rss_delta_mb": 1100.0}, 12: {"swapouts_delta": 8192}}, False),  # 128 MB swap burst
    ({10: {"foreign_rss_delta_mb": 1100.0}, 12: {"swapouts_delta": 100}}, True),    # 1.6 MB: no burst
    ({10: {"foreign_rss_delta_mb": 5000.0}}, True),                                  # growth alone is fine
])
def test_memory_counts_only_with_foreign_growth(quietlib, tmp_path, special, clean):
    v = judge(quietlib, tmp_path, special)
    assert v["clean"] is clean
    if not clean:
        assert "while foreign RSS grew" in v["reasons"][1]


def test_no_samples_is_not_clean(quietlib, tmp_path):
    path = write(quietlib, tmp_path / "m.tsv", n=5)
    v = quietlib.judge_window(path, T_START, T_END)
    assert v == {"clean": False, "reasons": ["no monitor samples in the window"], "peak_foreign_cpu": 0.0,
                 "samples": 0, "disturbed_samples": 0}
    v = quietlib.judge_window(tmp_path / "missing.tsv", T_START, T_END)
    assert not v["clean"] and v["reasons"] == ["monitor log unreadable: FileNotFoundError"]


def test_monitor_gap_is_not_clean(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, skip=(10, 11, 12, 13))
    assert not v["clean"] and v["reasons"][0].startswith("monitor gap of 25 s")
    sub = tmp_path / "one-lost"
    sub.mkdir()
    v = judge(quietlib, sub, skip=(12,))
    assert v["clean"]                                    # one lost sample (10 s) is fine


def test_judged_before_the_monitor_covers_the_end(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, n=12)                  # samples up to T0 + 55, window to T0 + 75
    assert not v["clean"] and v["reasons"][0].startswith("monitor gap of 20 s")


def test_persistence_never_spans_two_monitor_sessions(quietlib, tmp_path):
    path = tmp_path / "m.tsv"
    with path.open("w", encoding="utf-8") as f:
        f.write(quietlib.monitor_header(5.0, [100], frozenset(), 16384))
        for i in range(12):
            f.write(quietlib.format_row(row(quietlib, i, reasons="busy: foo(42) 40%" if i == 11 else "")))
        f.write(quietlib.monitor_header(5.0, [100], frozenset(), 16384))
        for i in range(12, 21):
            f.write(quietlib.format_row(row(quietlib, i, reasons="busy: foo(42) 40%" if i == 12 else "")))
    assert quietlib.judge_window(path, T_START, T_END)["clean"]


def test_partial_last_line_is_ignored(quietlib, tmp_path):
    path = write(quietlib, tmp_path / "m.tsv", n=17)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"x\t{T0 + 70:.1f}\tDISTURBED\t1.5\t20.0\t0\t0.9\t100\t0\t1\t0\t5.0\t0.0\tlock not")
    assert len(quietlib.read_monitor(path)) == 17
    assert quietlib.judge_window(path, T_START, T_END)["clean"]


def test_reads_v3_logs(quietlib, tmp_path):
    path = tmp_path / "v3.tsv"
    lines = ["# quietlib monitor v3 interval=5.0 max_proc_cpu=25.0 max_foreign_cpu=150.0 min_battery_run=20.0 allow=1",
             "# iso\tepoch\tstate\tload1\tforeign_cpu\tgpu_util\tgpu_mem_gb\tbattery\tthermal_pressure\treasons"]
    for i in range(21):
        reasons = "busy: foo(42) 40%" if i in (11, 12) else ""
        lines.append(f"x\t{T0 + 5 * i:.1f}\tQUIET\t1.5\t20.0\t0\t0.9\t100\t0\t{reasons}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = quietlib.read_monitor(path)
    assert len(rows) == 21 and "pressure" not in rows[0] and rows[11]["reasons"] == "busy: foo(42) 40%"
    v = quietlib.judge_window(path, T_START, T_END)
    assert not v["clean"] and "foo(42)" in v["reasons"][1]


def test_reasons_are_capped(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {i: {"reasons": "lock not held by us"} for i in range(21)})
    assert not v["clean"] and v["reasons"][0] == "7 disturbed sample(s) of 7" and v["disturbed_samples"] == 7
    path = tmp_path / "long.tsv"
    with path.open("w", encoding="utf-8") as f:
        f.write(quietlib.monitor_header(5.0, [], frozenset(), 16384))
        for i in range(40):
            f.write(quietlib.format_row(row(quietlib, i, reasons="tools: make")))
    v = quietlib.judge_window(path, T0, T0 + 195)
    assert len(v["reasons"]) == 1 + quietlib.MAX_REASONS + 1 and v["reasons"][-1] == "... and 32 more"


def test_window_flags(quietlib, tmp_path):
    path = write(quietlib, tmp_path / "m.tsv", {
        11: {"reasons": "busy: foo(42) 40%", "battery": 61}, 12: {"reasons": "busy: foo(42) 40%", "battery": 58},
        13: {"foreign_cpu": 99.0}, 17: {"battery": 10, "reasons": "battery: 10% < 20%"}})
    assert quietlib.window_flags(path, T_START, T_END) == {
        "disturbed_samples": 1, "peak_foreign_cpu": 99.0, "battery_min": 58, "samples": 7}
    assert quietlib.window_flags(path, T0 + 1000, T0 + 2000) == {
        "disturbed_samples": 0, "peak_foreign_cpu": 0.0, "battery_min": None, "samples": 0}


def test_row_state(quietlib):
    q = quietlib
    assert q.row_state(row(q, 1), row(q, 0)) == "QUIET"
    assert q.row_state(row(q, 1, foreign_gpu_ms=500.0), row(q, 0)) == "SPIKE"
    assert q.row_state(row(q, 1, foreign_gpu_ms=500.0), row(q, 0, foreign_gpu_ms=200.0)) == "DISTURBED"
    assert q.row_state(row(q, 1, reasons="thermal: x"), None) == "QUIET"
    assert q.row_state(row(q, 1, reasons="tools: make"), None) == "DISTURBED"


@pytest.mark.skipif(not SYSTEM_PY.exists(), reason="no /usr/bin/python3")
def test_judge_runs_on_the_system_python(quietlib, tmp_path):
    """quiet_check.sh runs quietlib on /usr/bin/python3 (3.9 here): the module must stay 3.9-clean."""
    path = write(quietlib, tmp_path / "m.tsv", {11: {"foreign_cpu": 160.0}, 12: {"foreign_cpu": 170.0}})
    code = ("import importlib.util, json, sys\n"
            "spec = importlib.util.spec_from_file_location('q', sys.argv[1])\n"
            "q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)\n"
            "print(json.dumps([q.judge_window(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])),\n"
            "                  q.window_flags(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))]))\n")
    out = subprocess.run([str(SYSTEM_PY), "-I", "-B", "-c", code, str(REPO / "scripts/quietlib.py"), str(path),
                          str(T_START), str(T_END)], capture_output=True, text=True, check=True).stdout
    verdict, flags = json.loads(out)
    assert verdict == quietlib.judge_window(path, T_START, T_END) and not verdict["clean"]
    assert flags["disturbed_samples"] == 1
    help_out = subprocess.run([str(SYSTEM_PY), "-I", "-B", str(REPO / "scripts/quietlib.py"), "monitor", "--help"],
                              capture_output=True, text=True, check=True).stdout
    assert "--count" in help_out
