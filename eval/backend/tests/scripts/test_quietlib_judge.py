"""judge_window and window_flags (DESIGN 5.4) on hand-made monitor TSVs.

Timeline used throughout: one sample every 5 s at T0 + 5 i (i = 0..20). The unit runs from
T0 + 50 to T0 + 70, so its window [T0 + 45, T0 + 75] holds samples 9..15; sample 8 is the one
just before, sample 16 the one just after.

Two levels (the user's instruction of 2026-10-26): a major flag repeats the unit, a minor one keeps
it with the flag. `level` in the cases below is the verdict's severity: clean, minor or major.

CPU and GPU at the major line follow the sustain/share rule (2026-10-26): such a level in two
consecutive samples is major only when its run in the window lasts 6 samples or the window's
samples at the line make up 25 % of it. In the 7-sample window above two samples are already 29 %,
so the sustain and burst cases use a long timeline: the unit runs from T0 + 50 to T0 + 435, its
window [T0 + 45, T0 + 440] holds samples 9..88 (80 samples, so 20 make 25 %)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

T0 = 1_800_000_000.0
T_START, T_END = T0 + 50, T0 + 70
LONG_END, LONG_N = T0 + 435, 95          # the long timeline: 80 window samples (9..88)
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


def judge_long(q, tmp_path, special=None) -> dict:
    return q.judge_window(write(q, tmp_path / "long.tsv", special, n=LONG_N), T_START, LONG_END)


def assert_level(v: dict, level: str, text: str | None = None) -> None:
    """The verdict's severity, its compatibility keys, and `text` in the reasons of that level."""
    assert v["severity"] == level and v["clean"] is (level == "clean"), v
    assert bool(v["major_reasons"]) is (level == "major") and bool(v["reasons"]) is (level != "clean"), v
    if level == "minor":
        assert v["minor_reasons"] and v["major_samples"] == 0, v
    if text is not None:
        own = v["major_reasons"] if level == "major" else v["minor_reasons"]
        assert any(text in r for r in own) and any(text in r for r in v["reasons"]), v


def test_quiet_window_is_clean(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {12: {"foreign_cpu": 40.0}})
    assert v == {"clean": True, "severity": "clean", "reasons": [], "major_reasons": [], "minor_reasons": [],
                 "peak_foreign_cpu": 40.0, "samples": 7, "disturbed_samples": 0, "major_samples": 0,
                 "minor_samples": 0}


@pytest.mark.parametrize("busy, level", [
    ({11: "foo(42) 40%", 12: "foo(42) 30%"}, "minor"),        # same pid twice inside
    ({11: "foo(42) 40%", 12: "bar(43) 30%"}, "clean"),        # two different pids
    ({12: "foo(42) 90%"}, "clean"),                           # one sample: a SPIKE
    ({8: "foo(42) 40%", 9: "foo(42) 40%"}, "minor"),          # persistence may use the sample just before
    ({7: "foo(42) 40%", 8: "foo(42) 40%"}, "clean"),          # both before the window
    ({15: "foo(42) 40%", 16: "foo(42) 40%"}, "clean"),        # the sample after never counts
    ({11: "a b (Renderer)(42) 40%", 12: "a b (Renderer)(42) 40%"}, "minor"),   # names with spaces
    ({11: "foo(42) 400%", 12: "foo(42) 400%"}, "minor"),      # one process is minor; the total judges load
])
def test_same_foreign_process_in_two_consecutive_samples(quietlib, tmp_path, busy, level):
    v = judge(quietlib, tmp_path, {i: {"reasons": f"busy: {b}"} for i, b in busy.items()})
    assert_level(v, level)
    if level != "clean":
        assert "busy in 2 consecutive samples" in v["reasons"][1] and "(42)" in v["reasons"][1]
        assert v["minor_reasons"][0] == "1 sample(s) of 7 with a minor flag"


@pytest.mark.parametrize("cpu, level", [
    ({11: 160.0, 12: 155.0}, "minor"), ({12: 400.0}, "clean"), ({11: 160.0, 13: 160.0}, "clean"),
    ({8: 200.0, 9: 151.0}, "minor"), ({11: 150.0, 12: 150.0}, "clean"),
    # In 7 window samples a pair at the major line is 29 %: the share rule makes it major.
    ({11: 300.0, 12: 300.0}, "major"),                        # the major line itself counts
    ({11: 299.9, 12: 800.0}, "minor"),                        # both samples must reach it
    ({8: 450.0, 9: 310.0}, "minor"),                          # the sample just before completes the pair; 1 of 7 is < 25 %
    ({11: 400.0, 13: 400.0}, "clean"),                        # never two in a row: lone samples never count
    ({11: 200.0, 12: 400.0, 13: 400.0}, "major"),
])
def test_foreign_cpu_total_in_two_consecutive_samples(quietlib, tmp_path, cpu, level):
    v = judge(quietlib, tmp_path, {i: {"foreign_cpu": c} for i, c in cpu.items()})
    assert_level(v, level, "foreign CPU" if level != "clean" else None)
    assert v["peak_foreign_cpu"] == max(c for i, c in cpu.items() if 9 <= i <= 15)
    if level == "major":
        assert any("window samples at or above it" in r and r.endswith(", >= 25%") for r in v["major_reasons"])


def _at(idx, value: float = 400.0, col: str = "foreign_cpu") -> dict[int, dict]:
    return {i: {col: value} for i in idx}


@pytest.mark.parametrize("special, level, why", [
    (_at((30, 31)), "minor", "short burst (2 consecutive sample(s), < 6; 2 of 80"),
    (_at((30, 31, 32)), "minor", "short burst (3 consecutive"),                  # the 2026-10-26 bursts
    (_at(range(30, 35)), "minor", "short burst (5 consecutive"),                # 25 s
    (_at(range(30, 36)), "major", "for 6 consecutive samples (sustained: >= 6)"),   # 30 s, 7.5 % of the window
    ({**_at((30, 31, 32, 34, 35, 36)), 33: {"foreign_cpu": 299.9}}, "minor", "3 consecutive"),  # two runs of 3
    (_at(tuple(x for s in range(20, 80, 6) for x in (s, s + 1))), "major",
     "20 of 80 window samples at or above it (25.0%), >= 25%"),
    (_at((20, 21, 30, 31, 40, 41, 60)), "minor", "7 of 80 window samples"),     # 8.75 %
    (_at((20, 21, *range(24, 60, 2))), "major", "20 of 80"),                  # lone samples count in the share
    (_at((20, 30, 40, 50, 60, 70, 80, 85)), "clean", None),                     # ... but are never flagged
    (_at((4, 5, 6, 7, 8, 9)), "minor", "1 consecutive sample(s)"),              # a run counts inside the window
    (_at(range(84, 95)), "minor", "5 consecutive sample(s)"),                   # samples after it never count
    (_at(range(30, 36), 1000.0, "foreign_gpu_ms"), "major", "foreign GPU 1000 ms >= 500 ms for 6 consecutive"),
    (_at((30, 31, 32), 1000.0, "foreign_gpu_ms"), "minor", "foreign GPU 1000 ms >= 500 ms in a short burst"),
    (_at(tuple(x for s in range(20, 80, 6) for x in (s, s + 1)), 600.0, "foreign_gpu_ms"), "major", "20 of 80"),
])
def test_major_cpu_and_gpu_need_a_sustained_or_frequent_level(quietlib, tmp_path, special, level, why):
    """A short burst at the major line is minor (the user's instruction of 2026-10-26)."""
    v = judge_long(quietlib, tmp_path, special)
    assert v["samples"] == 80
    assert_level(v, level, why)


def test_a_sustained_run_flags_its_persistent_samples(quietlib, tmp_path):
    v = judge_long(quietlib, tmp_path, _at(range(30, 37)))
    assert v["severity"] == "major" and v["major_samples"] == 6 and v["minor_samples"] == 0  # the first has no pair
    assert all("for 7 consecutive samples" in r for r in v["major_reasons"][1:])
    sub = tmp_path / "flags"
    sub.mkdir()
    path = write(quietlib, sub / "m.tsv", _at((30, 31, 32)), n=LONG_N)
    assert quietlib.window_flags(path, T_START, LONG_END) == {
        "disturbed_samples": 2, "major_samples": 0, "minor_samples": 2, "peak_foreign_cpu": 400.0,
        "battery_min": 100, "samples": 80}


def test_a_run_never_spans_two_monitor_sessions(quietlib, tmp_path):
    path = tmp_path / "m.tsv"
    with path.open("w", encoding="utf-8") as f:
        for lo, hi in ((0, 33), (33, LONG_N)):                  # a monitor restart between samples 32 and 33
            f.write(quietlib.monitor_header(5.0, [100], frozenset(), 16384))
            for i in range(lo, hi):
                f.write(quietlib.format_row(row(quietlib, i, foreign_cpu=400.0 if 30 <= i < 36 else 20.0)))
    v = quietlib.judge_window(path, T_START, LONG_END)
    assert v["severity"] == "minor" and v["minor_samples"] == 4, v      # 31, 32 and 34, 35
    assert any("short burst (3 consecutive" in r for r in v["minor_reasons"]), v


def test_cpu_samples_split_into_minor_and_major(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {11: {"foreign_cpu": 200.0}, 12: {"foreign_cpu": 400.0}, 13: {"foreign_cpu": 400.0}})
    assert v["major_samples"] == 1 and v["minor_samples"] == 1 and v["disturbed_samples"] == 2
    assert v["reasons"][0] == "2 flagged sample(s) of 7 (1 major, 1 minor)"
    assert v["major_reasons"][0] == "1 sample(s) of 7 with a major flag" and len(v["major_reasons"]) == 2
    assert "> 150% in 2 consecutive samples" in v["minor_reasons"][1]


@pytest.mark.parametrize("gpu, floor, level", [
    ({11: 300.0, 12: 400.0}, None, "minor"), ({11: 300.0}, None, "clean"), ({11: 142.0, 12: 114.0}, None, "clean"),
    ({11: 90.0, 12: 90.0}, 50.0, "minor"), ({11: 300.0, 12: None}, None, "clean"),
    ({11: 500.0, 12: 500.0}, None, "major"),                  # 10 % of GPU time: the major line counts
    ({11: 499.9, 12: 2000.0}, None, "minor"), ({11: 3000.0}, None, "clean"),
    ({11: 1000.0, 12: 1000.0, 13: None}, None, "major"),
])
def test_foreign_gpu_time_in_two_consecutive_samples(quietlib, tmp_path, gpu, floor, level):
    path = write(quietlib, tmp_path / "m.tsv", {i: {"foreign_gpu_ms": g} for i, g in gpu.items()})
    kw = {} if floor is None else {"gpu_floor_ms": floor}
    v = quietlib.judge_window(path, T_START, T_END, **kw)
    assert_level(v, level, "foreign GPU" if level != "clean" else None)
    if level != "clean":
        assert "foreign GPU" in v["reasons"][1]


def test_thresholds(quietlib):
    assert (quietlib.GPU_FLOOR_MS, quietlib.MAJOR_GPU_MS) == (250.0, 500.0)
    assert (quietlib.MAX_PROC_CPU, quietlib.MAX_FOREIGN_CPU, quietlib.MAJOR_FOREIGN_CPU) == (25.0, 150.0, 300.0)
    assert (quietlib.MAJOR_SUSTAIN_SAMPLES, quietlib.MAJOR_SHARE) == (6, 25.0)


def _reload(quietlib, monkeypatch, **env):
    for k, val in env.items():
        monkeypatch.setenv(k, val)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("_quietlib_env", quietlib.__file__)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    return q


def test_major_lines_follow_the_environment(quietlib, monkeypatch, tmp_path):
    q = _reload(quietlib, monkeypatch, QUIET_MAJOR_FOREIGN_CPU="200", QUIET_MAJOR_GPU_MS="300")
    assert (q.MAJOR_FOREIGN_CPU, q.MAJOR_GPU_MS) == (200.0, 300.0)
    assert judge(q, tmp_path, {11: {"foreign_cpu": 210.0}, 12: {"foreign_cpu": 205.0}})["severity"] == "major"
    sub = tmp_path / "gpu"
    sub.mkdir()
    assert judge(q, sub, {11: {"foreign_gpu_ms": 310.0}, 12: {"foreign_gpu_ms": 305.0}})["severity"] == "major"
    assert "major_foreign_cpu=200.0" in q.monitor_header(5.0, [], frozenset(), 16384)


@pytest.mark.parametrize("env, burst, short_window", [
    ({"QUIET_MAJOR_SUSTAIN_SAMPLES": "2"}, "major", "major"),     # the rule before 2026-10-26: any pair is major
    ({"QUIET_MAJOR_SUSTAIN_SAMPLES": "4", "QUIET_MAJOR_SHARE": "50"}, "minor", "minor"),
    ({"QUIET_MAJOR_SUSTAIN_SAMPLES": "3", "QUIET_MAJOR_SHARE": "50"}, "major", "minor"),
])
def test_sustain_and_share_follow_the_environment(quietlib, monkeypatch, tmp_path, env, burst, short_window):
    q = _reload(quietlib, monkeypatch, **env)
    assert q.MAJOR_SUSTAIN_SAMPLES == int(env["QUIET_MAJOR_SUSTAIN_SAMPLES"])
    assert judge_long(q, tmp_path, _at((30, 31, 32)))["severity"] == burst
    assert judge(q, tmp_path, {11: {"foreign_cpu": 400.0}, 12: {"foreign_cpu": 400.0}})["severity"] == short_window
    header = q.monitor_header(5.0, [], frozenset(), 16384)
    assert f"major_sustain_samples={env['QUIET_MAJOR_SUSTAIN_SAMPLES']} " in header
    assert f"major_share={float(env.get('QUIET_MAJOR_SHARE', '25'))} " in header


@pytest.mark.parametrize("reason, level", [
    ("lock not held by us", "major"), ("port 8500 listening", "major"), ("port 8642 listening", "major"),
    ("power: Battery Power; powermode 2 (High Power)", "major"), ("battery: 15% < 20%", "major"),
    ("tools: git clone (pid 4242)", "minor"),               # its CPU cost is what the CPU rules judge
    ("thermal: CPU_Speed_Limit = 80", "clean"),             # logged, not judged
    ("allowed thermal: CPU_Speed_Limit = 80", "clean"), ("allowed power: AC Power; powermode 1 (Low Power)", "clean"),
    ("memory pressure 2 (warn)", "clean"),                  # judged per window with the RSS rule
])
def test_instant_events_count_at_once(quietlib, tmp_path, reason, level):
    v = judge(quietlib, tmp_path, {12: {"reasons": reason}})
    assert_level(v, level, reason.split("; ")[0] if level != "clean" else None)
    if level != "clean":
        assert v["reasons"][1].endswith(reason.split("; ")[0])   # reasons are "; "-separated segments


def test_a_sample_with_both_levels_lists_each_under_its_own(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {12: {"reasons": "lock not held by us; tools: make (pid 7)"}})
    assert v["severity"] == "major" and v["major_samples"] == v["minor_samples"] == v["disturbed_samples"] == 1
    assert v["major_reasons"][1].endswith(" lock not held by us") and "tools" not in v["major_reasons"][1]
    assert v["minor_reasons"][1].endswith(" tools: make (pid 7)")
    assert v["reasons"][1].endswith(" lock not held by us; tools: make (pid 7)")


def test_battery_column_counts_without_a_reason(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {13: {"battery": 19}})
    assert_level(v, "major", "battery: 19% < 20%")
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
    assert_level(v, "clean" if clean else "major", None if clean else "while foreign RSS grew")
    if not clean:
        assert "while foreign RSS grew" in v["reasons"][1]


def test_no_samples_is_not_clean(quietlib, tmp_path):
    path = write(quietlib, tmp_path / "m.tsv", n=5)
    v = quietlib.judge_window(path, T_START, T_END)
    assert v == {"clean": False, "severity": "major", "reasons": ["no monitor samples in the window"],
                 "major_reasons": ["no monitor samples in the window"], "minor_reasons": [], "peak_foreign_cpu": 0.0,
                 "samples": 0, "disturbed_samples": 0, "major_samples": 0, "minor_samples": 0}
    v = quietlib.judge_window(tmp_path / "missing.tsv", T_START, T_END)
    assert v["severity"] == "major" and v["reasons"] == v["major_reasons"] == ["monitor log unreadable: FileNotFoundError"]


def test_monitor_gap_is_not_clean(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, skip=(10, 11, 12, 13))
    assert not v["clean"] and v["reasons"][0].startswith("monitor gap of 25 s")
    assert v["severity"] == "major" and v["major_reasons"][0].startswith("monitor gap of 25 s")
    sub = tmp_path / "one-lost"
    sub.mkdir()
    v = judge(quietlib, sub, skip=(12,))
    assert v["clean"]                                    # one lost sample (10 s) is fine


def test_judged_before_the_monitor_covers_the_end(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, n=12)                  # samples up to T0 + 55, window to T0 + 75
    assert not v["clean"] and v["reasons"][0].startswith("monitor gap of 20 s") and v["severity"] == "major"


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
    assert not v["clean"] and "foo(42)" in v["reasons"][1] and v["severity"] == "minor"


def test_reasons_are_capped(quietlib, tmp_path):
    v = judge(quietlib, tmp_path, {i: {"reasons": "lock not held by us"} for i in range(21)})
    assert not v["clean"] and v["reasons"][0] == "7 flagged sample(s) of 7 (7 major, 0 minor)"
    assert v["disturbed_samples"] == v["major_samples"] == 7 and v["minor_samples"] == 0
    path = tmp_path / "long.tsv"
    with path.open("w", encoding="utf-8") as f:
        f.write(quietlib.monitor_header(5.0, [], frozenset(), 16384))
        for i in range(40):
            f.write(quietlib.format_row(row(quietlib, i, reasons="tools: make")))
    v = quietlib.judge_window(path, T0, T0 + 195)
    assert len(v["reasons"]) == 1 + quietlib.MAX_REASONS + 1 and v["reasons"][-1] == "... and 32 more"
    assert v["severity"] == "minor" and v["minor_reasons"][-1] == "... and 32 more" and v["major_reasons"] == []
    assert len(v["minor_reasons"]) == 1 + quietlib.MAX_REASONS + 1


def test_window_flags(quietlib, tmp_path):
    path = write(quietlib, tmp_path / "m.tsv", {
        11: {"reasons": "busy: foo(42) 40%", "battery": 61}, 12: {"reasons": "busy: foo(42) 40%", "battery": 58},
        13: {"foreign_cpu": 99.0, "reasons": "lock not held by us"}, 17: {"battery": 10, "reasons": "battery: 10% < 20%"}})
    assert quietlib.window_flags(path, T_START, T_END) == {
        "disturbed_samples": 2, "major_samples": 1, "minor_samples": 1, "peak_foreign_cpu": 99.0, "battery_min": 58,
        "samples": 7}
    assert quietlib.window_flags(path, T0 + 1000, T0 + 2000) == {
        "disturbed_samples": 0, "major_samples": 0, "minor_samples": 0, "peak_foreign_cpu": 0.0, "battery_min": None,
        "samples": 0}


def test_row_state(quietlib):
    q = quietlib
    assert q.row_state(row(q, 1), row(q, 0)) == "QUIET"
    assert q.row_state(row(q, 1, foreign_gpu_ms=500.0), row(q, 0)) == "SPIKE"
    assert q.row_state(row(q, 1, foreign_gpu_ms=500.0), row(q, 0, foreign_gpu_ms=300.0)) == "MINOR"
    # at the major line in two samples: minor per sample; judge_window applies the sustain/share rule
    assert q.row_state(row(q, 1, foreign_gpu_ms=500.0), row(q, 0, foreign_gpu_ms=600.0)) == "MINOR"
    assert q.row_state(row(q, 1, foreign_cpu=200.0), row(q, 0, foreign_cpu=180.0)) == "MINOR"
    assert q.row_state(row(q, 1, foreign_cpu=320.0), row(q, 0, foreign_cpu=300.0)) == "MINOR"
    assert [f.burst for f in q.row_flags(row(q, 1, foreign_cpu=320.0, foreign_gpu_ms=600.0),
                                         row(q, 0, foreign_cpu=300.0, foreign_gpu_ms=500.0))] == [
        "foreign_cpu", "foreign_gpu_ms"]
    assert q.row_flags(row(q, 1, foreign_cpu=200.0), row(q, 0, foreign_cpu=180.0))[0].burst == ""
    assert q.row_state(row(q, 1, reasons="thermal: x"), None) == "QUIET"
    assert q.row_state(row(q, 1, reasons="tools: make"), None) == "MINOR"
    assert q.row_state(row(q, 1, reasons="tools: make; port 8500 listening"), None) == "DISTURBED"
    assert [f.severity for f in q.row_flags(row(q, 1, reasons="tools: make; lock not held by us"), None)] == [
        "major", "minor"]


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
    assert verdict["severity"] == "minor" and flags["disturbed_samples"] == flags["minor_samples"] == 1
    help_out = subprocess.run([str(SYSTEM_PY), "-I", "-B", str(REPO / "scripts/quietlib.py"), "monitor", "--help"],
                              capture_output=True, text=True, check=True).stdout
    assert "--count" in help_out
    help_out = subprocess.run([str(SYSTEM_PY), "-I", "-B", str(REPO / "scripts/quietlib.py"), "check", "--help"],
                              capture_output=True, text=True, check=True).stdout
    assert "--major-only" in help_out
