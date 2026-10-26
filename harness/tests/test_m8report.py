"""M8 in the harness: the `interference` header record, `run-all --no-summary`, and the slowdown
math of `agentbench m8-report` on synthetic raw files."""

import datetime
import json
from pathlib import Path

import pytest

from agentbench.cli import main
from agentbench.m8report import collect, compare, drift, report, slowdown

from conftest import cli_args, start_mock

T0 = datetime.datetime(2026, 10, 20, 23, 0, tzinfo=datetime.timezone.utc)


def write(raw: Path, suite: str, scenario: str, block: int, value: float | None, run: str = "r1", tag: str = "4bit-x",
          error: str | None = None) -> Path:
    """One block's raw file: header (with the run_m8.sh interference record) plus request lines."""
    created = T0 + datetime.timedelta(minutes=10 * block) + datetime.timedelta(hours=int(run[1:]))
    hdr = {"type": "header", "suite": suite, "tag": f"{tag}-m8-{scenario}", "created": created.isoformat(),
           "args": {"server_pid": 4242}, "interference": {"study": "m8", "scenario": scenario, "run": run, "block": block}}
    lines = [hdr]
    if suite == "m1":
        lines += [{"type": "request", "phase": "warm", "metrics": {"ok": True, "error": None, "ttft_s": value * f, "e2e_s": 5.0}}
                  for f in (0.9, 1.0, 1.1)]
        lines.append({"type": "request", "phase": "warmup", "metrics": {"ok": True, "error": None, "ttft_s": 30.0, "e2e_s": 30.0}})
    elif suite == "m3":
        cell = {"spec": "on", "thinking": False, "ctx": 16000, "cap": 150}
        lines += [{"type": "request", "phase": "trial", "cell": cell,
                   "metrics": {"ok": True, "error": None, "ttft_s": 0.5, "e2e_s": 5.0, "decode_tps": value * f}}
                  for f in (0.98, 1.0, 1.02)]
    else:
        per = value / 3
        for k in range(3):
            ok = not (error and k == 2)
            lines.append({"type": "request", "phase": "start" if k == 0 else "step", "run": 0, "step": k,
                          "metrics": {"ok": ok, "error": None if ok else error, "ttft_s": 1.0, "e2e_s": per}})
        lines.append({"type": "summary", "suite": "m4", "steps": 2, "runs": 1, "run_total_e2e_s": [value]})
    raw.mkdir(parents=True, exist_ok=True)
    f = raw / f"{suite}-{tag}-m8-{scenario}-{created.strftime('%Y%m%d-%H%M%S')}.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return f


@pytest.fixture
def root(tmp_path):
    raw = tmp_path / "llama.cpp" / "raw"
    # B cpu1 B gpu B on one server launch; per suite: (B1, cpu1, B3, gpu, B5)
    values = {"m1": (2.0, 2.2, 2.0, 3.0, 2.2), "m3": (30.0, 24.0, 30.0, 15.0, 30.0), "m4": (200.0, 250.0, 200.0, 300.0, 200.0)}
    for suite, vs in values.items():
        for block, (scen, v) in enumerate(zip(("base", "cpu1", "base", "gpu", "base"), vs), start=1):
            write(raw, suite, scen, block, v)
    return tmp_path


def rows_by(rows, suite, scenario):
    return [r for r in rows if r["suite"] == suite and r["scenario"] == scenario]


def test_slowdown_is_extra_time_for_the_same_work():
    assert slowdown("m1", 2.0, 2.5) == pytest.approx(0.25)    # TTFT: longer is slower
    assert slowdown("m4", 200.0, 250.0) == pytest.approx(0.25)
    assert slowdown("m3", 30.0, 24.0) == pytest.approx(0.25)   # tok/s: 30 -> 24 is 25 % more time per token
    assert slowdown("m3", None, 24.0) is None and slowdown("m1", 2.0, None) is None


def test_pairs_each_scenario_with_its_adjacent_baselines(root):
    blocks = collect(root)
    assert len(blocks) == 15 and {b.fw for b in blocks} == {"llama.cpp"} and {b.tag for b in blocks} == {"4bit-x"}
    rows = compare(blocks)
    (c1,) = rows_by(rows, "m1", "cpu1")
    assert (c1["before"], c1["after"], c1["base"]) == (2.0, 2.0, 2.0) and c1["slowdown"] == pytest.approx(0.1)
    (g1,) = rows_by(rows, "m1", "gpu")
    assert g1["base"] == pytest.approx(2.1) and g1["slowdown"] == pytest.approx(3.0 / 2.1 - 1) and g1["bracketed"]
    (g3,) = rows_by(rows, "m3", "gpu")
    assert g3["slowdown"] == pytest.approx(1.0)   # half the decode rate: twice the time per token
    (c4,) = rows_by(rows, "m4", "cpu1")
    assert c4["slowdown"] == pytest.approx(0.25)
    assert drift(blocks)[("llama.cpp", "4bit-x", "m1")] == pytest.approx(0.05)   # |0| and |+10 %| between baselines


def test_report_table(root):
    md = report(root)
    rows = {r.split("|")[3].strip(): [c.strip() for c in r.split("|")[4:7]] for r in md.splitlines() if r.startswith("| llama.cpp")}
    assert rows["baseline (median)"] == ["2.00 s", "30.0 tok/s", "200.00 s"]
    assert rows["baseline drift (B to B)"] == ["±5.0 %", "±0.0 %", "±0.0 %"]
    assert rows["cpu1"] == ["+10.0 %", "+25.0 %", "+25.0 %"]
    assert rows["gpu"] == ["+42.9 %", "+100.0 %", "+50.0 %"]
    assert list(rows) == ["baseline (median)", "baseline drift (B to B)", "cpu1", "gpu"]
    det = report(root, details=True)
    assert "### Per block" in det and "| llama.cpp | 4bit-x | gpu | 4 | m3 | 30.0 tok/s | 15.0 tok/s | 30.0 tok/s | +100.0 % |" in det


def test_quarantined_baselines_runs_and_incomplete_m4(root):
    raw = root / "llama.cpp" / "raw"
    unv = raw / "_unverified"
    unv.mkdir()
    last = max(raw.glob("m3-4bit-x-m8-base-*.jsonl"))   # the baseline after gpu was disturbed
    last.rename(unv / last.name)
    # a second server launch: its baseline never pairs with the first launch's scenarios
    write(raw, "m1", "base", 1, 9.0, run="r2")
    write(raw, "m1", "cpu1", 2, 9.9, run="r2")
    # an M4 run with a real error is not a total
    write(raw, "m4", "disk", 6, 150.0, error="HTTP 500: boom")
    write(raw, "m4", "base", 7, 200.0)
    rows = compare(collect(root))
    (g3,) = rows_by(rows, "m3", "gpu")
    assert (g3["before"], g3["after"], g3["bracketed"]) == (30.0, None, False) and g3["slowdown"] == pytest.approx(1.0)
    r2 = [r for r in rows_by(rows, "m1", "cpu1") if r["run"] == "r2"]
    assert len(r2) == 1 and r2[0]["base"] == 9.0 and r2[0]["slowdown"] == pytest.approx(0.1)
    (d4,) = rows_by(rows, "m4", "disk")
    assert d4["value"] is None and d4["slowdown"] is None
    md = report(root)
    gpu = next(r for r in md.splitlines() if r.startswith("| llama.cpp | 4bit-x | gpu |"))
    assert "+100.0 %*" in gpu   # one adjacent baseline only
    cpu1 = next(r for r in md.splitlines() if r.startswith("| llama.cpp | 4bit-x | cpu1 |"))
    assert "+10.0 %* (n=2)" in cpu1   # the second launch has no baseline after its cpu1 block
    assert next(r for r in md.splitlines() if "| disk |" in r).endswith("| - |")
    assert "unverified" not in md and report(root, unverified=True).count("| llama.cpp |") >= 2


def test_cli_and_empty_root(root, tmp_path, capsys):
    assert main(["m8-report", "--out-root", str(root), "--framework", "llama.cpp", "--tag", "4bit-x"]) == 0
    assert "| llama.cpp | 4bit-x | cpu1 |" in capsys.readouterr().out
    assert main(["m8-report", "--out-root", str(root), "--tag", "nope"]) == 0
    assert "_no M8 result files found_" in capsys.readouterr().out


def test_interference_header_and_no_summary(built, tmp_path):
    root, _ = built
    srv, _ = start_mock(root, "generic")
    meta = {"study": "m8", "scenario": "cpu1", "params": {"threads": 1}, "allow": ["load"], "load_pid": 123, "block": 2}
    try:
        args = cli_args(srv.url, root, tmp_path, "omlx", "omlx", "--suites", "m1", "--no-summary",
                        "--interference", json.dumps(meta))
        args[args.index("--tag") + 1] = "4bit-m8-cpu1"
        assert main(["run-all"] + args) == 0
        with pytest.raises(SystemExit, match="--interference must be a JSON object"):
            main(["m1"] + cli_args(srv.url, root, tmp_path, "omlx", "omlx", "--interference", "[1]"))
    finally:
        srv.stop()
    f = next((tmp_path / "omlx" / "raw").glob("m1-4bit-m8-cpu1-*.jsonl"))
    hdr = json.loads(f.read_text().splitlines()[0])
    assert hdr["interference"] == meta and hdr["tag"] == "4bit-m8-cpu1"
    assert not (tmp_path / "omlx" / "summary.md").exists()   # M8 blocks stay out of summary.md
