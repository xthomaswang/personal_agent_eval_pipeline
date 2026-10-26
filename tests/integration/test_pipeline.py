"""The modules together: run.py with the real speed.py against a mock oMLX, run.py's spec and runner
environment against the real runner's checks, the capability flags, and the exporter's allowlist
against redact.py's."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from modeleval import paths, report, run, speed
from modeleval.capability import export
from modeleval.server import port_listening

PROGRESS = re.compile(r"^\d\d:\d\d:\d\d [A-Z-]+( |$)")
REAL_REPO = Path(__file__).resolve().parents[2]


def _run(sandbox, **kw) -> int:
    return run.run(run.Options(model="mock-1b", notice_wait=0, settle_s=5, **kw), sandbox.models)


def _only_run(root: Path) -> Path:
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name != "_unverified"]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _assert_released(sandbox) -> None:
    assert not port_listening(sandbox.port)
    assert not sandbox.lock.exists()
    for pf in sandbox.runs.glob("mock-1b/*/pidfile"):
        assert pf.read_text() == "", pf.read_text()


def test_speed_leg_with_the_real_speed_module(sandbox, capsys):
    """Functional block and every timed unit of a spec-none, text-only model, judged and kept."""
    assert _run(sandbox, legs=("speed",)) == 0
    out = capsys.readouterr().out
    assert all(PROGRESS.match(line) for line in out.splitlines()), out   # harness tables went to a log
    res = _only_run(sandbox.results / "mock-1b")
    doc = json.loads((res / "speed/units.json").read_text())
    want = speed.timed_units(sandbox.model)
    assert "m3-think-16k" not in want and sorted(doc["units"]) == sorted(want)
    assert doc["functional"]["passed"] is True and doc["functional"]["launch"] == 1
    items = doc["functional"]["result"]["items"]
    assert [items[g]["result"] for g in ("G0", "G1", "G3", "G4")] == ["PASS"] * 4 and "G2" not in items
    for unit in want:
        rec = doc["units"][unit]
        assert rec["verdict"] == "clean", (unit, rec)
        (att,) = rec["attempts"]
        assert att["kept"] and att["valid"] and att["launch"] == 1 and att["metrics"], (unit, att)
        assert att["raw_file"] == f"speed/{unit}-1.jsonl" and (res / att["raw_file"]).exists()
    assert doc["units"]["g5+m3-64k"]["attempts"][0]["metrics"]["m3"]["warmup"] == "reused"
    assert doc["units"]["m1"]["attempts"][0]["metrics"]["cold_cached_tokens"] == [0]
    assert doc["units"]["m4-1"]["attempts"][0]["metrics"]["m4"]["headline_s"]
    assert "memory" in doc                                      # speed.finalize after run.py's last write
    man = json.loads((res / "manifest.json").read_text())
    assert man["status"] == "complete" and len(man["launches"]) == 1
    assert man["launches"][0]["effective_settings"]["log_expect"]
    for f in (res / "speed/units.json", res / "manifest.json"):
        assert str(Path.home()) not in f.read_text()
    assert not (res / "speed/_agentbench").exists()
    assert (sandbox.runs / "mock-1b" / res.name / "speed-stdout.log").exists()
    _assert_released(sandbox)
    text = report.build_report(sandbox.results)                 # report.py reads what run.py + speed.py wrote
    assert "No complete run yet" not in text and "Mock 1B" in text, text[:2000]
    assert str(Path.home()) not in text


def test_unverified_speed_leg_goes_to_unverified(sandbox):
    assert _run(sandbox, legs=("speed",), unverified=True) == 0
    res = _only_run(paths.UNVERIFIED / "mock-1b")
    doc = json.loads((res / "speed/units.json").read_text())
    assert all(u["verdict"] == "clean" for u in doc["units"].values())
    assert not (sandbox.results / "mock-1b").exists()
    _assert_released(sandbox)


def test_missing_fixtures_are_refused_before_the_lock(sandbox, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(speed, "FIXTURES_DIR", tmp_path / "no-fixtures")
    assert _run(sandbox, legs=("speed",)) == 5
    out = capsys.readouterr().out
    assert "fixtures not built" in out and "agentbench fixtures build" in out and "LOCK-NOTICE" not in out
    assert not (sandbox.root / "lock.log").exists() and not paths.BASE.exists()
    lines = run.plan_lines(run.Options(model="mock-1b"), sandbox.models)
    assert any(x.startswith("ready       fixtures MISSING m3-off-110000") for x in lines)


def test_epoch_problems_are_refused_before_the_lock(sandbox, monkeypatch, capsys):
    from modeleval.capability.epoch import EpochError

    def broken() -> str:
        raise EpochError("pins.toml names epoch test-epoch, but the pinned inputs now hash to abc")

    monkeypatch.setattr(run, "ensure_epoch", broken)
    assert _run(sandbox, legs=("capability",)) == 5
    out = capsys.readouterr().out
    assert "harness epoch: pins.toml names epoch" in out and "LOCK-NOTICE" not in out
    assert not (sandbox.root / "lock.log").exists()


def test_capability_leg_spec_env_and_flags(sandbox, monkeypatch):
    """run.py's spec passes the runner's own preflight, the environment is the allowlist (no
    credentials, no pipeline venv on PATH), and every task gets its disturbance flags."""
    monkeypatch.setenv("PATH", f"{paths.REPO / '.venv/bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-pass")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/nowhere")
    monkeypatch.setenv("BENCH_FAKE_RUNNER_DISTURB", str(sandbox.disturb))
    assert _run(sandbox, legs=("capability",), capability="gate") == 0
    (seen,) = [json.loads(x) for x in (sandbox.root / "runner.log").read_text().splitlines()]
    assert seen["spec_problems"] == [] and seen["forbidden"] == []
    assert str(paths.REPO / ".venv") not in seen["path"] and seen["pydwb"] == "1" and seen["git_locks"] == "0"
    spec = seen["spec"]
    assert spec["sets"] == ["gate"] and spec["epoch"] == "test-epoch" and Path(spec["vendor"]).name == "test-epoch"
    assert spec["model"]["served_id"] == "Mock-1B" and spec["model"]["sampling"]["top_k"] == 0
    res = _only_run(sandbox.results / "mock-1b")
    doc = json.loads((res / "capability/results.json").read_text())
    (row,) = doc["rows"]
    (warm,) = doc["warmups"]
    assert row["flags"]["disturbed"] is True and row["flags"]["disturbed_samples"] >= 1, row["flags"]
    assert warm["flags"]["disturbed"] is False and warm["flags"]["battery_min"] == 95, warm["flags"]
    state = json.loads((sandbox.runs / "mock-1b" / res.name / "capability/state.json").read_text())
    assert all(r["flags"] is not None for r in state["records"])   # the runner's rewrites keep them
    assert str(Path.home()) not in (res / "capability/results.json").read_text()
    _assert_released(sandbox)


def test_capability_drift_yield_stops_resumably(sandbox, monkeypatch, capsys):
    monkeypatch.setenv("BENCH_FAKE_RUNNER_EXIT", "75")
    monkeypatch.setenv("BENCH_FAKE_RUNNER_REASON", "drift")
    assert _run(sandbox, legs=("capability",), capability="gate") == 4
    out = capsys.readouterr().out
    assert "capability runner yielded (drift: fake yield)" in out and "resume=yes" in out
    _assert_released(sandbox)


# ---------------------------------------------------------------- static seams


def _load_redact():
    spec = importlib.util.spec_from_file_location("redact_under_test", REAL_REPO / "scripts/redact.py")
    mod = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


@pytest.mark.parametrize("rel", [
    "results.json", "claude_code/transpose/calls.jsonl", "hermes/wordy/summary.json", "localmelo/react/check.txt",
    "claude_code/transpose__a2/check.txt", "claude-code/transpose/calls.jsonl", "localmelo/react/row.json",
    "claude_code/transpose/prompt.md", "claude_code/calls.jsonl", "claude_code/t/workspace/x.py", "manifest.json",
    "hermes/../calls.jsonl", "hermes/a b/calls.jsonl", "state.json",
])
def test_redact_allowlist_matches_the_exporter(rel):
    """redact.py runs under the system Python in the pre-commit hook and cannot import the package, so
    it mirrors export.is_allowlisted; the two must agree."""
    redact = _load_redact()
    assert (redact.capability_problem(f"results/m/r1/capability/{rel}") is None) is export.is_allowlisted(rel)


def test_runner_imports_in_its_own_isolated_interpreter(tmp_path):
    """`-I` drops the working directory and PYTHONPATH: the runner must still import (editable env)."""
    cap = REAL_REPO / "envs/capability/.venv/bin/python"
    if not cap.exists():
        pytest.skip("envs/capability is not synced")
    code = "import sys, modeleval.capability.runner as r; print(sys.version_info[:2] >= (3, 13), r.EXIT_YIELD)"
    p = subprocess.run([str(cap), "-I", "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert p.returncode == 0 and p.stdout.split() == ["True", "75"], p.stderr


def test_registry_sampling_matches_the_vendored_proxy_presets():
    """The runner refuses a model whose proxy preset injects other sampling than the registry's
    (its selftest); check every registry model now, against the pinned epoch's snapshot."""
    from modeleval import registry

    cap = REAL_REPO / "envs/capability/.venv/bin/python"
    epoch = registry.epoch_id(registry.load_pins(REAL_REPO / "pins.toml"))
    pab = REAL_REPO / "vendor" / str(epoch) / "personal-agent-bench"
    if not cap.exists() or not (pab / "bench").is_dir():
        pytest.skip("envs/capability or the vendor snapshot is missing")
    code = ("import json, sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); "
            "from bench.proxy import sampling as ps; "
            "print(json.dumps({p: ps.SamplingPolicy.parse(p).values for p in sys.argv[2:]}))")
    models = registry.load_models(REAL_REPO / "models.toml")
    presets = sorted({m.proxy_sampling for m in models.values()})
    p = subprocess.run([str(cap), "-I", "-B", "-c", code, str(pab), *presets], capture_output=True, text=True,
                       timeout=60, env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert p.returncode == 0, p.stderr
    values = json.loads(p.stdout)
    for m in models.values():
        assert values[m.proxy_sampling] == m.sampling, (m.key, m.proxy_sampling, values[m.proxy_sampling])
