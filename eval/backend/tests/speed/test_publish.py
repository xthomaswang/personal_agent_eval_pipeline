"""publish.py: the results page's summary.json rows (the repository's docs/benchmark/README.md)."""

from __future__ import annotations

import json

import pytest
from test_speed_report import cap_run, speed_run

from modeleval import paths, publish, report

REQUIRED = ("schema", "suite", "protocol", "title", "model", "family", "quant", "run_id", "date", "system", "metrics")


def _values(row: dict) -> list[str]:
    return [m["value"] for m in row["metrics"]]


def test_rows_of_the_committed_runs_are_the_numbers_of_results_md():
    runs = {r.model_key: r for r in report.discover(paths.RESULTS) if publish.ours(r)}
    if not {"qwen3-8-27b-oq5", "nanbeige4-3-33b-oq5"} <= set(runs):
        pytest.skip("the committed runs are not under docs/benchmark")
    by = {}
    for key in ("qwen3-8-27b-oq5", "nanbeige4-3-33b-oq5"):
        out, msgs = publish.rows(runs[key])
        assert msgs == []
        by[key] = {row["suite"]: (path, row) for path, row in out}
    q, n = by["qwen3-8-27b-oq5"], by["nanbeige4-3-33b-oq5"]
    assert _values(q["backend"][1]) == ["111.4 s", "3.20 s", "55.1 tok/s", "51.0 tok/s", "41.9 tok/s", "36.9 tok/s",
                                        "133.4 s", "58.0 GiB"]
    assert _values(n["backend"][1]) == ["210.6 s", "5.05 s", "15.9 tok/s", "13.5 tok/s", "10.5 tok/s", "9.0 tok/s",
                                        "328.5 s", "74.0 GiB"]
    assert _values(q["agent_ability"][1]) == ["21/22", "20/22", "22/22", "63/66", "61/66", "21/22"]
    assert _values(n["agent_ability"][1]) == ["16/22", "15/18 (partial)", "—", "31/40 (partial)", "—", "17/22"]
    assert "40 of 66 gate-set runs" in n["agent_ability"][1]["notes"]
    assert "minor disturbance flag (eval/backend/DESIGN.md §5.4): Cached-prefix TTFT." in n["backend"][1]["notes"]
    assert q["backend"][1]["reference"] is True and "reference" not in n["backend"][1]
    for rows in by.values():
        for suite, (path, row) in rows.items():
            assert all(k in row for k in REQUIRED)
            machine, family, quant, dir_suite, run_id = paths.run_parts(path.parent)
            assert (family, quant, dir_suite, run_id) == (row["family"], row["quant"], suite, row["run_id"])
            assert row["system"]["machine"] == machine and row["system"]["macos_build"] == "26B5101f"
            names = publish.SPEED_METRICS if suite == "backend" else publish.CAP_METRICS
            assert [m["name"] for m in row["metrics"]] == list(names)
            assert (path.parent / row["details"]).resolve().is_file()


def test_publish_writes_rows_and_skips_settings_without_a_protocol_id(tmp_path):
    root = tmp_path / "results"
    speed_run(root, "20261027-010000")                               # protocol 1 on 26B5101f / 27A9275: speed-1
    cap_run(root, "qwen-ref", "20261027-010000", {}, reference=True)  # harness epoch e1: no capability id
    speed_run(root, "20261028-010000", build="26C0000a")             # another macOS build: no speed id
    lines = publish.publish(root)
    assert sum(line.startswith("wrote ") for line in lines) == 1
    assert any("no capability protocol id" in line and "e1" in line for line in lines)
    assert any("no speed protocol id" in line and "26C0000a" in line for line in lines)
    row = json.loads((root / "mac/qwen-ref/oq5/backend/20261027-010000/summary.json").read_text())
    assert row["suite"] == "backend" and row["protocol"] == "speed-1" and row["details"] == "speed/units.json"
    assert row["title"] == publish.SPEED_PROTOCOLS["speed-1"]["title"] and row["reference"] is True
    assert not (root / "mac/qwen-ref/oq5/agent_ability").exists()
    assert publish.publish(root, "nobody") == [f"no run of this pipeline under {paths.tilde(root)} for nobody"]


def test_capability_reference_is_scoped_by_protocol_and_epoch(tmp_path):
    root = tmp_path / "results"
    old = cap_run(root, "qwen3-8-27b-oq5", "old", {}, epoch="7582825160ad", reference=True)
    flash = cap_run(root, "qwen3-8-flash-next-oq4e", "new", {}, epoch="4271c2f4a925")
    paused = cap_run(root, "qwen3-8-27b-oq5", "paused", {}, epoch="4271c2f4a925", reference=True)
    man = flash / "manifest.json"
    doc = json.loads(man.read_text())
    man.write_text(json.dumps({**doc, "protocol_version": 2}))
    runs = {r.run_id: r for r in report.discover(root)}
    old_rows, _ = publish.rows(runs["old"])
    flash_rows, _ = publish.rows(runs["new"])
    paused_rows, messages = publish.rows(runs["paused"])
    assert old_rows[0][1]["protocol"] == "capability-1" and old_rows[0][1]["reference"] is True
    assert flash_rows[0][1]["protocol"] == "capability-2" and flash_rows[0][1]["reference"] is True
    assert not paused_rows and any("no capability protocol id" in line for line in messages)
    assert report.capability_is_reference(runs["old"]) and report.capability_is_reference(runs["new"])
    runs["paused"].manifest["protocol_version"] = 2
    assert not report.capability_is_reference(runs["paused"])


def test_runs_in_progress_unverified_or_foreign_are_not_published(tmp_path):
    root = tmp_path / "results"
    ctx = speed_run(root, "20261027-010000")
    man = ctx.run_dir / "manifest.json"
    doc = json.loads(man.read_text())
    man.write_text(json.dumps({**doc, "status": "running"}))
    foreign = root / "mac/qwen-ref/oq5/backend/2026-10-21-quant-compare"
    foreign.mkdir(parents=True)
    (foreign / "summary.json").write_text("{}")
    assert [r.run_id for r in report.discover(root)] == ["20261027-010000"]   # summary-only dirs are not runs
    assert publish.publish(root)[0].startswith("no run of this pipeline")
    man.write_text(json.dumps({**doc, "unverified": True}))
    assert publish.publish(root)[0].startswith("no run of this pipeline")


def test_notes_name_the_columns_of_the_row_not_unit_ids():
    """A note names what the row shows (SPEED_METRICS) and where DESIGN.md is in the repository."""
    assert publish.design_ref("5.4") == "eval/backend/DESIGN.md §5.4"
    assert publish.unit_columns(["m1"]) == ["Cached-prefix TTFT"]
    assert publish.unit_columns(["m4-2", "m3-32k", "g5+m3-64k"]) == [
        "64K cold TTFT", "Decode 32K", "Decode 64K", "20-step agent trajectory (1 of its 3 runs)"]
    assert publish.unit_columns(["m4-1", "m4-3", "m3-think-16k"]) == [
        "20-step agent trajectory (2 of its 3 runs)", publish.UNIT_LABELS["m3-think-16k"]]
    every = [u for u in publish.UNIT_COLUMNS] + list(publish.UNIT_LABELS)
    assert sorted(every) == sorted(report.speed.UNIT_ORDER)   # every timed unit has a name
    assert {c for cols in publish.UNIT_COLUMNS.values() for c in cols} <= set(publish.SPEED_METRICS)
