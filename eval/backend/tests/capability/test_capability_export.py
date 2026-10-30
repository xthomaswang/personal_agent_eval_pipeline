"""The allowlisted export into results/ (DESIGN §8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modeleval import paths
from modeleval.capability import checks, export

HOME = str(paths.HOME)


def _run_dir(root: Path) -> Path:
    """A harness run directory as bench.harness and smoke leave it, with home paths in nested places."""
    run = root / "runs" / "claude_code" / "aider__transpose"
    for name in ("claude-home", "hermes-home", "mem", "workspace", "check"):
        (run / name).mkdir(parents=True)
        (run / name / "x.txt").write_text(f"{HOME}/secret\n")
    calls = [{"call_index": 0, "model": "M", "upstream": "http://127.0.0.1:8703", "bodies": None,
              "warnings": [f"read {HOME}/Developer/x"]}, {"call_index": 1, "model": "M", "bodies": None}]  # fmt: skip
    (run / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls) + '{"truncat')
    summary = {"workspace": f"{HOME}/Developer/mlsys/experiments/agent-model-eval-ws/aider__transpose",
               "harness_report": {"run_record": {"config": {"memory_dir": f"{HOME}/runs/x/mem"}}}}  # fmt: skip
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "check.log").write_text(f"rootdir: {HOME}/runs/x/check\n8 passed\n"
                                   f"in: {HOME}/Developer/mlsys/experiments/agent-model-eval-ws/mcpmark__x\n")
    for name in ("prompt.md", "manifest.json", "stdout.log", "stderr.log", "proxy.log", "fake_seen.json"):
        (run / name).write_text(f"{HOME}\n")
    return run


def test_only_the_allowlist_is_exported_with_home_paths_rewritten(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    dest = tmp_path / "results" / "claude_code" / "aider__transpose"
    written = export.export_run(run, dest)
    assert sorted(p.name for p in written) == ["calls.jsonl", "check.txt", "summary.json"]
    assert sorted(p.name for p in dest.iterdir()) == ["calls.jsonl", "check.txt", "summary.json"]
    for path in dest.iterdir():
        assert HOME not in path.read_text()
    summary = json.loads((dest / "summary.json").read_text())
    assert summary["workspace"] == "ws/aider__transpose"   # relative to its declared root (paths.portable)
    assert summary["harness_report"]["run_record"]["config"] == {}  # memory_dir is not an allowlisted key
    calls = [json.loads(line) for line in (dest / "calls.jsonl").read_text().splitlines()]
    assert [c["call_index"] for c in calls] == [0, 1]  # the truncated record is dropped
    assert calls[0]["warnings"] == ["read ~/Developer/x"]
    assert (dest / "check.txt").read_text().startswith("rootdir: ~/runs/x/check")
    assert "in: ws/mcpmark__x\n" in (dest / "check.txt").read_text()


def test_summary_config_keeps_only_the_allowlisted_localmelo_fields(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config = {"memory_dir": f"{HOME}/runs/x/mem", "tools": ["installed-server"], "window_eviction": "lru",
              "unknown_key": 1}
    summary = {"exit_code": 0, "harness_report": {"steps": 3, "run_record": {"status": "stuck", "config": config}}}
    (run / "summary.json").write_text(json.dumps(summary))
    export.export_run(run, tmp_path / "out")
    text = (tmp_path / "out" / "summary.json").read_text()
    assert HOME not in text
    exported = json.loads(text)
    assert exported["harness_report"]["run_record"]["config"] == {"window_eviction": "lru"}
    assert exported["harness_report"]["run_record"]["status"] == "stuck"
    assert exported["exit_code"] == 0 and exported["harness_report"]["steps"] == 3


def test_bodies_are_refused(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    (run / "calls.jsonl").write_text(json.dumps({"call_index": 0, "bodies": {"request": "bodies/0.json"}}) + "\n")
    with pytest.raises(export.ExportError, match="bodies"):
        export.export_run(run, tmp_path / "out")
    (run / "calls.jsonl").write_text(json.dumps({"call_index": 0, "bodies": None}) + "\n")
    (run / "bodies").mkdir()
    with pytest.raises(export.ExportError, match="bodies"):
        export.export_run(run, tmp_path / "out2")


def test_rows_keep_only_allowlisted_fields() -> None:
    record = {"set": "gate", "harness": "hermes", "task": "aider/wordy", "verdict": "pass", "solved": True,
              "run_dir": "hermes/aider__wordy", "harness_report": {"x": HOME}, "watch": None,
              "check": {"passed": True, "detail": f"{HOME}/x 1 passed", "tail": "t", "extra_files": []},
              "error": f"OSError: {HOME}/a"}  # fmt: skip
    row = export.clean_row(record)
    assert set(row) == set(export.ROW_FIELDS)
    assert "harness_report" not in row and "watch" not in row
    assert row["check"] == {"passed": True, "detail": "~/x 1 passed", "extra_files": []}
    assert row["error"] == "OSError: ~/a"


def test_rows_publish_budget_exhaustion_next_to_localmelos_status_and_the_error_count() -> None:
    """A refused main call is counted by n_llm_err like any failed call, and localmelo calls its run provider_error.
    The row carries the plain flag and the refusal count (calls.budget_refusals), and keeps both of those."""
    refused = {"calls": 31, "generation_calls": 30, "budget_refusals": 1, "examples": []}
    melo = {"set": "ranking", "harness": "localmelo", "task": "aider/pov", "run_dir": "localmelo/aider__pov",
            "verdict": "fail", "localmelo_status": "provider_error", "n_llm": 30, "n_llm_err": 1,
            "budget_exhausted": True, "calls": refused}  # fmt: skip
    row = export.clean_row(melo)
    assert row["budget_exhausted"] is True and row["calls"]["budget_refusals"] == 1
    assert (row["verdict"], row["localmelo_status"], row["n_llm"], row["n_llm_err"]) == ("fail", "provider_error", 30, 1)
    # Another harness: the flag is not localmelo's, and a run without a refusal says false, not nothing
    other = {"set": "gate", "harness": "hermes", "task": "aider/wordy", "run_dir": "hermes/aider__wordy",
             "verdict": "pass", "localmelo_status": None, "n_llm": 6, "n_llm_err": 0,
             "budget_exhausted": False, "calls": {**refused, "calls": 6, "generation_calls": 6, "budget_refusals": 0}}  # fmt: skip
    row = export.clean_row(other)
    assert row["budget_exhausted"] is False and row["calls"]["budget_refusals"] == 0 and row["localmelo_status"] is None
    refused_hermes = {**other, "verdict": "fail", "n_llm": 30, "n_llm_err": 6, "budget_exhausted": True,
                      "calls": {**refused, "budget_refusals": 6}}  # fmt: skip
    row = export.clean_row(refused_hermes)
    assert (row["budget_exhausted"], row["calls"]["budget_refusals"], row["n_llm_err"]) == (True, 6, 6)


def test_rows_from_before_the_budget_flag_derive_it_from_the_refusal_count() -> None:
    """A record written before outcome() had budget_exhausted gets it from calls.budget_refusals on a re-export."""
    old = {"harness": "claude_code", "verdict": "fail", "calls": {"calls": 40, "budget_refusals": 10}}
    assert export.clean_row(old)["budget_exhausted"] is True
    assert export.clean_row({**old, "calls": {"calls": 12, "budget_refusals": 0}})["budget_exhausted"] is False
    assert export.clean_row({"harness": "hermes", "verdict": "invalid"})["budget_exhausted"] is None  # no calls summary


def test_every_field_of_a_verdict_row_is_on_the_row_allowlist() -> None:
    """checks.outcome() fills the runner's record; a field it adds that ROW_FIELDS lacks would never be published."""
    calls = checks.check_calls([], served_id="M", sampling={}, thinking="off")
    out = checks.outcome({"harness": "claude_code", "verdict": "pass", "check": {"passed": True}}, calls, max_steps=30)
    assert set(out) <= set(export.ROW_FIELDS)
    assert "budget_exhausted" in out


def test_write_results(tmp_path: Path) -> None:
    path = export.write_results(tmp_path / "results.json", {"rows": [], "ws_root": f"{HOME}/ws",
                                                             "snapshot": {"repo": paths.tilde(paths.MONOREPO)}})
    doc = json.loads(path.read_text())
    assert doc["ws_root"] == "~/ws" and doc["snapshot"]["repo"] == "."
    with pytest.raises(export.ExportError):
        export.write_results(tmp_path / "other.json", {})


@pytest.mark.parametrize(
    ("rel", "ok"),
    [
        ("results.json", True),
        ("claude_code/aider__transpose/calls.jsonl", True),
        ("hermes/mcpmark__papers__papers_counting__a2/summary.json", True),
        ("localmelo/warmup/check.txt", True),
        ("localmelo/warmup/prompt.md", False),
        ("claude_code/aider__transpose/claude-home/settings.json", False),
        ("claude_code/aider__transpose/workspace/transpose.py", False),
        ("codex/aider__transpose/calls.jsonl", False),
        ("claude_code/../calls.jsonl", False),
        ("state.json", False),
    ],
)
def test_is_allowlisted(rel: str, ok: bool) -> None:
    assert export.is_allowlisted(rel) is ok


def test_export_capability_reexports_every_listed_run(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    results = tmp_path / "results" / "capability" / "results.json"
    export.write_results(results, {"rows": [{"run_dir": "claude_code/aider__transpose"}], "superseded": [], "warmups": []})
    written = export.export_capability(run.parent.parent, results)
    assert len(written) == 3
    export.write_results(results, {"rows": [{"run_dir": "../../etc"}]})
    with pytest.raises(export.ExportError):
        export.export_capability(run.parent.parent, results)
