"""Native T4 driver, store-everything and Hippo routing checks; no model server needed."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from memory.t4 import gen, native, routing, scenario
from memory.t4 import store as st
from memory.t4.oracle import Fixtures


@pytest.fixture
def facts() -> dict[str, Any]:
    return gen.build(1)["facts"]


def record_ids(facts: dict[str, Any]) -> dict[str, str]:
    return {r["key"]: rid for rid, r in facts["records"].items() if r["key"]}


def decision(ids: list[str], action: str = "NEW", entry: int | None = 1, **extra: Any) -> dict[str, Any]:
    return {"source_ids": ids, "action": action, "entry_id": entry, **extra}


def test_later_version_labels(facts: dict[str, Any]) -> None:
    assert Counter(v["change"] for item in facts["items"] for v in item["versions"][1:]) == {"replaces": 12, "adds": 4}
    assert Counter(v["change"] for item in scenario.ITEMS for v in item["versions"][1:]) == {"replaces": 12, "adds": 4}
    assert facts == Fixtures().facts


def test_gold_sequence_routes_every_version(facts: dict[str, Any]) -> None:
    versions = sorted((v["day"], key, i, v) for key, item in enumerate(facts["items"]) for i, v in enumerate(item["versions"]))
    decisions = [decision(v["source_ids"], "NEW" if i == 0 else "UPDATE", entry=n)
                 for n, (_, _, i, v) in enumerate(versions, 1)]
    result = routing.check(facts, decisions)
    assert result["per_pair"]["errors"] == 0
    assert result["per_row"]["unscorable"] == 0
    assert result["missed_versions"] == []


@pytest.mark.parametrize(("action", "ok", "conflict"), [("UPDATE", True, False), ("CONFLICT", True, True),
                                                       ("NEW", False, False), ("IGNORE", False, False)])
def test_replaces(facts: dict[str, Any], action: str, ok: bool, conflict: bool) -> None:
    ids = record_ids(facts)
    result = routing.check(facts, [decision([ids["R1"]]), decision([ids["R1"], ids["R2"]], action)])
    pair = result["pairs"][-1]
    assert pair["expected"] == ["UPDATE"] and pair["accepted"] is ok
    assert pair["conflict_left_open"] is conflict
    if conflict:
        assert {"item": "rent", "version": 2, "day": 4} in result["missed_versions"]


@pytest.mark.parametrize("action", ["NEW", "UPDATE", "RELATED"])
def test_adds(facts: dict[str, Any], action: str) -> None:
    ids = record_ids(facts)
    result = routing.check(facts, [decision([ids["G1"]]), decision([ids["G1"], ids["G2"]], action)])
    assert result["pairs"][-1]["accepted"] is True
    assert result["pairs"][-1]["expected"] == ["UPDATE", "RELATED"]


def test_only_admitted_state_advances(facts: dict[str, Any]) -> None:
    rid = record_ids(facts)["R2"]
    rows = [decision([rid], "IGNORE", None), decision([rid], "RELATED"), decision([rid], "IGNORE")]
    pairs = routing.check(facts, rows)["pairs"]
    assert [p["expected"] for p in pairs] == [["NEW"], ["NEW"], ["IGNORE"]]
    assert [p["accepted"] for p in pairs] == [False, True, True]


def test_introduction_and_support_do_not_credit_future_or_duplicate_items(facts: dict[str, Any]) -> None:
    ids = record_ids(facts)
    result = routing.check(facts, [decision([ids["C1"]]), decision([ids["C1"], ids["C4"]])])
    assert [p["item"] for p in result["pairs"]] == ["sister", "family_dinner"]
    assert result["per_pair"]["n"] == result["per_row"]["n"] == 2


def test_multi_item_rows_have_independent_states(facts: dict[str, Any]) -> None:
    ids = record_ids(facts)
    result = routing.check(facts, [decision([ids["A1"]]), decision([ids["A1"], ids["R1"]])])
    assert result["per_pair"] == {"n": 3, "accepted": 2, "errors": 1, "conflict_left_open": 0}
    assert result["per_row"]["n"] == 2 and result["per_row"]["errors"] == 1
    assert {p["item"]: p["expected"] for p in result["pairs"][1:]} == {"offsite": ["IGNORE"], "rent": ["NEW"]}


def test_unscorable_and_distractor_decisions_are_reported(facts: dict[str, Any]) -> None:
    ids = record_ids(facts)
    distractor = next(rid for rid, r in facts["records"].items() if r["key"] is None)
    result = routing.check(facts, [decision([]), decision(["not-a-pab-record"]), decision([ids["A2"]]),
                                  decision([distractor]), decision([distractor], "IGNORE", None)])
    assert result["per_row"]["unscorable"] == 3
    assert result["per_row"]["distractor_admitted"] == 1
    assert result["per_row"]["errors"] == 0
    assert result["rows"][1]["unknown_source_ids"] == ["not-a-pab-record"]
    assert result["rows"][2]["unmapped_source_ids"] == [ids["A2"]]


@pytest.mark.parametrize(("action", "entry", "admitted", "accepted"), [
    ("NEW", 1, True, True), ("RELATED", 1, True, True),
    ("NEW", None, False, True), ("RELATED", None, False, True),
    ("IGNORE", None, False, True), ("IGNORE", 1, False, True),
    ("UPDATE", 1, False, False), ("CONFLICT", 1, False, False),
])
def test_distractor_admission_table(facts: dict[str, Any], action: str, entry: int | None,
                                    admitted: bool, accepted: bool) -> None:
    distractor = next(rid for rid, r in facts["records"].items() if r["key"] is None)
    result = routing.check(facts, [decision([distractor], action, entry)])
    assert result["rows"][0]["distractor_admitted"] is admitted
    assert result["rows"][0]["accepted"] is accepted
    assert result["per_row"] == {"n": 1, "scorable": 1, "accepted": int(accepted),
                                  "errors": int(not accepted), "unscorable": 0,
                                  "distractor_admitted": int(admitted)}


def test_scripted_retrieval_and_reader_replies() -> None:
    from memory.tests.native_stub import observed_fact, response

    fact = observed_fact([{"role": "tool", "content": json.dumps({"results": [
        {"kind": "event", "title": "Yoga", "start": "2026-04-11T09:00:00-04:00", "location": "Studio Verde"}]})}])
    assert fact == "Yoga starts 2026-04-11T09:00:00-04:00 at Studio Verde."
    got = response({"messages": [{"role": "user", "content": "[m1] fact, [m2] fact. Return at most 5 ids."}]})
    assert json.loads(got["choices"][0]["message"]["content"]) == {"ids": ["m1", "m2"]}
    got = response({"messages": [{"role": "system", "content": "You answer questions about Jamie's life"},
                                 {"role": "user", "content": "Question?"}]})
    assert json.loads(got["choices"][0]["message"]["content"]) == {"answer": None}


def test_everything_keeps_ignored_duplicates_and_requires_text(tmp_path: Path) -> None:
    rows = [decision(["r1"], text="Same fact", seq=1), decision(["r1"], "IGNORE", text="Same fact", seq=2)]
    out = tmp_path / "everything.jsonl"
    assert native.rebuild_everything(rows, out) == 2
    entries = st.load(out)
    assert entries == [{"id": "decision-1", "text": "Same fact", "source_ids": ["r1"]},
                       {"id": "decision-2", "text": "Same fact", "source_ids": ["r1"]}]
    with pytest.raises(ValueError, match="keep-decision-text"):
        native.rebuild_everything([decision(["r1"], text=None)], out)


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch, facts: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, Any] = {"days": [], "exports": [], "evaluations": []}
    monkeypatch.setattr(native, "identity", lambda *args: {"protocol": "test", "localmelo_sha": "stub", "model": "stub"})

    def run_day(prompt: str, workspace: Path, run_dir: Path, cfg: Path, trial: Path, localmelo: Path) -> dict[str, Any]:
        day = int(workspace.parent.name[-2:])
        seen["days"].append(day)
        config = json.loads(cfg.read_text())["mcpServers"]["pab"]
        assert config["args"][:2] == ["-I", "-B"]
        assert config["args"][-3:] == ["t4-native", "--fixtures", str(native.FIXTURES)]
        assert "--session" in config["args"] and str(day) in config["args"]
        assert 'status "done"' in prompt and "remember" in prompt
        assert (trial / "home").is_dir()
        run_dir.mkdir()
        summary = {"exit_code": 2 if day == 2 else 0, "timed_out": day == 2,
                   "steps": {"n_llm": 3}, "wall_s": 0.5}
        native.save(run_dir / "summary.json", summary)
        (run_dir / "stdout.log").write_text(json.dumps({"task_id": f"task-{day}", "status": "completed"}) + "\n")
        return summary

    def export(localmelo: Path, trial: Path, flag: str, target: Path) -> None:
        seen["exports"].append(flag)
        rows = [decision([record_ids(facts)["A1"]], text="Fact", seq=i, task_id=f"task-{i}", admitted_by="request")
                for i in range(1, 11)]
        if flag == "--export-store":
            rows = [{"text": "Fact", "source_ids": [record_ids(facts)["A1"]]}]
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(native, "_run_day", run_day)
    monkeypatch.setattr(native, "_export", export)
    monkeypatch.setattr(native, "_evaluate", lambda trial, store, *args: seen["evaluations"].append((trial, store)))
    seen["run_day"] = run_day
    return seen


@pytest.mark.parametrize(("missing", "code"), [(True, 0), (True, 2), (False, 0), (False, 2)])
@pytest.mark.parametrize("flag", ["--export-store", "--export-decisions"])
def test_missing_store_exports_and_real_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              missing: bool, code: int, flag: str) -> None:
    import subprocess
    from types import SimpleNamespace

    (tmp_path / "mem").mkdir()
    if not missing:
        (tmp_path / "mem/long_term.db").touch()
    target = tmp_path / "export.jsonl"

    def run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        kwargs["stdout"].write('{"text":"candidate"}\n' if not code else "")
        kwargs["stderr"].write("missing file" if code else "exported")
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(native.subprocess, "run", run)
    if not missing and code:
        with pytest.raises(subprocess.CalledProcessError):
            native._export(tmp_path / "repo", tmp_path, flag, target)
        assert not target.exists()
    else:
        native._export(tmp_path / "repo", tmp_path, flag, target)
        assert st.load(target) == ([] if missing else [{"id": "1", "text": "candidate", "source_ids": []}])
    diagnostic = json.loads(target.with_suffix(".diagnostic.json").read_text())
    assert diagnostic == {"exit_code": code, "stderr": "missing file" if code else "exported", "missing_store": missing}


def test_harness_budget_flags_and_shared_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import harness, manifest

    seen = {}
    repo = tmp_path / "localmelo"
    monkeypatch.setattr(manifest, "LOCALMELO_REPO", repo)

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert args[0] == "localmelo"
        return {}

    monkeypatch.setattr(harness, "run", run)
    cfg = tmp_path / "pab.json"
    for day in (1, 2):
        native._run_day("remember", tmp_path / "ws", tmp_path / f"day{day}", cfg, tmp_path, repo)
        assert seen["max_steps"] == 30 and seen["timeout_s"] == 900 and seen["thinking"] == "off"
        assert seen["harness_args"] == ["--mcp-only", "--mcp-config", str(cfg), "--reasoning", "compress",
                                        "--keep-decision-text", "--output-format", "json"]
        assert seen["extra_env"]["LOCALMELO_MEMORY_DIR"] == str(tmp_path / "mem")
        assert seen["extra_env"]["HOME"] == str(tmp_path / "home")


def test_proxy_budget_refusal_is_recorded(tmp_path: Path) -> None:
    (tmp_path / "calls.jsonl").write_text(json.dumps({"budget_exhausted": True}) + "\n")
    result = native.session_result(1, tmp_path, {"timed_out": False, "steps": {"n_llm": 30}})
    assert result["budget_exhausted"] and not result["timed_out"] and result["model_calls"] == 30


def test_ten_days_budget_and_resume(tmp_path: Path, driver: dict[str, Any]) -> None:
    trial = tmp_path / "trial"
    result = native.run_trial(trial)
    assert result["complete"] and driver["days"] == list(range(1, 11))
    assert result["sessions"]["2"]["budget_exhausted"] is True
    assert result["sessions"]["3"]["model_calls"] == 3
    assert result["sessions"]["1"]["admitted_by"] == {"request": 1}
    assert len(driver["exports"]) == len(driver["evaluations"]) == 2
    assert (trial / "comparators/store-everything/events").resolve() == trial / "events"
    assert len(st.load(trial / "t4/store-everything.jsonl")) == 10
    native.run_trial(trial)
    assert driver["days"] == list(range(1, 11)) and len(driver["evaluations"]) == 2


def test_interrupted_day_is_not_replayed(tmp_path: Path, driver: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    trial = tmp_path / "trial"

    def interrupted(*args: Any) -> dict[str, Any]:
        if args[1].parent.name == "day03":
            args[2].mkdir()
            (args[2] / "stdout.log").write_text('{"unfinished":')
            raise KeyboardInterrupt
        return driver["run_day"](*args)

    monkeypatch.setattr(native, "_run_day", interrupted)
    with pytest.raises(KeyboardInterrupt):
        native.run_trial(trial)
    monkeypatch.setattr(native, "_run_day", driver["run_day"])
    result = native.run_trial(trial)
    assert driver["days"] == [1, 2, *range(4, 11)]
    assert result["sessions"]["3"]["state"] == "interrupted"


def test_evaluation_failure_resumes_without_ingesting_again(tmp_path: Path, driver: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    trial = tmp_path / "trial"

    def fail(*args: Any) -> None:
        raise RuntimeError("reader offline")

    monkeypatch.setattr(native, "_evaluate", fail)
    with pytest.raises(RuntimeError, match="reader offline"):
        native.run_trial(trial)
    monkeypatch.setattr(native, "_evaluate", lambda *args: None)
    native.run_trial(trial)
    assert len(driver["days"]) == 10 and len(driver["exports"]) == 2


def test_changed_identity_and_unowned_directory_are_refused(tmp_path: Path, driver: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    trial = tmp_path / "trial"
    native.run_trial(trial)
    monkeypatch.setattr(native, "identity", lambda *args: {"model": "different"})
    with pytest.raises(ValueError, match="changed"):
        native.run_trial(trial)
    other = tmp_path / "other"
    other.mkdir()
    (other / "history.db").touch()
    with pytest.raises(ValueError, match="Non-empty"):
        native.run_trial(other)
