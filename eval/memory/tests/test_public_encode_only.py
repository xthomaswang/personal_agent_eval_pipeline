"""Encode-only checkpoint/stop behavior on invented data and a fake ingest only."""
from dataclasses import replace
import json
from pathlib import Path
import signal

import pytest

from memory.public import encode_only as enc
from memory.public import run
from memory.tests.public_support import mini_questions

PRIVATE_TEXT = "Synthetic question text that must never survive in output."


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A real process/network/gateway call is forbidden in this test")

    monkeypatch.setattr(run.subprocess, "run", forbidden)
    monkeypatch.setattr(run, "ask", forbidden)
    monkeypatch.setattr(run.judge, "judge", forbidden)
    monkeypatch.setattr(run.server, "model_ids", forbidden)
    monkeypatch.setattr(run.server, "git_sha", lambda path: "51c7a59e-fixed")
    monkeypatch.setattr(run.server, "git_dirty", lambda path: False)
    monkeypatch.setattr(enc.across, "admin_token", lambda command: None)
    monkeypatch.setattr(enc.server, "set_admin_token", lambda token: None)


@pytest.fixture
def study(tmp_path):
    questions = [{"question_id": "q0", "question": PRIVATE_TEXT}, {"question_id": "q1", "question": PRIVATE_TEXT}]
    return run.Study(
        run_dir=tmp_path / "encode", questions=questions, ids=("q0", "q1"), arms={}, arm_names=(), repeats=0,
        localmelo=tmp_path / "app", expect_sha="51c7a59e", base_url="synthetic", model="synthetic-model",
        embedding_model="synthetic-embedding", judge_model="", ingest_x=("ingest=compress",),
        snapshot=lambda study: {"sw_vers": "synthetic", "xcodebuild_version": "synthetic"},
    )


def report(**updates):
    return {"compress_summary": {"windows": 5, "cached": 3, "failed": 1},
            "cache_hits": 4, "model_calls": 2, "seconds": 1.25, "error": "", **updates}


def checkpoint(study):
    return json.loads((study.run_dir / "encode.json").read_text())


def fake_ingest(monkeypatch, seen, *, after=None, result=None, error=False):
    def ingest(study, question, qdir):
        seen.append(question["question_id"])
        # Mimic both a store and a sensitive proxy/raw report inside the private qdir.
        (qdir / "template").mkdir()
        (qdir / "template/store.json").write_text(question["question"])
        (qdir / "ingest_proxy.jsonl").write_text(question["question"])
        if after:
            after(study, question)
        if error:
            raise run.IngestError(question["question"])
        return result if result is not None else report()

    monkeypatch.setattr(run, "ingest", ingest)


def test_complete_full_ingests_keep_only_numeric_metadata_and_skip_on_resume(study, monkeypatch, capsys):
    seen = []
    fake_ingest(monkeypatch, seen)
    assert enc.encode(study) == 0
    state = checkpoint(study)
    assert state["complete"] is True and seen == ["q0", "q1"]
    assert all(row["state"] == "done" for row in state["rows"])
    assert state["rows"][0] == {
        "question_id": "q0", "state": "done", "metrics_available": True, "windows": 5,
        "cached_windows": 3, "failed_windows": 1, "cache_hits": 4, "model_calls": 2,
        "cache_misses": 2, "seconds": 1.25,
    }
    assert list(study.work.iterdir()) == []
    assert sorted(p.name for p in study.run_dir.iterdir()) == ["encode.json", "work"]
    assert PRIVATE_TEXT not in (study.run_dir / "encode.json").read_text()
    assert enc.encode(study, deadline=0) == 0  # all done, no work even after a deadline
    assert seen == ["q0", "q1"] and len(checkpoint(study)["rows"]) == 2
    assert PRIVATE_TEXT not in capsys.readouterr().out


def test_stop_file_is_a_boundary_stop_not_a_failure(study, monkeypatch):
    seen = []
    fake_ingest(monkeypatch, seen, after=lambda study, question: (study.run_dir / "STOP").touch())
    assert enc.encode(study) == 75
    state = checkpoint(study)
    assert seen == ["q0"] and state["complete"] is False and state["stop_reason"] == "STOP"
    assert list(study.work.iterdir()) == []


def test_signal_stop_then_resume_skips_completed_question(study, monkeypatch):
    requested = [False]
    seen = []
    fake_ingest(monkeypatch, seen, after=lambda *args: requested.__setitem__(0, True))
    assert enc.encode(study, stop_requested=lambda: requested[0]) == 75
    assert checkpoint(study)["stop_reason"] == "signal"
    requested[0] = False
    fake_ingest(monkeypatch, seen)
    assert enc.encode(study, stop_requested=lambda: requested[0]) == 0
    assert seen == ["q0", "q1"] and len(checkpoint(study)["rows"]) == 2


def test_deadline_stops_before_the_next_question(study, monkeypatch):
    now = [0.0]
    seen = []
    fake_ingest(monkeypatch, seen, after=lambda *args: now.__setitem__(0, 10.0))
    assert enc.encode(study, deadline=10.0, clock=lambda: now[0]) == 75
    assert seen == ["q0"] and checkpoint(study)["stop_reason"] == "deadline"


def test_expired_deadline_and_guard_do_not_start_an_ingest(study, monkeypatch):
    seen = []
    fake_ingest(monkeypatch, seen)
    assert enc.encode(study, deadline=1.0, clock=lambda: 1.0) == 75
    guarded = replace(study, run_dir=study.run_dir.parent / "guarded", cell_guard=lambda: "synthetic guard")
    assert enc.encode(guarded) == 75
    assert checkpoint(guarded)["stop_reason"] == "guard" and seen == []


def test_failed_process_is_not_done_its_text_is_redacted_and_its_store_is_removed(study, monkeypatch, capsys):
    seen = []
    fake_ingest(monkeypatch, seen, error=True)
    assert enc.encode(study) == 1
    row = checkpoint(study)["rows"][0]
    assert row == {"question_id": "q0", "state": "failed", "metrics_available": False, "error_type": "IngestError"}
    assert list(study.work.iterdir()) == []
    assert PRIVATE_TEXT not in (study.run_dir / "encode.json").read_text()
    assert PRIVATE_TEXT not in capsys.readouterr().out
    fake_ingest(monkeypatch, seen)
    assert enc.encode(study) == 0
    assert seen == ["q0", "q0", "q1"]  # the failed attempt is retained, not counted done
    assert len(checkpoint(study)["rows"]) == 3


def test_top_level_error_retains_available_metrics_without_error_prose(study, monkeypatch, capsys):
    fake_ingest(monkeypatch, [], result=report(error=PRIVATE_TEXT))
    assert enc.encode(study) == 1
    row = checkpoint(study)["rows"][0]
    assert row["metrics_available"] and row["failed_windows"] == 1 and row["state"] == "failed"
    assert PRIVATE_TEXT not in (study.run_dir / "encode.json").read_text()
    assert PRIVATE_TEXT not in capsys.readouterr().out


def test_resume_rejects_different_inputs_before_ingesting_or_overwriting(study, monkeypatch):
    fake_ingest(monkeypatch, [])
    assert enc.encode(study) == 0
    before = (study.run_dir / "encode.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        enc.encode(replace(study, model="different-model"))
    assert (study.run_dir / "encode.json").read_bytes() == before


@pytest.mark.parametrize("options", [
    {"encode_share": 0.5}, {"encode_turn_share": 0.9}, {"keep_answer_sessions_plain": True},
    {"keep_homes": True}, {"ingest_x": ()},
])
def test_only_full_compress_ingestion_is_allowed(study, options):
    with pytest.raises(ValueError):
        enc.encode(replace(study, **options))
    assert not study.run_dir.exists()


@pytest.mark.parametrize("bad", [
    report(compress_summary={"windows": 1, "cached": 2, "failed": 0}),
    report(cache_hits=-1), report(seconds=float("nan")), report(model_calls=True),
])
def test_invalid_metrics_cannot_become_a_completed_question(bad):
    with pytest.raises(ValueError):
        enc.metrics(bad)


@pytest.mark.parametrize("received_signal", [signal.SIGTERM, signal.SIGINT])
def test_term_and_int_handlers_request_stopping_and_restore_the_previous_handlers(received_signal):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with enc.stop_signals() as requested:
        assert not requested()
        signal.getsignal(received_signal)(received_signal, None)  # invoke the handler, send no real signal
        assert requested()
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


@pytest.mark.parametrize("relative_vendor", [False, True])
def test_cli_builds_the_same_ingest_settings_without_a_vendor_build(tmp_path, monkeypatch, relative_vendor):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "synthetic.json"
    dataset.write_text(json.dumps(mini_questions()))
    selected = tmp_path / "synthetic-ids.json"
    selected.write_text(json.dumps({"ids": ["q1aa", "q2bb_abs"]}))
    app = tmp_path / "stub-app"
    (app / ".venv/bin").mkdir(parents=True)
    (app / ".venv/bin/melo").write_text("Synthetic executable placeholder, never run.")
    monkeypatch.setattr(enc.across, "make_guards", lambda *args: (lambda: None, lambda: None))
    args = enc.parse_args([
        "--run-dir", str(tmp_path / "encode"), "--localmelo", "stub-app" if relative_vendor else str(app),
        "--localmelo-sha", "51c7a59e", "--base-url", "synthetic",
        "--data", str(dataset), "--questions-file", str(selected), "--compress-cache", str(tmp_path / "cache.json"),
    ])
    built = enc.build_study(args)
    assert built.ids == ("q1aa", "q2bb_abs") and not built.arm_names and not built.judge_model
    assert built.ingest_x == ("ingest=compress",)
    assert built.ingest_settings == {"memory.compress_parallel": 4, "memory.compress_cache": str(tmp_path / "cache.json")}
    assert built.encode_share is built.encode_turn_share is None and not built.keep_answer_sessions_plain
    assert not built.cold_start and not built.do_warmup and not built.keep_homes
    assert not built.run_dir.exists()
    built.snapshot = lambda study: {"synthetic": True}
    seen = []

    def fake_subprocess(command, **kwargs):
        # Exercise real run.ingest/run_ingest, but emulate exec's cwd resolution;
        # no child process, socket, gateway, vendor code or model is ever run.
        executable = Path(command[0])
        actual = executable if executable.is_absolute() else Path(kwargs["cwd"]) / executable
        if not actual.is_file():
            raise FileNotFoundError(2, "Synthetic executable is absent from the child cwd", command[0])
        assert command[1:3] == ["memory", "ingest"]
        seen.append((executable, Path(kwargs["cwd"])))
        raw = {"sessions": 1, "turns": 2, "compress": {"synthetic": {"windows": [{"cached": False, "failed": False}]}},
               "cache_hits": 0, "model_calls": 1, "seconds": 0.1, "error": ""}
        return run.subprocess.CompletedProcess(command, 0, json.dumps(raw), "")

    monkeypatch.setattr(run.subprocess, "run", fake_subprocess)
    assert enc.encode(built) == 0
    assert len(seen) == 2 and all(executable == app / ".venv/bin/melo" for executable, cwd in seen)
    assert all(cwd != tmp_path and "template" == cwd.name for executable, cwd in seen)
    assert built.localmelo == app.resolve()
    assert list(built.work.iterdir()) == []


def test_main_passes_night_deadline_and_stopped_code(study, monkeypatch):
    monkeypatch.setenv("NIGHT_DEADLINE", "12345")
    monkeypatch.setattr(enc, "build_study", lambda args: study)
    received = []

    def fake_encode(study, **kwargs):
        received.append(kwargs["deadline"])
        assert not kwargs["stop_requested"]()
        return 75

    monkeypatch.setattr(enc, "encode", fake_encode)
    assert enc.main(["--run-dir", "/synthetic", "--localmelo", "/synthetic", "--localmelo-sha", "stub",
                     "--data", "/synthetic.json"]) == 75
    assert received == [12345.0]
