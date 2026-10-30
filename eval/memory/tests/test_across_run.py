"""The across-chats driver against a fake gateway: flow, isolation, resume, guards and preflight."""
import json
import socket
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import pytest
from memory.across import client, proxy, report, run, server
from memory.tests.across_fake import FakeGateway, FakeModelServer
from memory.tests.across_support import Launcher, git, make_repo, make_study


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def models():
    server_ = FakeModelServer(["chat-model", "embed-model"])
    yield server_
    server_.stop()


@pytest.fixture(autouse=True)
def real_home(tmp_path, monkeypatch):
    """The driver must never touch the real HOME: point it at an empty directory and check it stays empty."""
    home = tmp_path / "realhome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    yield home
    assert list(home.iterdir()) == []


def turns(study, cell):
    return [r for r in run.Journal(study.run_dir / "cells" / cell / "turns.jsonl").read() if r["event"] == "turn"]


def test_schedule_rotates_the_arms_across_repeats():
    cells = run.schedule(("M2", "L1", "B0", "N0"), 3)
    assert [c["cell_id"] for c in cells] == [
        "r1__M2", "r1__L1", "r1__B0", "r1__N0", "r2__L1", "r2__B0", "r2__N0", "r2__M2", "r3__B0", "r3__N0", "r3__M2", "r3__L1"]
    assert {c["arm"]: c["position"] for c in cells if c["repeat"] == 3} == {"B0": 1, "N0": 2, "M2": 3, "L1": 4}
    with pytest.raises(ValueError):
        run.schedule(("M2",), 0)
    with pytest.raises(ValueError):
        run.schedule(("M2", "M2"), 1)


def test_arms_are_data():
    assert run.ARMS["M2"].x == ("memory.retriever=embedding",) and run.ARMS["M2"].embedding
    assert run.ARMS["L1"].x == ("memory.retriever=llm",) and not run.ARMS["L1"].embedding
    assert run.ARMS["B0"].x == ("memory.retriever=bm25",)
    assert run.ARMS["N0"].x == ("memory.across_chats=false",)


def test_load_arms_replaces_the_builtin_arms(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({"E1": {"label": "x", "x": ["memory.retriever=embedding", "memory.k=3"], "embedding": True}}))
    arms = run.load_arms(path)
    assert arms["E1"].x == ("memory.retriever=embedding", "memory.k=3") and arms["E1"].embedding


def test_config_document_round_trips_through_toml(tmp_path):
    doc = server.config_document("http://127.0.0.1:8500/v1", "chat-model", "embed-model", run.DEFAULT_EXTRA_BODY)
    path = server.write_config(tmp_path, doc)
    parsed = tomllib.loads(path.read_text())
    assert parsed["chat_backend"] == "omlx" and parsed["embedding_backend"] == "omlx"
    assert parsed["vllm"]["embedding_model"] == "embed-model" and parsed["vllm"]["chat_model"] == "chat-model"
    assert parsed["vllm"]["extra_body"] == {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    plain = tomllib.loads(server.write_config(tmp_path / "b", server.config_document("http://127.0.0.1:1/v1", "m", None, None)).read_text())
    assert plain["embedding_backend"] == "none" and "embedding_model" not in plain["vllm"] and "extra_body" not in plain["vllm"]


def test_serve_command_and_environment(tmp_path):
    cmd = server.serve_command(Path("/x/melo"), 8123, ("memory.retriever=llm", "a.b=1"))
    assert cmd == ["/x/melo", "serve", "--port", "8123", "-X", "memory.retriever=llm", "-X", "a.b=1"]
    env = server.serve_env(tmp_path, {"PATH": "/usr/bin", "HOME": "/real", "EXAMPLE_API_KEY": "k", "EXAMPLE_TOKEN": "t", "SSH_TTY": "/dev/x", "KEEP": "1"})
    assert env["HOME"] == str(tmp_path) and env["BROWSER"] == "/usr/bin/true" and env["TMPDIR"] == str(tmp_path / "tmp")
    assert "EXAMPLE_API_KEY" not in env and "EXAMPLE_TOKEN" not in env and "SSH_TTY" not in env and env["KEEP"] == "1"


def test_full_flow_with_fresh_memory_per_cell(tmp_path, repo, models, real_home):
    launcher = Launcher()
    study = make_study(tmp_path, repo, models, launcher, repeats=2)
    assert run.run_study(study) == 0
    state = json.loads((study.run_dir / "across.json").read_text())
    assert state["complete"] and len(state["cells"]) == 6
    assert launcher.cells == ["r1__M2", "r1__L1", "r1__N0", "r2__L1", "r2__N0", "r2__M2"]
    env = json.loads((study.run_dir / "environment.json").read_text())
    assert env["start"]["sw_vers"] == "ProductName: test" and "end" in env
    order = json.loads((study.run_dir / "tasks.json").read_text())
    assert [f["id"] for f in order["teach_order"]] == ["F1", "F2", "F3", "F4"] and order["teach_order"][3]["supersedes"] == "F3"
    assert order["ask_order"] == ["Q2", "Q1", "Q3", "Q4", "Q5"]

    homes = [study.run_dir / "cells" / c / "home" for c in launcher.cells]
    assert len({str(h) for h in homes}) == 6
    for cell, gateway in zip(launcher.cells, launcher.gateways, strict=True):
        rows = turns(study, cell)
        assert [r["phase"] for r in rows] == ["teach"] * 4 + ["ask"] * 5 and all(r["state"] == "finished" for r in rows)
        config = tomllib.loads((study.run_dir / "cells" / cell / "home" / ".cache" / "localmelo" / "config.toml").read_text())
        assert ("embedding_model" in config["vllm"]) == cell.endswith("M2")
        deleted = [p for m, p in gateway.requests if m == "DELETE"]
        asked = [r["session_id"] for r in rows if r["phase"] == "ask"]
        assert sorted(deleted) == sorted(f"/api/sessions/{s}" for s in asked)
        if cell.endswith("N0"):
            assert not gateway.memory_path.exists() and all(r["memory"] is None and r["recall"] is None for r in rows)
        else:
            memory = json.loads(gateway.memory_path.read_text())
            # four teaching chats left their record and chat: the second repeat did not see the first's
            assert len(memory["records"]) == 3 and len(memory["documents"]) == 4
            assert all(not (study.run_dir / "cells" / cell / "home" / ".cache" / "localmelo" / "gateway" / "auth").exists() for _ in [0])
    assert real_home.exists() and list(real_home.iterdir()) == []


def test_what_a_turn_records(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, arm_names=("L1",))
    assert run.run_study(study) == 0
    rows = turns(study, "r1__L1")
    teach = {r["item"]: r for r in rows if r["phase"] == "teach"}
    ask = {r["item"]: r for r in rows if r["phase"] == "ask"}
    assert teach["F1"]["memory"]["remembered"][0]["text"] == "My cat is called Biscuit."
    q1 = ask["Q1"]
    assert q1["memory"]["used"][0]["session_id"] == teach["F1"]["session_id"]
    assert q1["recall"]["retriever"] == "llm" and q1["recall"]["fallback"] == "none"
    assert q1["recall"]["select_tokens"] == {"prompt": 640, "completion": 12}
    assert q1["answer"].startswith("From my notes")
    assert q1["wall_source"] == "stream" and q1["wall_ms"] > 0 and q1["ttfe_ms"] is not None
    assert q1["server_ms"]["run_started"] > 0 and q1["server_ms"]["finished"] >= q1["server_ms"]["first_message"]
    assert q1["usage"] == {"completion_tokens": 7} and q1["key"] == "acm-r1-L1-Q1"
    assert ask["Q5"]["memory"]["used"] == [] and ask["Q5"]["recall"]["used"] == 0  # memory on: the event comes with nothing used
    assert ask["Q5"]["answer"] == "I don't know."
    assert 0 < q1["started_epoch"] < q1["ended_epoch"] and q1["approvals_denied"] == 0 and q1["timed_out"] is False
    assert ask["Q4"]["answer"] == "From my notes: I live in Utrecht."  # the newer note replaced the older
    info = json.loads((study.run_dir / "cells" / "r1__L1" / "cell.json").read_text())
    assert info["state"] == "finished" and info["rss"]["before"]["rss_kb"] == 2048 and info["warmup"] == {"chat_ms": 10.0}
    assert set(info["rss"]) == {"before", "after_warmup", "after_teach", "after"}


def test_recall_is_null_when_the_build_does_not_report_it(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, Launcher(recall_timings=False), arm_names=("L1",))
    assert run.run_study(study) == 0
    asked = [r for r in turns(study, "r1__L1") if r["phase"] == "ask" and r["memory"]]
    assert asked and all(r["recall"] is None for r in asked) and all(r["memory"]["used"] for r in asked)


@pytest.mark.parametrize("launcher_kw, stream", [({}, False), ({"drop_stream": True}, True)])
def test_a_missing_stream_falls_back_to_polling(tmp_path, repo, models, launcher_kw, stream):
    study = make_study(tmp_path, repo, models, Launcher(**launcher_kw), arm_names=("L1",), use_stream=stream)
    assert run.run_study(study) == 0
    rows = {r["item"]: r for r in turns(study, "r1__L1") if r["phase"] == "ask"}
    assert all(r["wall_source"] == "poll" and r["ttfe_ms"] is None and r["state"] == "finished" for r in rows.values())
    assert rows["Q1"]["memory"]["used"] and rows["Q1"]["server_ms"]["run_started"] is not None  # events were read afterwards


def test_ask_chats_are_deleted_so_questions_do_not_feed_each_other(tmp_path, repo, models):
    def second_question_used(keep):
        study = make_study(tmp_path / ("keep" if keep else "drop"), repo, models, arm_names=("B0",), keep_ask_chats=keep)
        assert run.run_study(study) == 0
        rows = {r["item"]: r for r in turns(study, "r1__B0") if r["phase"] == "ask"}
        first, second = [q.id for q in study.asked() if q.id in ("Q1", "Q2")]
        return len(rows[second]["memory"]["used"]), rows[first]["session_id"], rows[second]["memory"]["used"]

    kept_n, first_session, kept_used = second_question_used(True)
    dropped_n, _, _ = second_question_used(False)
    assert dropped_n == 1 and kept_n == 2
    assert first_session in [u["session_id"] for u in kept_used]


def test_a_busy_chat_is_deleted_on_retry(tmp_path, repo, models):
    launcher = Launcher(busy_deletes=2)
    study = make_study(tmp_path, repo, models, launcher, arm_names=("L1",))
    assert run.run_study(study) == 0
    rows = run.Journal(study.run_dir / "cells" / "r1__L1" / "turns.jsonl").read()
    assert all(r["ok"] for r in rows if r["event"] == "deleted") and len([r for r in rows if r["event"] == "deleted"]) == 5


def test_stop_then_resume_continues_in_the_same_home(tmp_path, repo, models):
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 6 else None  # after 5 turns: 4 teaching, 1 question

    launcher = Launcher()
    study = make_study(tmp_path, repo, models, launcher, arm_names=("L1", "N0"), turn_guard=guard)
    assert run.run_study(study) == 3
    state = json.loads((study.run_dir / "across.json").read_text())
    assert state["cells"]["r1__L1"]["state"] == "paused" and not state["complete"]
    assert len(turns(study, "r1__L1")) == 5 and "r1__N0" not in state["cells"]
    assert "stopping" in (study.run_dir / "across.log").read_text()

    study.turn_guard = lambda: None
    assert run.run_study(study) == 0
    assert launcher.cells == ["r1__L1", "r1__L1", "r1__N0"]  # the second start is the same cell, same HOME
    rows = turns(study, "r1__L1")
    assert len(rows) == 9 and len({r["key"] for r in rows}) == 9  # nothing was sent twice
    info = json.loads((study.run_dir / "cells" / "r1__L1" / "cell.json").read_text())
    assert info["resumed"] == 1 and info["state"] == "finished"
    memory = json.loads(launcher.gateways[1].memory_path.read_text())
    assert len(memory["records"]) == 3  # teaching was not repeated
    assert not list((study.run_dir / "cells").glob("*.aborted-*"))


def test_a_question_chat_left_behind_by_a_crash_is_deleted_on_resume(tmp_path, repo, models):
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 7 else None  # after 4 teaching turns and 2 questions

    study = make_study(tmp_path, repo, models, arm_names=("L1",), turn_guard=guard)
    assert run.run_study(study) == 3
    path = study.run_dir / "cells" / "r1__L1" / "turns.jsonl"
    rows = run.Journal(path).read()
    last_deleted = max(i for i, r in enumerate(rows) if r["event"] == "deleted")
    left_behind = rows[last_deleted]["session_id"]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for i, r in enumerate(rows) if i != last_deleted))  # the crash
    study.turn_guard = lambda: None
    assert run.run_study(study) == 0
    deleted = [r for r in run.Journal(path).read() if r["event"] == "deleted" and r["session_id"] == left_behind]
    assert len(deleted) == 1 and deleted[0]["ok"]


def test_a_cell_that_died_in_a_turn_is_set_aside_and_run_again(tmp_path, repo, models):
    launcher = Launcher()
    calls = []

    def guard():
        calls.append(1)
        if len(calls) == 3:
            launcher.gateways[-1].stop()  # the gateway dies before the third message
        return None

    study = make_study(tmp_path, repo, models, launcher, arm_names=("L1",), turn_guard=guard)
    assert run.run_study(study) == 1
    state = json.loads((study.run_dir / "across.json").read_text())
    assert state["cells"]["r1__L1"]["state"] == "error"
    rows = run.Journal(study.run_dir / "cells" / "r1__L1" / "turns.jsonl").read()
    assert rows[-1]["event"] == "start"  # a message that never got a result

    study.turn_guard = lambda: None
    assert run.run_study(study) == 0
    assert (study.run_dir / "cells" / "r1__L1.aborted-1").is_dir()
    assert len(turns(study, "r1__L1")) == 9 and len(launcher.gateways) == 2
    memory = json.loads(launcher.gateways[1].memory_path.read_text())
    assert len(memory["records"]) == 3  # a fresh HOME: nothing from the dead run


def test_failed_turn_is_recorded_and_the_cell_goes_on(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, Launcher(fail_texts=("roommate",)), arm_names=("L1",))
    assert run.run_study(study) == 0
    rows = {r["item"]: r for r in turns(study, "r1__L1")}
    assert rows["Q3"]["state"] == "failed" and rows["Q3"]["status"] == "failed" and rows["Q3"]["error"] == "boom"
    assert rows["Q4"]["state"] == "finished"


def test_a_changed_study_needs_a_fresh_directory(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, arm_names=("L1",))
    assert run.run_study(study) == 0
    again = make_study(tmp_path, repo, models, arm_names=("L1",), model="other")
    assert run.run_study(again) == 2  # preflight: the model is not listed
    again = make_study(tmp_path, repo, models, arm_names=("L1", "N0"))
    assert run.run_study(again) == 2
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "notes.txt").write_text("x")
    assert run.run_study(make_study(tmp_path, repo, models, run_dir=tmp_path / "stray", arm_names=("L1",))) == 2


def test_question_selection_teaches_only_what_it_needs(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, arm_names=("N0",), question_ids=("Q4",))
    assert [f.id for f in study.taught()] == ["F3", "F4"] and [q.id for q in study.asked()] == ["Q4"]
    assert run.run_study(study) == 0
    assert [r["item"] for r in turns(study, "r1__N0")] == ["F3", "F4", "Q4"]


def test_preflight_reports_everything_wrong(tmp_path, repo, models):
    assert run.preflight(make_study(tmp_path, repo, models)) == []
    bad_sha = make_study(tmp_path, repo, models, expect_sha="deadbeef")
    assert "not deadbeef" in run.preflight(bad_sha)[0]
    (repo / "stray.txt").write_text("x")
    assert "uncommitted changes" in run.preflight(make_study(tmp_path, repo, models))[0]
    (repo / "stray.txt").unlink()
    only_chat = FakeModelServer(["chat-model"])
    try:
        assert any("embedding model" in p for p in run.preflight(make_study(tmp_path, repo, only_chat)))
        assert run.preflight(make_study(tmp_path, repo, only_chat, arm_names=("L1", "N0"))) == []
        assert any("chat model 'nope'" in p for p in run.preflight(make_study(tmp_path, repo, only_chat, model="nope")))
    finally:
        only_chat.stop()
    unknown = dict(run.ARMS, X9=run.Arm("X9", "x", ("memory.nope=1",), False))
    problems = run.preflight(make_study(tmp_path, repo, models, arms=unknown, arm_names=("X9",)))
    assert problems == ["arm X9: this localmelo build does not know the setting memory.nope"]
    assert any("does not exist" in p for p in run.preflight(make_study(tmp_path, tmp_path / "nowhere", models, expect_sha="abc")))
    assert any("question Q99" in p for p in run.preflight(make_study(tmp_path, repo, models, question_ids=("Q99",))))
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{dead.getsockname()[1]}/v1"
    dead.close()
    assert any("not readable" in p for p in run.preflight(make_study(tmp_path, repo, models, base_url=url)))


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the preflight's throwaway HOME
    argv = ["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--localmelo-sha", git(repo, "rev-parse", "HEAD")[:8],
            "--base-url", models.base_url, "--model", "chat-model", "--embedding-model", "embed-model", "--dry-run",
            "--gpu-lock", str(tmp_path / "no-lock")]
    assert run.main(argv) == 0
    out = capsys.readouterr().out
    assert "12 cells = 4 arms x 3 repeats" in out and "Arm M2: melo serve -X memory.retriever=embedding (+ embed-model)" in out
    assert "Arm N0: melo serve -X memory.across_chats=false" in out and "Preflight: ok" in out
    assert "r2__M2" in out and "the GPU lock is absent" in out
    assert not (tmp_path / "run").exists()
    assert run.main([*argv[:-2], "--gpu-lock", str(tmp_path / "no-lock"), "--arms", "L1", "--repeats", "1", "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        run.main([*argv, "--arms", "Z9"])


def test_guards(tmp_path):
    lock = tmp_path / "lock"
    turn_guard, cell_guard = run.make_guards(tmp_path, lock, "agent-benchmark", 40, 20)
    assert "absent" in turn_guard()
    lock.write_text("holder: someone-else\npurpose: x\n")
    assert "held by another session (holder: someone-else)" in turn_guard()
    lock.write_text("holder: agent-benchmark\npurpose: x\n")
    assert turn_guard() is None
    (tmp_path / "STOP").write_text("")
    assert turn_guard() == "STOP file found" and cell_guard() == "STOP file found"
    assert run.battery_problem(20, lambda: (True, 19)) == "on battery at 19% (below 20%)"
    assert run.battery_problem(20, lambda: (True, 25)) is None and run.battery_problem(20, lambda: (False, 5)) is None
    assert run.battery_problem(40, lambda: (True, None)) is None


def test_a_real_gateway_process_runs_isolated(tmp_path, repo, models, monkeypatch):
    """server.launch starts the fake `melo serve` for real: no tty, no browser, its own HOME, stopped at the end."""
    monkeypatch.setenv("ACROSS_TEST_SECRET_KEY", "hunter2")
    study = make_study(tmp_path, repo, models, server.launch, arm_names=("L1",), question_ids=("Q1",))
    assert run.run_study(study) == 0
    cell = study.run_dir / "cells" / "r1__L1"
    seen = json.loads((cell / "home" / "env.json").read_text())
    assert seen["home"] == str(cell / "home") and seen["browser"] == "/usr/bin/true"
    assert seen["stdin_tty"] is False and seen["cwd"] == str(cell / "home") and seen["secret_visible"] is False
    assert seen["argv"][:3] == ["serve", "--port", str(json.loads((cell / "cell.json").read_text())["port"])]
    assert seen["argv"][3:] == ["-X", "memory.retriever=llm"]
    info = json.loads((cell / "cell.json").read_text())
    assert info["gateway_exit"] == 0
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", info["port"])) != 0  # stopped
    assert [r["item"] for r in turns(study, "r1__L1")] == ["F1", "Q1"]


def test_a_gateway_that_exits_at_start_is_reported(tmp_path, repo, models):
    melo = repo / ".venv" / "bin" / "melo"
    melo.write_text(f"#!{sys.executable}\nimport sys\nprint('config error: boom')\nsys.exit(3 if sys.argv[1] == 'serve' else 0)\n")
    # `config get` must still succeed for the preflight, so print a value for it
    melo.write_text(f"#!{sys.executable}\nimport sys\nif sys.argv[1] == 'config':\n    print('x')\n    sys.exit(0)\nprint('config error: boom')\nsys.exit(3)\n")
    study = make_study(tmp_path, repo, models, server.launch, arm_names=("L1",))
    assert run.run_study(study) == 1
    assert "melo serve exited with 3" in (study.run_dir / "across.log").read_text()
    assert "config error: boom" in (study.run_dir / "across.log").read_text()


def cell_config(study, cell):
    return tomllib.loads((study.run_dir / "cells" / cell / "home" / ".cache" / "localmelo" / "config.toml").read_text())["vllm"]


def port_of(url):
    return int(url.split("//")[1].split("/")[0].split(":")[1])


def test_model_calls_go_through_the_proxy_and_are_logged_per_cell(tmp_path, repo, models):
    launcher = Launcher(model_calls=True)
    study = make_study(tmp_path, repo, models, launcher)  # M2, L1, N0, one repeat of the 9-turn mini set
    assert run.run_study(study) == 0
    logs = {c: proxy.read_log(study.run_dir / "cells" / c / "proxy.jsonl") for c in ("r1__M2", "r1__L1", "r1__N0")}
    for cell in logs:
        assert cell_config(study, cell)["chat_url"] != models.base_url  # the gateway was given the proxy, not the model server
        assert cell_config(study, cell)["chat_url"].endswith("/v1")
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port_of(cell_config(study, cell)["chat_url"]))) != 0  # and it is stopped now
    chat = lambda rows, tools: [r for r in rows if r["path"].endswith("/chat/completions") and r["tools"] is tools]
    embeds = lambda rows: [r for r in rows if r["path"].endswith("/embeddings")]
    assert len(chat(logs["r1__M2"], True)) == len(chat(logs["r1__L1"], True)) == len(chat(logs["r1__N0"], True)) == 9
    assert all(r["stream"] and r["model"] == "chat-model" for r in chat(logs["r1__M2"], True))
    assert len(embeds(logs["r1__M2"])) >= 9 and {r["model"] for r in embeds(logs["r1__M2"])} == {"embed-model"}  # the query of each turn
    assert embeds(logs["r1__L1"]) == [] and embeds(logs["r1__N0"]) == []  # no second model without M2
    assert len(chat(logs["r1__L1"], False)) == 9 and chat(logs["r1__N0"], False) == []  # L1's chat model picks the notes, N0 recalls nothing
    assert len(models.calls) == sum(len(v) for v in logs.values())  # everything the gateways sent upstream was logged
    for r in turns(study, "r1__M2"):  # every turn's window (POST to end) holds the agent's chat request, so the report can match them
        assert any(r["started_epoch"] <= c["t_start"] <= r["ended_epoch"] for c in chat(logs["r1__M2"], True)), r["item"]
    info = json.loads((study.run_dir / "cells" / "r1__M2" / "cell.json").read_text())
    assert info["proxy"] == {"log": "proxy.jsonl", "upstream": models.base_url}


def test_without_the_proxy_the_gateway_talks_to_the_model_server(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, Launcher(model_calls=True), arm_names=("M2",), use_proxy=False)
    assert run.run_study(study) == 0
    assert cell_config(study, "r1__M2")["chat_url"] == models.base_url and not (study.run_dir / "cells" / "r1__M2" / "proxy.jsonl").exists()
    assert models.calls  # the gateway's calls went straight there


def test_a_resumed_cell_gets_a_new_proxy_and_keeps_one_log(tmp_path, repo, models):
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 6 else None

    study = make_study(tmp_path, repo, models, Launcher(model_calls=True), arm_names=("L1",), turn_guard=guard)
    assert run.run_study(study) == 3
    first_url = cell_config(study, "r1__L1")["chat_url"]
    before = len(proxy.read_log(study.run_dir / "cells" / "r1__L1" / "proxy.jsonl"))
    study.turn_guard = lambda: None
    assert run.run_study(study) == 0
    assert cell_config(study, "r1__L1")["chat_url"] != first_url  # a new port, written into the same HOME's config
    assert len(proxy.read_log(study.run_dir / "cells" / "r1__L1" / "proxy.jsonl")) > before


@pytest.mark.parametrize("use_stream", [True, False])
def test_a_tool_approval_is_denied_and_counted(tmp_path, repo, models, use_stream):
    launcher = Launcher(approval_texts=("Biscuit",))  # the model tries to save "My cat is called Biscuit." to a file
    study = make_study(tmp_path, repo, models, launcher, arm_names=("L1",), use_stream=use_stream, teach_timeout_s=20.0)
    assert run.run_study(study) == 0
    teach = {r["item"]: r for r in turns(study, "r1__L1") if r["phase"] == "teach"}
    assert (teach["F1"]["approvals_denied"], teach["F1"]["denied_tools"], teach["F1"]["state"], teach["F1"]["timed_out"]) == (1, ["file_write"], "finished", False)
    assert all(r["approvals_denied"] == 0 for item, r in teach.items() if item != "F1")
    assert [a["outcome"] for a in launcher.gateways[0].approvals.values()] == ["denied"]
    assert any(m == "POST" and p.startswith("/api/approvals/") for m, p in launcher.gateways[0].requests)
    info = json.loads((study.run_dir / "cells" / "r1__L1" / "cell.json").read_text())
    assert info["counts"]["teach"] == {"turns": 4, "failed": 0, "timed_out": 0, "approvals_denied": 1, "turns_with_denied": 1}
    assert info["counts"]["ask"]["approvals_denied"] == 0


def test_a_teaching_turn_that_does_not_end_is_stopped_and_counted(tmp_path, repo, models):
    launcher = Launcher(hang_texts=("小满",))
    study = make_study(tmp_path, repo, models, launcher, arm_names=("L1",), teach_timeout_s=0.3)
    assert run.run_study(study) == 0
    rows = {r["item"]: r for r in turns(study, "r1__L1")}
    hung = rows["F2"]
    assert (hung["state"], hung["status"], hung["timed_out"], hung["cancelled"]) == ("failed", "timeout", True, True)
    assert 300 <= hung["wall_ms"] < 2500 and "did not finish" in hung["error"]
    assert rows["F3"]["state"] == "finished" and rows["Q3"]["state"] == "finished"  # the cell goes on, and questions have their own limit
    info = json.loads((study.run_dir / "cells" / "r1__L1" / "cell.json").read_text())
    assert info["counts"]["teach"]["timed_out"] == 1 and info["counts"]["teach"]["failed"] == 1 and info["counts"]["ask"]["timed_out"] == 0


def test_a_question_turn_has_its_own_timeout(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, Launcher(hang_texts=("board game",)), arm_names=("N0",), turn_timeout_s=0.3)
    assert run.run_study(study) == 0
    row = {r["item"]: r for r in turns(study, "r1__N0")}["Q5"]
    assert row["timed_out"] and row["state"] == "failed"
    assert json.loads((study.run_dir / "cells" / "r1__N0" / "cell.json").read_text())["counts"]["ask"]["timed_out"] == 1


def test_the_warmup_loads_only_the_models_an_arm_uses(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, warm=None)  # the real warm-up requests
    assert run.run_study(study) == 0
    assert sorted((c["path"].rsplit("/", 1)[1], c["model"]) for c in models.calls) == [
        ("completions", "chat-model"), ("completions", "chat-model"), ("completions", "chat-model"), ("embeddings", "embed-model")]
    info = {c: json.loads((study.run_dir / "cells" / c / "cell.json").read_text()) for c in ("r1__M2", "r1__L1", "r1__N0")}
    assert set(info["r1__M2"]["warmup"]) == {"chat_ms", "embed_ms"} and set(info["r1__L1"]["warmup"]) == set(info["r1__N0"]["warmup"]) == {"chat_ms"}
    assert all(set(i["rss"]) == {"before", "after_warmup", "after_teach", "after"} for i in info.values())


def test_the_model_server_state_lists_what_is_loaded(models, tmp_path):
    state = server.loaded_models(models.base_url)
    assert state["loaded"] == ["chat-model", "embed-model"] and state["status"]["models_loaded"] == 2
    assert {m["id"]: m["loaded"] for m in state["models"]} == {"chat-model": True, "embed-model": True}
    quiet = FakeModelServer(["chat-model", "embed-model"], loaded=["chat-model"])
    try:
        assert server.model_server_state(quiet.base_url)["loaded"] == ["chat-model"]
        assert {"pid", "rss_kb", "at", "loaded", "status", "models"} <= set(server.model_server_state(quiet.base_url))
    finally:
        quiet.stop()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}/v1"
    assert server.loaded_models(dead) == {"loaded": None}


def test_approvals_and_cancel_answer_as_the_gateway_does(tmp_path):
    fake = FakeGateway(tmp_path, [], approval_texts=("save",))
    try:
        gw = client.Gateway(fake.base_url, "fake-token", timeout=5.0)
        assert gw.deny("nope", "x") is False  # 404: no such request
        done = client.run_turn(gw, "please save this", "k-1")
        assert done.status == "completed" and [d["tool"] for d in done.denied] == ["file_write"]
        assert gw.deny(done.denied[0]["request_id"], "again") is False  # 409: answered already
        assert gw.cancel_turn(done.turn_id) is False  # 409: not running
        assert client.run_turn(gw, "save it but do not deny", "k-2", deny_approvals=False, timeout_s=0.3, use_stream=False).status == "timeout"
    finally:
        fake.stop()


def test_the_default_pin_and_models(tmp_path, repo, models, capsys, monkeypatch):
    assert run.LOCALMELO_SHA == "5f1498f4" and run.CHAT_MODEL == "Qwen3.8-Flash-Next-oQ4e-mtp" and run.EMBEDDING_MODEL == "Qwen3-Embedding-0.6B"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    args = run.parse_args(["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--base-url", models.base_url])
    assert (args.localmelo_sha, args.model, args.embedding_model, args.teach_timeout, args.no_proxy) == ("5f1498f4", run.CHAT_MODEL, run.EMBEDDING_MODEL, 60.0, False)
    assert run.main(["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--base-url", models.base_url, "--dry-run",
                     "--gpu-lock", str(tmp_path / "none")]) == 2  # the fake checkout is not at 5f1498f4
    out = capsys.readouterr().out
    assert "not 5f1498f4" in out and "Teaching turns stop after 60 s" in out and "timing proxy" in out


def cell_info(study, cell):
    return json.loads((study.run_dir / "cells" / cell / "cell.json").read_text())


def test_every_cell_starts_with_the_embedding_model_unloaded(tmp_path, repo):
    models = FakeModelServer(["chat-model", "embed-model"])  # both loaded to begin with, as a server left running would be
    try:
        study = make_study(tmp_path, repo, models, warm=None, probe_rss=lambda: server.model_server_state(models.base_url))
        assert run.run_study(study) == 0
        # M2 (loaded at the start) and L1 (left loaded by M2's warm-up) were asked to unload it; N0 found it unloaded
        assert models.unloads == ["embed-model", "embed-model"] and "chat-model" not in models.unloads
        cold = {c: cell_info(study, c)["cold_start"] for c in ("r1__M2", "r1__L1", "r1__N0")}
        assert [cold[c]["was_loaded"] for c in cold] == [True, True, False] and all(c["loaded_after"] is False for c in cold.values())
        assert cold["r1__M2"]["unload_status"] == 200 and cold["r1__N0"]["attempts"] == 0 and cold["r1__M2"]["model"] == "embed-model"
        for c in cold:
            rss = cell_info(study, c)["rss"]
            assert "embed-model" not in rss["before"]["loaded"] and "chat-model" in rss["before"]["loaded"]  # cold for one model, not for the chat model
            assert ("embed-model" in rss["after_warmup"]["loaded"]) == c.endswith("M2")  # only M2's warm-up loads it again
        assert cell_info(study, "r1__M2")["warmup"].keys() == {"chat_ms", "embed_ms"}
        rep = report.analyse(study.run_dir, study.tasks)
        assert rep["cold_start"] == {"cells": 3, "cold": 3, "unloaded_by_driver": 2, "skipped": 0}
        assert "Every cell started with the embedding model unloaded (3 of 3 cells; the driver unloaded it in 2, the chat model never)." in report.render(rep)
    finally:
        models.stop()


def test_the_driver_waits_for_the_unload_to_finish(tmp_path, repo):
    models = FakeModelServer(["chat-model", "embed-model"], unload_delay=0.3)
    try:
        study = make_study(tmp_path, repo, models, arm_names=("N0",), sleep=time.sleep)
        assert run.run_study(study) == 0
        cold = cell_info(study, "r1__N0")["cold_start"]
        assert cold["was_loaded"] and cold["wait_ms"] >= 250 and cold["attempts"] == 1 and models.unloads == ["embed-model"]
    finally:
        models.stop()


def test_a_model_that_will_not_unload_stops_the_run_before_the_cell(tmp_path, repo, capsys):
    models = FakeModelServer(["chat-model", "embed-model"], stuck=True)
    try:
        launcher = Launcher()
        study = make_study(tmp_path, repo, models, launcher, arm_names=("L1",), unload_timeout_s=0.4, sleep=time.sleep)
        assert run.run_study(study) == 2
        assert "Preflight failed" in capsys.readouterr().err and launcher.gateways == []  # no gateway was started
        assert "still loaded" in (study.run_dir / "across.log").read_text() and "cold_start" not in cell_info(study, "r1__L1")
        models.stuck = False  # the server lets go: the same command goes on, and the cell does its unload first
        assert run.run_study(study) == 0 and cell_info(study, "r1__L1")["cold_start"]["was_loaded"]
    finally:
        models.stop()


def test_a_server_that_does_not_say_what_is_loaded_cannot_be_cold(tmp_path, repo, capsys):
    models = FakeModelServer(["chat-model", "embed-model"], status_routes=False)
    try:
        assert run.run_study(make_study(tmp_path, repo, models, arm_names=("L1",))) == 2
        assert "cannot tell whether 'embed-model' is loaded" in capsys.readouterr().err and models.unloads == []
    finally:
        models.stop()


def test_cold_start_can_be_switched_off_and_the_report_then_says_so(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, arm_names=("L1",), cold_start=False)
    assert run.run_study(study) == 0
    assert models.unloads == [] and cell_info(study, "r1__L1")["cold_start"] == {"model": "embed-model", "skipped": True}
    rep = report.analyse(study.run_dir, study.tasks)
    assert rep["cold_start"]["skipped"] == 1 and "**Not every cell started with the embedding model unloaded**" in report.render(rep)
    assert run.parse_args(["--run-dir", "x", "--localmelo", "y", "--base-url", "z", "--no-cold-start", "--unload-timeout", "5"]).no_cold_start


def test_a_continued_cell_does_not_unload_the_model_it_loaded(tmp_path, repo):
    models = FakeModelServer(["chat-model", "embed-model"])
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 4 else None

    try:
        study = make_study(tmp_path, repo, models, arm_names=("M2",), warm=None, turn_guard=guard)
        assert run.run_study(study) == 3 and models.unloads == ["embed-model"] and "embed-model" in models.loaded  # M2's warm-up loaded it
        study.turn_guard = lambda: None
        assert run.run_study(study) == 0
        assert models.unloads == ["embed-model"] and "embed-model" in models.loaded  # still one unload, still loaded
        assert cell_info(study, "r1__M2")["resumed"] == 1 and cell_info(study, "r1__M2")["cold_start"]["was_loaded"]
    finally:
        models.stop()


def test_ensure_unloaded_by_itself(models):
    quiet = FakeModelServer(["chat-model"], loaded=["chat-model"])  # the embedding model is not even listed
    try:
        assert server.ensure_unloaded(quiet.base_url, "embed-model") == {
            "model": "embed-model", "was_loaded": False, "unload_status": None, "attempts": 0, "wait_ms": 0.0, "loaded_after": False}
        assert quiet.unloads == []
    finally:
        quiet.stop()
    assert server.model_is_loaded({"models": [{"id": "a", "loaded": True}, {"id": "b", "loaded": False}]}, "a") is True
    assert server.model_is_loaded({"models": [{"id": "a", "loaded": True}]}, "b") is False and server.model_is_loaded({"loaded": ["a"]}, "a") is True
    assert server.model_is_loaded({"loaded": None}, "a") is None
    assert server.unload_model(models.base_url, "no-such-model") == 404 and server.unload_model("http://127.0.0.1:9/v1", "x") == 0
    assert server.unload_model(models.base_url, "embed-model") == 200 and server.unload_model(models.base_url, "embed-model") in (200, 400)


def test_admin_routes_carry_the_key_and_nothing_else_does(tmp_path) -> None:
    """oMLX's status and unload routes need its API key; the key is sent there only and comes from a command."""
    import http.server
    import threading

    from memory.across import run as across_run
    from memory.across import server as across_server

    seen: list[tuple[str, str | None]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self) -> None:
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path in ("/api/status", "/v1/models/status") and self.headers.get("Authorization") != "Bearer k-123":
                self.send_response(401)
                self.end_headers()
                return
            body = {"/api/status": {"loaded_models": ["E"]}, "/v1/models/status": {"models": [{"id": "E", "loaded": True}]}}
            data = json.dumps(body.get(self.path, {"data": [{"id": "E"}]})).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _answer

        def log_message(self, *args) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        across_server.set_admin_token(None)
        assert across_server.loaded_models(base)["loaded"] is None  # 401 without the key
        script = tmp_path / "key.sh"
        script.write_text("#!/bin/sh\necho k-123\n")
        script.chmod(0o700)
        across_server.set_admin_token(across_run.admin_token(str(script)))
        assert across_server.loaded_models(base)["loaded"] == ["E"]
        assert across_server.unload_model(base, "E") == 200
        assert across_server.model_ids(base)[0] == ["E"]
        auth = {path: header for path, header in seen[2:]}
        assert auth["/api/status"] == auth["/v1/models/status"] == auth["/v1/models/E/unload"] == "Bearer k-123"
        assert auth["/v1/models"] is None  # the open routes never get the key
        with pytest.raises(SystemExit, match="output is not shown"):
            across_run.admin_token("/usr/bin/false")
    finally:
        across_server.set_admin_token(None)
        httpd.shutdown()
