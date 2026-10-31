"""The driver on a v2 task file against the fake gateway: arms, the per-kind arm table, long chats, workspaces, two-question
chats, timeouts, planning, and that v1 runs keep their protocol and identity."""
import json
import tempfile
import tomllib

import pytest
from memory.across import client, data, data_v2, run, server, stores
from memory.tests.across_fake import FakeModelServer, mini_raw_v2
from memory.tests.across_support import git, make_repo, make_study, v2_launcher, v2_run, v2_study


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


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return v2_run(tmp_path_factory)


def turns(study, cell):
    return [r for r in run.Journal(study.run_dir / "cells" / cell / "turns.jsonl").read() if r["event"] == "turn"]


def cell_dir(study, cell):
    return study.run_dir / "cells" / cell


# --- arms ---


def test_fc_is_n0_with_a_full_window_and_no_embedding_model():
    n0, fc = run.ARMS["N0"], run.ARMS["FC"]
    assert n0.x == run.MEMORY_OFF_X and not n0.embedding
    assert fc.x == run.MEMORY_OFF_X and fc.config == (("agent.window_messages", 256),) and not fc.embedding  # the embedding backend stays "none": no step memory either
    assert not any("window_messages" in x for x in fc.x)  # `melo serve -X agent.window_messages=N` exits: "only for direct mode"
    assert run.WINDOW_MESSAGES_FULL == 256 and fc.v1 is False and n0.v1 is True
    assert run.memory_is_off(n0.x) and run.memory_is_off(fc.x) and not run.memory_is_off(run.ARMS["M2"].x) and not run.memory_is_off(run.ARMS["B0"].x)
    assert run.DEFAULT_ARMS == ("M2", "L1", "B0", "N0") and run.DEFAULT_ARMS_V2 == ("M2", "B0", "N0", "FC")


def test_fc_cell_starts_memory_off_with_a_256_window_and_no_embedding_backend(ran):
    study = ran.study
    fc = next(g for g, c in zip(ran.launcher.gateways, ran.launcher.cells, strict=True) if c == "r1__FC")
    n0 = next(g for g, c in zip(ran.launcher.gateways, ran.launcher.cells, strict=True) if c == "r1__N0")
    assert fc.x_pairs == n0.x_pairs == ("memory.across_chats=false",)  # the cap is not a -X setting
    assert (fc.window_cap, n0.window_cap) == (256, 64)
    for cell in ("r1__FC", "r1__N0"):
        config = tomllib.loads((cell_dir(study, cell) / "home" / ".cache" / "localmelo" / "config.toml").read_text())
        assert config["embedding_backend"] == "none" and "embedding_model" not in config["vllm"]
        assert config.get("agent") == ({"window_messages": 256} if cell == "r1__FC" else None)  # [agent] window_messages = 256, for FC only
        assert json.loads((cell_dir(study, cell) / "cell.json").read_text()).get("config") == ({"agent.window_messages": 256} if cell == "r1__FC" else None)
    assert json.loads((study.run_dir / "across.json").read_text())["config"]["arms"]["FC"] == {"x": ["memory.across_chats=false"], "embedding": False, "config": {"agent.window_messages": 256}}


def test_a_window_cap_passed_as_x_is_refused_as_melo_serve_refuses_it(tmp_path, repo, models):
    arms = dict(run.ARMS, FC=run.Arm("FC", "wrong", (*run.MEMORY_OFF_X, "agent.window_messages=256"), False, v1=False))
    study = v2_study(tmp_path, repo, models, arms=arms, arm_names=("FC",))
    assert run.run_study(study) == 1
    assert "only for direct mode" in (study.run_dir / "across.log").read_text()


def test_arm_config_entries_are_written_into_nested_tables_of_the_config(tmp_path):
    doc = server.config_document("http://127.0.0.1:1/v1", "m", None, None, {"agent.window_messages": 256, "agent.window_eviction": "slide", "request_timeout_s": 30})
    parsed = tomllib.loads(server.write_config(tmp_path, doc).read_text())
    assert parsed["agent"] == {"window_messages": 256, "window_eviction": "slide"} and parsed["request_timeout_s"] == 30 and parsed["vllm"]["chat_model"] == "m"
    assert "agent" not in server.config_document("http://127.0.0.1:1/v1", "m", None, None)
    with pytest.raises(ValueError):
        server.config_document("http://127.0.0.1:1/v1", "m", None, None, {"vllm.chat_model.x": 1})


def test_each_kind_lists_the_arms_that_run_it(ran, models, repo, tmp_path):
    study = v2_study(tmp_path, repo, models, arm_names=("N0", "M2", "B0", "FC"))
    ids = lambda arm: [i.id for i in study.items(arm)]
    assert ids("M2") == ids("N0") and sorted(ids("M2")) == ["L1", "L2", "P1", "P2"]
    assert all(i.startswith("L") for i in ids("FC")) and sorted(ids("FC")) == ["L1", "L2"]
    assert all(i.startswith("P") for i in ids("B0")) and sorted(ids("B0")) == ["P1", "P2"]
    assert [i.kind for i in study.items("M2")] == ["longrange", "longrange", "followup", "followup"]  # kinds in the table's order
    assert study.items("L1") == ()  # no kind lists it
    assert {c: sum(r["phase"] == "long" for r in turns(ran.study, c)) for c in ("r1__N0", "r1__M2", "r1__B0", "r1__FC")} == {
        "r1__N0": 65, "r1__M2": 65, "r1__B0": 0, "r1__FC": 65}
    only = v2_study(tmp_path / "x", repo, models, question_ids=("L2", "P1"))
    assert [i.id for i in only.items()] == ["L2", "P1"] and [i.id for i in only.items("FC")] == ["L2"]


def test_preflight_names_arms_with_nothing_to_run(tmp_path, repo, models):
    problems = run.preflight(v2_study(tmp_path, repo, models, arm_names=("M2", "L1")))
    assert problems == ["arm L1 has nothing to run here: the selected items are of kinds that do not list it (it runs no kind)"]
    assert run.preflight(v2_study(tmp_path, repo, models, arm_names=("FC",), question_ids=("P1",))) == [
        "arm FC has nothing to run here: the selected items are of kinds that do not list it (it runs ['longrange'])"]
    assert run.preflight(v2_study(tmp_path, repo, models, arm_names=("N0", "M2", "FC"))) == []  # FC's setting is known to the build
    v1 = make_study(tmp_path, repo, models, arm_names=("FC",))
    assert run.preflight(v1) == ["arm FC runs only the in-chat kinds of a v2 task file, not the v1 questions"]


# --- the cells ---


def test_a_v2_study_finishes_and_each_cell_runs_its_kinds(ran):
    study = ran.study
    assert ran.code == 0
    state = json.loads((study.run_dir / "across.json").read_text())
    assert state["complete"] and sorted(state["cells"]) == ["r1__B0", "r1__FC", "r1__M2", "r1__N0"]
    by_phase = lambda cell: {p: sum(r["phase"] == p for r in turns(study, cell)) for p in ("long", "teach", "ask")}
    assert by_phase("r1__N0") == by_phase("r1__M2") == {"long": 65, "teach": 4, "ask": 6}  # L1: 32+1, L2: 33+1, two follow-ups: 2+2 each
    assert by_phase("r1__B0") == {"long": 0, "teach": 4, "ask": 4} and by_phase("r1__FC") == {"long": 65, "teach": 0, "ask": 2}
    assert all(r["state"] == "finished" for c in ("r1__N0", "r1__M2", "r1__B0", "r1__FC") for r in turns(study, c))
    info = json.loads((cell_dir(study, "r1__FC") / "cell.json").read_text())
    assert info["counts"]["long"] == {"turns": 65, "failed": 0, "timed_out": 0, "approvals_denied": 0, "turns_with_denied": 0}
    assert info["counts"]["teach"]["turns"] == 0 and info["counts"]["ask"]["turns"] == 2
    assert "after_teach" not in info["rss"]  # there is no single teaching phase


def test_a_long_chat_is_one_chat_in_its_own_workspace_and_stays(ran):
    study = ran.study
    n0 = next(g for g, c in zip(ran.launcher.gateways, ran.launcher.cells, strict=True) if c == "r1__N0")
    rows = turns(study, "r1__N0")
    for item in ("L1", "L2"):
        mine = [r for r in rows if r["item"].startswith(item + ".")]
        sessions = {r["session_id"] for r in mine}
        assert len(sessions) == 1 and len(mine) in (33, 34)  # the opener, the fact, 30 filler turns and the question in one chat
        session = sessions.pop()
        assert [r["item"] for r in mine][-1] == f"{item}.q1" and session in n0.sessions  # the chat is kept, not deleted
        assert n0.sessions[session]["workspace"] == str((cell_dir(study, "r1__N0") / "work" / item).resolve())
        assert ("DELETE", f"/api/sessions/{session}") not in n0.requests
    assert len({r["session_id"] for r in rows if r["phase"] == "long"}) == 2
    assert all(r["key"].startswith("acm-r1-N0-") for r in rows) and len({r["key"] for r in rows}) == len(rows)


def test_each_long_chat_gets_the_fixture_files(ran):
    for cell in ("r1__N0", "r1__M2", "r1__FC"):
        for item in ("L1", "L2"):
            work = cell_dir(ran.study, cell) / "work" / item
            assert sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()) == ["docs/faq.md", "notes.txt", "todo.md"]
            assert all(len(p.read_text()) <= 300 for p in work.rglob("*") if p.is_file()) and not (work / ".git").exists()  # nothing reads git any more
    assert not (cell_dir(ran.study, "r1__B0") / "work" / "L1").exists()  # B0 runs no long chat


def test_prepare_workspace_writes_the_fixtures_and_leaves_a_resumed_chat_alone(tmp_path):
    tasks = data_v2.parse(mini_raw_v2(), None)
    path = tmp_path / "chat"
    run.prepare_workspace(tasks, path)
    (path / "todo.md").write_text("changed by the chat\n")
    run.prepare_workspace(tasks, path)  # a resumed chat keeps its workspace as it is
    assert (path / "todo.md").read_text() == "changed by the chat\n" and (path / "docs" / "faq.md").read_text() == "Sample docs/faq.md.\n"


def test_a_long_chat_turn_records_the_tools_it_used_and_v1_rows_do_not(ran, tmp_path, repo, models):
    rows = turns(ran.study, "r1__N0")
    filler = [r for r in rows if r["phase"] == "long" and r["item"].startswith("L1.")]
    used = [t for r in filler for t in r["tools"]]
    assert set(used) == {"file_read", "list_files"} and len(used) == 16 and all(r["tools"] == [] for r in rows if r["phase"] != "long")
    v1 = make_study(tmp_path, repo, models, arm_names=("L1",), question_ids=("Q1",))
    assert run.run_study(v1) == 0
    assert all("tools" not in r for r in turns(v1, "r1__L1"))


def test_the_run_memory_stats_are_kept_when_a_build_sends_them(ran):
    stats = {"recalls": 2, "recall_chars": 120, "recall_log": [{"x": 1}], "recalls_by_trigger": {"start": 1}, "flag": True}
    res = client.TurnResult("s", "t", 1, "completed", events=[{"type": "run.finished", "data": {"result": {"memory_stats": stats}}}])
    assert run.run_memory_stats(res) == {"recalls": 2, "recall_chars": 120}
    assert run.run_memory_stats(client.TurnResult("s", "t", 1, "completed", events=[{"type": "run.finished", "data": {"result": {"status": "completed"}}}])) == {}
    assert not any("memory_stats" in r for r in turns(ran.study, "r1__M2"))  # the default (legacy) mode does not send them


def test_a_followup_item_has_two_teaching_chats_and_one_two_turn_ask_chat(ran):
    study = ran.study
    m2 = next(g for g, c in zip(ran.launcher.gateways, ran.launcher.cells, strict=True) if c == "r1__M2")
    rows = turns(study, "r1__M2")
    for item in ("P1", "P2"):
        f1, f2, q1, q2 = (next(r for r in rows if r["item"] == f"{item}.{k}") for k in ("f1", "f2", "q1", "q2"))
        assert len({f1["session_id"], f2["session_id"], q1["session_id"]}) == 3  # three chats
        assert q2["session_id"] == q1["session_id"]  # the follow-up is the second turn of the question's chat
        assert (f1["phase"], q1["phase"]) == ("teach", "ask") and rows.index(f1) < rows.index(q1)
        assert ("DELETE", f"/api/sessions/{q1['session_id']}") in m2.requests  # deleted after the item
        assert ("DELETE", f"/api/sessions/{f1['session_id']}") not in m2.requests and f1["session_id"] in m2.sessions
    assert [t["turn"] for t in m2.sessions[next(r for r in rows if r["item"] == "P1.f1")["session_id"]]["turns"]] == [1]  # a teaching chat has one turn
    deleted = [r for r in run.Journal(cell_dir(study, "r1__M2") / "turns.jsonl").read() if r["event"] == "deleted"]
    assert sorted(r["item"] for r in deleted) == ["P1", "P2"] and all(r["ok"] for r in deleted)


def test_followup_ask_chats_are_kept_on_request(tmp_path, repo, models):
    launcher = v2_launcher()
    study = v2_study(tmp_path, repo, models, launcher, arm_names=("B0",), keep_ask_chats=True)
    assert run.run_study(study) == 0
    assert not [r for r in run.Journal(cell_dir(study, "r1__B0") / "turns.jsonl").read() if r["event"] == "deleted"]
    assert not any(method == "DELETE" for method, _ in launcher.gateways[0].requests) and len(launcher.gateways[0].sessions) == 6  # 4 teaching chats and 2 ask chats


def test_a_failed_first_question_still_gets_its_follow_up_in_the_same_chat(tmp_path, repo, models):
    launcher = v2_launcher(fail_texts=("Who is my advisor",))  # the first question's turn fails, but its chat exists
    study = v2_study(tmp_path, repo, models, launcher, arm_names=("B0",), question_ids=("P1",))
    assert run.run_study(study) == 0
    rows = {r["item"]: r for r in turns(study, "r1__B0")}
    assert rows["P1.q1"]["state"] == "failed" and rows["P1.q2"]["state"] == "finished"
    assert rows["P1.q2"]["session_id"] == rows["P1.q1"]["session_id"]


def test_a_follow_up_is_not_sent_into_a_new_chat_when_the_first_question_started_none(tmp_path, repo, models):
    study = v2_study(tmp_path, repo, models, arm_names=("B0",), question_ids=("P1",))
    cell = {"cell_id": "c", "repeat": 1, "arm": "B0"}
    calls = []

    def send(phase, item, text, **kw):
        calls.append(item)
        return run.turn_record(cell, phase, item, "k", None, "refused")  # no session came back

    live = run.Live(study, run.ARMS["B0"], cell, None, run.Journal(tmp_path / "j.jsonl"), tmp_path, {"turns": {("teach", "P1.f1"): {}, ("teach", "P1.f2"): {}}, "deleted": set()}, send)
    assert run.run_followup(live, [study.tasks.item("P1")]) is False
    assert calls == ["P1.q1"]
    assert [(r["item"], r["state"], r["error"]) for r in run.Journal(tmp_path / "j.jsonl").read()] == [("P1.q2", "failed", "the first question started no chat")]


def test_long_chat_turns_have_their_own_timeout_and_the_chat_goes_on(tmp_path, repo, models):
    launcher = v2_launcher(hang_texts=("report line 3",), hang_seconds=3.0)
    study = v2_study(tmp_path, repo, models, launcher, arm_names=("FC",), question_ids=("L1",), long_timeout_s=0.3, teach_timeout_s=60.0, turn_timeout_s=60.0)
    assert run.run_study(study) == 0
    rows = turns(study, "r1__FC")
    hung = [r for r in rows if r["timed_out"]]
    assert len(hung) == 1 and hung[0]["state"] == "failed" and hung[0]["cancelled"] and hung[0]["phase"] == "long" and 300 <= hung[0]["wall_ms"] < 2500
    assert rows[-1]["item"] == "L1.q1" and rows[-1]["state"] == "finished" and len({r["session_id"] for r in rows}) == 1  # the question is asked in the same chat
    assert json.loads((cell_dir(study, "r1__FC") / "cell.json").read_text())["counts"]["long"]["timed_out"] == 1
    assert run.LONG_TIMEOUT_S == 120.0 and run.Study.__dataclass_fields__["long_timeout_s"].default == 120.0  # the default; --long-timeout changes it


def test_the_question_of_a_long_chat_uses_the_long_timeout_too(tmp_path, repo, models):
    launcher = v2_launcher(hang_texts=("Which room",), hang_seconds=3.0)
    study = v2_study(tmp_path, repo, models, launcher, arm_names=("FC",), question_ids=("L1",), long_timeout_s=0.3, turn_timeout_s=60.0)
    assert run.run_study(study) == 0
    assert turns(study, "r1__FC")[-1]["timed_out"]


def test_a_stopped_long_chat_continues_in_its_own_session(tmp_path):
    """The journal says which turns of a chat are done; the rest go into the same session, nothing is sent twice."""
    tasks = data_v2.parse(mini_raw_v2(), None)
    item = tasks.item("L1")
    steps = tasks.long_steps(item)
    sent = []

    def send(phase, label, text, **kw):
        sent.append((label, kw.get("session_id"), kw.get("timeout_s")))
        return {"event": "turn", "phase": phase, "item": label, "session_id": kw.get("session_id") or "chat-1", "timed_out": False, "state": "finished"}

    done = {"turns": {(s.phase, s.label): {"session_id": "chat-1"} for s in steps[:10]}, "deleted": set()}
    study = type("S", (), {"tasks": tasks, "turn_guard": staticmethod(lambda: None), "long_timeout_s": 120.0, "sleep": staticmethod(lambda s: None)})()
    live = run.Live(study, run.ARMS["FC"], {}, None, None, tmp_path, done, send)
    run.prepare_workspace(tasks, tmp_path / "L1")  # the workspace exists already
    assert run.run_longrange(live, [item]) is False
    assert [label for label, _, _ in sent] == [s.label for s in steps[10:]]
    assert {session for _, session, _ in sent} == {"chat-1"} and {t for _, _, t in sent} == {120.0}


def test_a_guard_stops_the_cell_inside_a_long_chat(tmp_path, repo, models):
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 12 else None

    study = v2_study(tmp_path, repo, models, arm_names=("FC",), question_ids=("L1",), turn_guard=guard)
    assert run.run_study(study) == 3
    assert len(turns(study, "r1__FC")) == 11 and json.loads((study.run_dir / "across.json").read_text())["cells"]["r1__FC"]["state"] == "paused"


# --- planning and identity ---


def test_the_v2_dry_run_plans_by_arm_and_writes_nothing(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    argv = ["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--localmelo-sha", git(repo, "rev-parse", "HEAD")[:8],
            "--base-url", models.base_url, "--model", "chat-model", "--embedding-model", "embed-model", "--dry-run",
            "--gpu-lock", str(tmp_path / "no-lock"), "--tasks", str(data_v2.DATA_FILE), "--questions", "L01,L02,P01,P02"]
    assert run.main(argv) == 0
    out = capsys.readouterr().out
    assert "Memory-bench-v2 plan: 4 cells = 4 arms x 1 repeats, 4 items (longrange 2, followup 2)" in out
    assert "turns after 120 s" not in out and "turns of a long chat after 120 s" in out
    for line in ("Arm M2: melo serve -X memory.retriever=embedding (+ embed-model)", "Arm B0: melo serve -X memory.retriever=bm25",
                 "Arm N0: melo serve -X memory.across_chats=false", "Arm FC: melo serve -X memory.across_chats=false [config.toml: agent.window_messages=256]"):
        assert line in out
    assert "  FC: 66 turns (64 in long chats, 0 teaching, 2 questions); kinds longrange" in out and "  B0: 8 turns (0 in long chats, 4 teaching, 4 questions); kinds followup" in out
    assert "  M2: 74 turns" in out and "222 turns in all" in out and "Preflight: ok" in out and not (tmp_path / "run").exists()
    assert run.main([*argv, "--arms", "FC", "--long-timeout", "30", "--repeats", "2"]) == 0
    assert "2 cells = 1 arms x 2 repeats" in capsys.readouterr().out
    assert run.parse_args(["--run-dir", "x", "--localmelo", "y", "--base-url", "z"]).long_timeout == 120.0


def test_an_arms_file_with_variants_of_m2_works_on_a_v2_task_file(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    arms = tmp_path / "arms.json"
    base = ["memory.retriever=embedding"]
    arms.write_text(json.dumps({
        "M2-off": {"x": [*base, "memory.context_query=off"], "embedding": True, "base": "M2"},
        "M2-on": {"x": [*base, "memory.context_query=on"], "embedding": True, "base": "M2"},
        "N0": {"x": list(run.MEMORY_OFF_X)}, "FC": {"x": list(run.MEMORY_OFF_X), "config": {"agent.window_messages": 256}, "v1": False}}))
    argv = ["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--localmelo-sha", git(repo, "rev-parse", "HEAD")[:8],
            "--base-url", models.base_url, "--model", "chat-model", "--embedding-model", "embed-model", "--dry-run", "--gpu-lock", str(tmp_path / "none"),
            "--tasks", str(data_v2.DATA_FILE), "--questions", "L01,P01", "--arms-file", str(arms), "--arms", "M2-off,M2-on,N0,FC"]
    assert run.main(argv) == 0
    out = capsys.readouterr().out
    assert "Arm M2-on: melo serve -X memory.retriever=embedding -X memory.context_query=on (+ embed-model)" in out and "Preflight: ok" in out
    assert "Arm FC: melo serve -X memory.across_chats=false [config.toml: agent.window_messages=256]" in out
    assert run.load_arms(arms)["FC"].config == (("agent.window_messages", 256),) and run.load_arms(arms)["N0"].config == ()
    assert "  M2-off: 37 turns" in out and "  M2-on: 37 turns" in out and "  FC: 33 turns" in out  # L01 (33 turns) and P01 (4): the variants follow M2's row
    with pytest.raises(SystemExit, match="unknown arm M2"):
        run.main([*argv, "--arms", "M2"])  # the arms file replaces the built-in table


def test_a_v2_study_has_its_own_protocol_and_a_v1_study_keeps_its_identity(tmp_path, repo, models):
    v1 = make_study(tmp_path / "a", repo, models).identity()
    assert v1["protocol"] == "across-run-v2" and "long_timeout_s" not in v1 and v1["questions"][0] == "Q2"
    study = v2_study(tmp_path / "b", repo, models, long_timeout_s=90.0)
    v2 = study.identity()
    assert v2["protocol"] == "across-run-v3" and v2["long_timeout_s"] == 90.0 and v2["tasks_sha256"] == study.tasks.sha256()
    assert v2["arms"]["FC"]["config"] == {"agent.window_messages": 256} and "config" not in v2["arms"]["N0"]  # the config record shows the setting
    assert v2["questions"] == [i.id for i in study.items()] and set(v1) < set(v2)
    assert {k: v for k, v in v2.items() if k not in ("protocol", "tasks_sha256", "questions", "long_timeout_s", "arms")} == {
        k: v for k, v in v1.items() if k not in ("protocol", "tasks_sha256", "questions", "arms")}
    assert run.run_study(study) == 0
    changed = v2_study(tmp_path / "b", repo, models, long_timeout_s=30.0)
    assert run.run_study(changed) == 2  # the study's config changed: a fresh directory is needed


def test_tasks_json_lists_the_items_and_the_kinds_arms(ran):
    doc = json.loads((ran.study.run_dir / "tasks.json").read_text())
    assert doc["protocol"] == "mini-v2" and doc["kinds"]["longrange"] == {"N0": "out", "M2": "out", "FC": "in"}
    assert [i["id"] for i in doc["items"]] == [i.id for i in ran.study.items()] and doc["items"][0]["kind"] == "longrange"


def test_a_v1_study_still_runs_the_old_flow(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, arm_names=("N0",))
    assert not study.v2 and run.run_study(study) == 0
    assert [r["phase"] for r in turns(study, "r1__N0")] == ["teach"] * 4 + ["ask"] * 5
    info = json.loads((cell_dir(study, "r1__N0") / "cell.json").read_text())
    assert set(info["counts"]) == {"teach", "ask"} and set(info["rss"]) == {"before", "after_warmup", "after_teach", "after"}
    assert data.load().sha256() == "8cd7464b028ab3d76887d2cbc3ae70bff8e2a44cf8048dc66c4c4f61d557a77b"


def test_the_stores_of_a_finished_cell_can_be_read_after_it_ends(ran):
    n0, fc = (cell_dir(ran.study, c) / "home" for c in ("r1__N0", "r1__FC"))
    chat = next(r["session_id"] for r in turns(ran.study, "r1__N0") if r["item"] == "L1.q1")
    facts = stores.window_facts(n0, chat, "Today's call is in room 4B.")
    assert not facts.fact_in_window and facts.cap == 64 and facts.messages_before > 64 and stores.judge(facts, "out") == (True, "")
    chat = next(r["session_id"] for r in turns(ran.study, "r1__FC") if r["item"] == "L1.q1")
    facts = stores.window_facts(fc, chat, "Today's call is in room 4B.")
    assert facts.fact_in_window and facts.cap == 256 and stores.judge(facts, "in") == (True, "")
