"""The public-benchmark driver against a fake `melo memory ingest`, the FakeGateway and a fake judge: flow, resume, controls."""
import json
import tempfile
import tomllib
from pathlib import Path

import pytest
from memory.across import client, server
from memory.public import judge, run
from memory.tests.across_fake import FakeModelServer
from memory.tests.public_support import (
    PublicLauncher,
    judge_server,
    make_public_study,
    make_repo,
    mini_questions,
    write_dataset,
)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def models():
    server_ = judge_server()
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


def results(study):
    return run.read_rows(study.run_dir / "results.jsonl")


def by_question(study):
    return {r["question_id"]: r for r in results(study)}


# --- arms, prompts, the environment of a gateway ---


def test_launch_env_is_merged_and_cannot_undo_the_isolation(tmp_path):
    base = {"PATH": "/usr/bin", "HOME": "/real", "EXAMPLE_API_KEY": "k"}
    assert "LOCALMELO_NOW" not in server.serve_env(tmp_path, base) and server.serve_env(tmp_path, base) == server.serve_env(tmp_path, base, None)
    env = server.serve_env(tmp_path, base, {"LOCALMELO_NOW": "2023-05-30T23:40:00", "HOME": "/elsewhere"})
    assert env["LOCALMELO_NOW"] == "2023-05-30T23:40:00" and env["HOME"] == str(tmp_path) and "EXAMPLE_API_KEY" not in env
    spec = server.LaunchSpec(Path("/x/melo"), tmp_path, 1, (), tmp_path / "log")
    assert spec.env is None and server.LaunchSpec(Path("/x/melo"), tmp_path, 1, (), tmp_path / "log", {"A": "1"}).env == {"A": "1"}


def test_load_arms(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({
        "base": {"label": "production", "x": ["memory.retriever=embedding"]},
        "wide": {"x": [], "config": {"memory.recall_records": 8}, "extra_body": {"temperature": 0.6}, "embedding": False}}))
    arms = run.load_arms(path)
    assert arms["base"].extra_body == {} and arms["base"].embedding and arms["base"].x == ("memory.retriever=embedding",)
    assert arms["wide"].config == {"memory.recall_records": 8} and arms["wide"].extra_body == {"temperature": 0.6} and not arms["wide"].embedding
    assert arms["wide"].identity() == {"x": [], "config": {"memory.recall_records": 8}, "extra_body": {"temperature": 0.6}, "embedding": False}
    for bad in ({"a": {"xx": []}}, {"a": {"x": ["no-equals"]}}, {"a": {"config": []}}, {}, [], {"a": 3}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            run.load_arms(path)


def test_control_prompts(tmp_path):
    assert run.load_control_prompts("default") == run.CONTROL_PROMPTS and len(run.CONTROL_PROMPTS) == 20
    assert len(set(run.CONTROL_PROMPTS)) == 20 and all(p.strip() for p in run.CONTROL_PROMPTS)
    (tmp_path / "p.json").write_text(json.dumps(["What is 2+2?", " Name a colour. "]))
    assert run.load_control_prompts(str(tmp_path / "p.json")) == ("What is 2+2?", "Name a colour.")
    (tmp_path / "p.txt").write_text("# comment\nWhat is 2+2?\n\nName a colour.\n")
    assert run.load_control_prompts(str(tmp_path / "p.txt")) == ("What is 2+2?", "Name a colour.")
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(ValueError):
        run.load_control_prompts(str(tmp_path / "empty.txt"))


def test_ingest_report_is_read_from_the_last_json_line():
    assert run.parse_report('{"sessions": 2, "ids": {}}\n') == {"sessions": 2, "ids": {}}
    assert run.parse_report('warming up\n{"sessions": 1}\n')["sessions"] == 1
    with pytest.raises(run.IngestError):
        run.parse_report("Ingested 2 sessions.")


# --- hits ---


def test_auto_hit_maps_store_ids_to_file_ids():
    q1, q2 = mini_questions()
    ids = {"sess_a": "st1", "sess_b": "st2"}
    memory = {"used": [{"kind": "chat", "id": 3, "session_id": "st2", "turn": 1}, {"kind": "record", "id": 1, "session_id": "st1", "turn": 1},
                       {"kind": "chat", "id": 9, "session_id": "own-chat", "turn": 2}], "recall": {"search_hint": "try search_memory"}}
    row = run.retrieval(q1, memory, [], ids)
    assert row["auto_hit"] is True and row["auto_cover"] == 1.0 and row["answer_sessions"] == ["sess_a"]
    assert [u["file_id"] for u in row["used"]] == ["sess_b", "sess_a", None]  # the third is not a chat of the ingest file
    assert row["search_hint"] == "try search_memory" and row["searches"] == 0 and row["search_hit"] is None and row["search_via"] is None
    assert run.retrieval(q1, {"used": memory["used"][:1], "search_hint": "top level"}, [], ids)["auto_hit"] is False
    assert run.retrieval(q1, {"used": memory["used"][:1], "search_hint": "top level"}, [], ids)["search_hint"] == "top level"
    assert run.retrieval(q1, None, [], ids)["auto_hit"] is False  # memory off: nothing was used
    assert run.retrieval(q2, memory, [], ids)["auto_hit"] is False
    multi = {**q1, "answer_session_ids": ["sess_a", "sess_b", "sess_z"]}
    assert run.retrieval(multi, memory, [], ids)["auto_cover"] == 0.6667
    none = {**q1, "answer_session_ids": []}
    assert run.retrieval(none, memory, [], ids)["auto_hit"] is None and run.retrieval(none, memory, [], ids)["auto_cover"] is None


def search_events(*outputs, **extra):
    events = []
    for output in outputs:
        events += [{"type": "tool.started", "data": {"name": "search_memory", "arguments": {"query": "degree"}}},
                   {"type": "tool.finished", "data": {"name": "search_memory", "status": "success", "output": output, **extra}},
                   {"type": "tool.started", "data": {"name": "file_read"}}]
    return events


def test_search_hit_by_words_when_the_event_names_no_session():
    q = mini_questions()[0]
    ids = {"sess_a": "st1", "sess_b": "st2"}
    chat = "1. User: I hold a degree in Business Administration. melo: Congratulations on the degree! (a chat, May 20)"
    note = "1. the user said “I hold a degree in Business Administration.” (a chat, May 20)"
    other = "1. User: My favourite hobby is bouldering at the gym. melo: That sounds like fun. (a chat, May 21)"
    for text in (chat, note):
        row = run.retrieval(q, None, search_events(text), ids)
        assert (row["searches"], row["search_hit"], row["search_via"], row["auto_hit"]) == (1, True, "text", False)
    assert run.retrieval(q, None, search_events(other), ids)["search_hit"] is False
    assert run.retrieval(q, None, search_events("Nothing in melo's memory matches that."), ids)["search_hit"] is False
    twice = run.retrieval(q, None, search_events(other, chat), ids)
    assert twice["searches"] == 2 and twice["search_hit"] is True
    unfinished = [{"type": "tool.started", "data": {"name": "search_memory"}}]
    assert run.retrieval(q, None, unfinished, ids)["search_hit"] is False
    assert run.retrieval(q, None, [{"type": "tool.started", "data": {"name": "file_read"}}], ids)["searches"] == 0


def test_search_hit_from_the_sessions_an_event_names():
    q = mini_questions()[0]
    ids = {"sess_a": "st1", "sess_b": "st2"}
    named = run.retrieval(q, None, search_events("", items=[{"session_id": "st1", "text": "x"}]), ids)
    assert (named["search_hit"], named["search_via"]) == (True, "items")
    wrong = run.retrieval(q, None, search_events("", items=[{"session_id": "st2"}]), ids)
    assert (wrong["search_hit"], wrong["search_via"]) == (False, "items")
    as_json = run.retrieval(q, None, search_events(json.dumps({"results": [{"session_id": "st1"}]})), ids)
    assert (as_json["search_hit"], as_json["search_via"]) == (True, "items")


def test_prompt_tokens_come_from_the_first_model_call():
    res = client.TurnResult("s", "t", 1, "completed", usage={"prompt_tokens": 900.0, "completion_tokens": 30.0})
    assert run.first_prompt_tokens(res) == 900
    res.events = [{"type": "assistant.message", "data": {"usage": {"prompt_tokens": 412, "completion_tokens": 9}}},
                  {"type": "assistant.message", "data": {"usage": {"prompt_tokens": 488}}}]
    assert run.first_prompt_tokens(res) == 412
    assert run.first_prompt_tokens(client.TurnResult("s", "t", 1, "completed")) is None


def test_proxy_view_reads_the_agent_request():
    log = [
        {"path": "/v1/embeddings", "t_start": 100.0, "t_first_byte": 100.1},
        {"path": "/v1/chat/completions", "tools": None, "t_start": 100.2, "t_first_byte": 100.3, "req_bytes": 90, "stream": False},
        {"path": "/v1/chat/completions", "tools": True, "t_start": 100.5, "t_first_byte": 101.0, "req_bytes": 4000, "stream": True},
        {"path": "/v1/chat/completions", "tools": True, "t_start": 500.0, "t_first_byte": 501.0, "req_bytes": 1, "stream": True},
    ]
    view = run.proxy_view(log, 99.0, 102.0)
    assert view == {"ttft_ms": 500.0, "req_bytes": 4000, "streamed": True, "chat_requests": 2, "embed_requests": 1}
    assert run.proxy_view([], 1.0, 2.0)["ttft_ms"] is None


# --- a run ---


def test_a_run_ingests_once_per_question_and_asks_in_a_fresh_gateway(tmp_path, repo, models, real_home):
    launcher = PublicLauncher()
    study = make_public_study(tmp_path, repo, models, launcher, keep_homes=True)
    assert run.run_study(study) == 0
    state = json.loads((study.run_dir / "public.json").read_text())
    assert state["complete"] and state["config"]["protocol"] == "public-lme-v1"
    assert [q["question_id"] for q in json.loads((study.run_dir / "questions.json").read_text())] == ["q1aa", "q2bb_abs"]
    assert "Business" not in (study.run_dir / "questions.json").read_text()  # the run's own files hold no text of the dataset

    # ingest: the file, the arguments, the template HOME
    ingest = json.loads((study.run_dir / "questions" / "q1aa" / "ingest.json").read_text())
    assert (ingest["sessions"], ingest["turns"], ingest["docs"], ingest["notes"], ingest["skipped"]) == (3, 3, 3, 3, 1)
    assert ingest["ids"] == {"sess_a": "st001", "sess_b": "st002", "sess_c": "st003"} and ingest["error"] == "" and ingest["model"] == "embed-model"
    template = tmp_path / "work" / "q1aa" / "template"
    argv = json.loads((template / "ingest-argv.json").read_text())
    assert argv[1:] == ["-j", "-X", "notes=rules"] and argv[0].endswith("ingest_input.json")
    assert json.loads(Path(argv[0]).read_text())["sessions"][0]["date"] == "2023-05-20T02:21:00"
    config = tomllib.loads((template / ".cache" / "localmelo" / "config.toml").read_text())
    assert config["ui"]["mode"] == "developer" and config["memory"]["retriever"] == "embedding"
    assert config["vllm"]["chat_url"] == models.base_url and config["vllm"]["embedding_model"] == "embed-model"  # ingest talks to the server itself
    assert not (template / ".cache" / "localmelo" / "gateway" / "auth").exists()

    # the cell: a copy of the template, pointed at a timing proxy, with the day of the question
    assert [s.env for s in launcher.specs] == [{"LOCALMELO_NOW": "2023-05-30T23:40:00"}, {"LOCALMELO_NOW": "2023-06-02T08:05:00"}]
    assert launcher.cells == ["A__r1", "A__r1"] and all(s.x_pairs == () for s in launcher.specs)
    cell_home = tmp_path / "work" / "q1aa" / "A__r1" / "home"
    cell_config = tomllib.loads((cell_home / ".cache" / "localmelo" / "config.toml").read_text())
    assert cell_config["vllm"]["chat_url"] != models.base_url and cell_config["vllm"]["chat_url"].startswith("http://127.0.0.1:")
    assert "extra_body" not in cell_config["vllm"]  # production decoding: no sampling field
    assert cell_config["ui"]["mode"] == "developer" and (cell_home / "ingest-argv.json").exists()
    assert (study.run_dir / "cells" / "q1aa__A__r1" / "turn.json").exists()

    # the rows
    rows = by_question(study)
    q1, q2 = rows["q1aa"], rows["q2bb_abs"]
    assert (q1["state"], q1["answer"], q1["correct"]) == ("finished", "From my notes: I hold a degree in Business Administration.", True)
    assert q1["judge"]["label"] is True and q1["judge"]["retry"] is False and q1["judge"]["model"] == "chat-model"
    assert (q1["judge_retry"], q1["judge_unparsed"]) == (0, 0)
    assert q1["retrieval"]["auto_hit"] is True and q1["retrieval"]["used"][0]["file_id"] == "sess_a" and q1["retrieval"]["searches"] == 0
    assert (q1["question_type"], q1["abstention"], q1["arm"], q1["repeat"], q1["cold_start"]) == ("single-session-user", False, "A", 1, {"skipped": True})
    assert (q2["question_type"], q2["abstention"], q2["answer"], q2["correct"]) == ("temporal-reasoning", True, "I don't know.", True)
    assert q2["retrieval"]["auto_hit"] is False and q2["retrieval"]["used"] == []
    assert q1["wall_ms"] > 0 and q1["recall"]["fallback"] == "none" and q1["tools"] == []
    # the judge saw the question's own prompt, and the benchmark's request
    prompts = [b["messages"][0]["content"] for b in models.bodies if "Model Response:" in str(b.get("messages"))]
    assert len(prompts) == 2 and "unanswerable question" in prompts[1] and "Correct Answer: Business Administration" in prompts[0]
    # the ask was the question's own words, in a chat of its own
    gateway = launcher.gateways[0]
    assert gateway.requests.count(("POST", "/api/sessions")) == 1 and len(gateway.sessions) == 1  # one new chat, no other
    assert next(iter(gateway.sessions.values()))["turns"][0]["user_text"] == "Which degree do I hold?"
    # the environment: once
    env = json.loads((study.run_dir / "environment.json").read_text())
    assert env["start"]["sw_vers"] == "ProductName: test" and "end" not in env
    assert real_home.exists() and list(real_home.iterdir()) == []


def test_a_real_gateway_process_gets_the_day_of_the_question(tmp_path, repo, models, monkeypatch):
    """server.launch starts the fake `melo serve` for real: LOCALMELO_NOW reaches it, the isolation holds, it is stopped at the end."""
    monkeypatch.setenv("ACROSS_TEST_SECRET_KEY", "hunter2")
    study = make_public_study(tmp_path, repo, models, server.launch, questions=mini_questions()[1:], keep_homes=True)
    assert run.run_study(study) == 0
    seen = json.loads((tmp_path / "work" / "q2bb_abs" / "A__r1" / "home" / "env.json").read_text())
    assert seen["now"] == "2023-06-02T08:05:00" and seen["secret_visible"] is False and seen["stdin_tty"] is False
    assert seen["home"] == str(tmp_path / "work" / "q2bb_abs" / "A__r1" / "home") and seen["browser"] == "/usr/bin/true"
    assert seen["argv"][0] == "serve" and seen["argv"][3:] == []
    assert results(study)[0]["answer"] == "I don't know." and results(study)[0]["correct"] is True


def test_homes_and_ingest_files_are_removed_unless_kept(tmp_path, repo, models):
    study = make_public_study(tmp_path, repo, models)
    assert run.run_study(study) == 0
    assert not (tmp_path / "work" / "q1aa").exists() and not (tmp_path / "work" / "q2bb_abs").exists()
    assert not list(tmp_path.rglob("ingest_input.json"))
    assert not list(study.run_dir.rglob("token"))  # a token stays out of the run records
    assert len(results(study)) == 2


def test_environment_snapshot_names_the_builds(tmp_path, repo, models):
    study = make_public_study(tmp_path, repo, models, data_path=write_dataset(tmp_path / "mini.json"))
    snap = run.system_snapshot(study)
    assert {"sw_vers", "xcodebuild_version", "localmelo_sha", "bench_sha", "bench_dirty", "data", "model_server"} <= set(snap)
    assert snap["localmelo_sha"] == server.git_sha(repo) and len(snap["bench_sha"]) in (0, 40)
    assert snap["data"] == {"name": "mini.json", "sha256": "0" * 64, "questions": 2}
    assert "chat-model" in snap["model_server"]["listed"]


def test_stop_then_resume_runs_only_what_is_missing(tmp_path, repo, models, monkeypatch):
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 2 else None  # asked once per question, before its ingest

    launcher = PublicLauncher()
    study = make_public_study(tmp_path, repo, models, launcher, cell_guard=guard, turn_guard=lambda: None)
    assert run.run_study(study) == 3
    assert [r["question_id"] for r in results(study)] == ["q1aa"] and len(launcher.specs) == 1
    assert "stopping" in (study.run_dir / "public.log").read_text()
    assert not json.loads((study.run_dir / "public.json").read_text())["complete"]
    first = results(study)[0]

    ingested = []
    real_ingest = run.ingest
    monkeypatch.setattr(run, "ingest", lambda s, q, d: ingested.append(q["question_id"]) or real_ingest(s, q, d))
    again = PublicLauncher()
    study2 = make_public_study(tmp_path, repo, models, again)
    assert run.run_study(study2) == 0
    assert ingested == ["q2bb_abs"] and [s.env["LOCALMELO_NOW"] for s in again.specs] == ["2023-06-02T08:05:00"]
    assert [r["question_id"] for r in results(study2)] == ["q1aa", "q2bb_abs"] and results(study2)[0] == first  # the first row was not touched
    assert json.loads((study.run_dir / "public.json").read_text())["complete"]

    ingested.clear()
    nothing = PublicLauncher()
    assert run.run_study(make_public_study(tmp_path, repo, models, nothing)) == 0
    assert ingested == [] and nothing.specs == [] and len(results(study2)) == 2


def test_a_stop_between_two_arms_of_a_question_resumes_at_the_missing_cell(tmp_path, repo, models, monkeypatch):
    arms = {"A": run.Arm("A", "a"), "B": run.Arm("B", "b", x=("memory.retriever=bm25",))}
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 2 else None  # before the second cell of question 1

    launcher = PublicLauncher()
    study = make_public_study(tmp_path, repo, models, launcher, arms=arms, questions=mini_questions()[:1], turn_guard=guard)
    assert run.run_study(study) == 3
    assert [(r["arm"], r["repeat"]) for r in results(study)] == [("A", 1)]
    ingested = []
    real_ingest = run.ingest
    monkeypatch.setattr(run, "ingest", lambda s, q, d: ingested.append(q["question_id"]) or real_ingest(s, q, d))
    again = PublicLauncher()
    assert run.run_study(make_public_study(tmp_path, repo, models, again, arms=arms, questions=mini_questions()[:1])) == 0
    assert ingested == ["q1aa"] and again.cells == ["B__r1"]  # the history is loaded again, and only the missing cell is asked
    assert sorted((r["arm"], r["repeat"]) for r in results(study)) == [("A", 1), ("B", 1)]


def test_a_run_with_other_settings_needs_a_fresh_directory(tmp_path, repo, models, capsys):
    assert run.run_study(make_public_study(tmp_path, repo, models)) == 0
    other = make_public_study(tmp_path, repo, models, repeats=2)
    assert run.run_study(other) == 2 and "use a fresh directory" in capsys.readouterr().err
    stray = tmp_path / "other-run"
    stray.mkdir()
    (stray / "notes.txt").write_text("x")
    assert run.run_study(make_public_study(tmp_path, repo, models, run_dir=stray)) == 2


def test_arms_repeats_and_the_settings_of_a_cell(tmp_path, repo, models):
    arms = {
        "A": run.Arm("A", "production decoding"),
        "B": run.Arm("B", "words only, short window", x=("memory.retriever=bm25",), config={"agent.window_messages": 32},
                     extra_body={"temperature": 0.3}),
    }
    launcher = PublicLauncher()
    study = make_public_study(tmp_path, repo, models, launcher, arms=arms, repeats=2, keep_homes=True, questions=mini_questions()[:1])
    assert run.run_study(study) == 0
    rows = results(study)
    assert sorted((r["arm"], r["repeat"]) for r in rows) == [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    # the arms are rotated across repeats and questions
    assert launcher.cells == ["A__r1", "B__r1", "B__r2", "A__r2"]
    config_b = tomllib.loads((tmp_path / "work" / "q1aa" / "B__r1" / "home" / ".cache" / "localmelo" / "config.toml").read_text())
    assert config_b["vllm"]["extra_body"] == {"temperature": 0.3} and config_b["agent"]["window_messages"] == 32
    assert config_b["memory"]["retriever"] == "embedding"  # the shared setting stays unless the arm overrides it
    b_spec = next(s for s in launcher.specs if "bm25" in " ".join(s.x_pairs))
    assert b_spec.x_pairs == ("memory.retriever=bm25",)
    identity = json.loads((study.run_dir / "public.json").read_text())["config"]
    assert identity["arms"]["B"]["extra_body"] == {"temperature": 0.3} and identity["arms"]["A"]["extra_body"] == {}


def test_the_embedding_model_is_unloaded_before_each_cell(tmp_path, repo, models):
    study = make_public_study(tmp_path, repo, models, cold_start=True)
    assert run.run_study(study) == 0
    first, second = results(study)
    assert first["cold_start"]["was_loaded"] is True and first["cold_start"]["loaded_after"] is False
    assert second["cold_start"]["was_loaded"] is False  # still out of memory: nothing asked for it
    assert models.unloads == ["embed-model"]


def test_a_failed_turn_is_a_wrong_answer_and_is_not_judged(tmp_path, repo, models):
    launcher = PublicLauncher(fail_texts=("degree",))
    study = make_public_study(tmp_path, repo, models, launcher)
    assert run.run_study(study) == 0
    q1 = by_question(study)["q1aa"]
    assert (q1["state"], q1["correct"], q1["judge"]) == ("failed", False, None) and q1["error"] == "boom"
    assert by_question(study)["q2bb_abs"]["correct"] is True


def test_a_judge_reply_that_is_neither_yes_nor_no_is_counted_in_the_row(tmp_path, repo):
    replies = {"n": 0}

    def unclear_first(body):  # the first ask of every question is unclear; the second is "no"
        replies["n"] += 1
        return "hmm" if replies["n"] % 2 else "No"

    models = judge_server(unclear_first)
    try:
        study = make_public_study(tmp_path, repo, models, questions=mini_questions()[:1])
        assert run.run_study(study) == 0
        row = results(study)[0]
        assert (row["judge_retry"], row["judge_unparsed"], row["correct"]) == (1, 0, False) and row["judge"]["retry_reply"] == "No"
    finally:
        models.stop()


def test_a_judge_that_is_down_is_asked_again_on_the_next_run(tmp_path, repo, models):
    down = judge.Verdict(label=None, error="URLError: refused", model="chat-model")
    launcher = PublicLauncher()
    study = make_public_study(tmp_path, repo, models, launcher, verdict=lambda q, a: down)
    assert run.run_study(study) == 0
    assert [(r["correct"], r["judge"]["error"]) for r in results(study)] == [(None, "URLError: refused")] * 2
    again = PublicLauncher()
    study2 = make_public_study(tmp_path, repo, models, again)
    assert run.run_study(study2) == 0
    assert again.specs == [] and [r["correct"] for r in results(study2)] == [True, True]  # judged again, nothing asked again
    assert all(not r["judge"]["error"] for r in results(study2))


def test_a_gateway_that_dies_stops_the_run_and_a_bad_ingest_is_named(tmp_path, repo, models, monkeypatch, capsys):
    def refuse(spec):
        raise server.LaunchError("melo serve exited with 2")

    study = make_public_study(tmp_path, repo, models, refuse)
    assert run.run_study(study) == 1 and not results(study) and "melo serve exited" in capsys.readouterr().err
    assert not (tmp_path / "work" / "q1aa").exists()
    monkeypatch.setattr(run, "BASE_SETTINGS", {"memory.retriever": "embedding"})  # no developer mode: ingest exits 2
    bare = make_public_study(tmp_path / "b", repo, models)
    with pytest.raises(run.IngestError, match="exited 2"):
        run.ingest(bare, mini_questions()[0], tmp_path / "b" / "q")


# --- the control ---


def test_control_prompts_are_asked_on_the_first_homes_and_false_recall_is_counted(tmp_path, repo, models):
    arms = {"A": run.Arm("A", "recall on"), "B": run.Arm("B", "memory off", x=("memory.across_chats=false",))}
    launcher = PublicLauncher(model_calls=True)
    study = make_public_study(
        tmp_path, repo, models, launcher, arms=arms, control_prompts=("Hold degree?", "Name a colour."), control_homes=1, repeats=1)
    assert run.run_study(study) == 0
    controls = {(c["question_id"], c["arm"]): c for c in run.read_rows(study.run_dir / "controls.jsonl")}
    assert sorted(controls) == [("q1aa", "A"), ("q1aa", "B")]  # only the first HOME, once per arm
    a, b = controls[("q1aa", "A")], controls[("q1aa", "B")]
    assert a["prompt"] == b["prompt"] == "Hold degree?" and a["index"] == 0
    assert a["false_recall"] is True and a["used"][0]["file_id"] == "sess_a" and b["false_recall"] is False and b["used"] == []
    assert a["state"] == "finished" and isinstance(a["ttft_ms"], float) and a["ttft_ms"] >= 0 and a["chat_requests"] >= 1
    assert a["req_bytes"] > 0 and a["streamed"] is True and len(a["answer"]) <= 300
    # a control cell asks the control prompt in the question's own HOME copy, with its day
    asked = [next(iter(g.sessions.values()))["turns"][0]["user_text"] for g in launcher.gateways]
    assert [asked.count(t) for t in ("Which degree do I hold?", "Hold degree?", "Which instrument do I play?")] == [2, 2, 2]
    assert {s.env["LOCALMELO_NOW"] for s in launcher.specs[:6]} <= {"2023-05-30T23:40:00", "2023-06-02T08:05:00"}
    assert len(results(study)) == 4 and json.loads((study.run_dir / "public.json").read_text())["complete"]

    again = PublicLauncher()
    assert run.run_study(make_public_study(
        tmp_path, repo, models, again, arms=arms, control_prompts=("Hold degree?", "Name a colour."), control_homes=1)) == 0
    assert again.specs == []  # the controls resume like the questions


# --- the command line ---


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the preflight's throwaway HOME
    monkeypatch.delenv("BENCH_UPSTREAM_TOKEN_CMD", raising=False)
    monkeypatch.delenv("LONGMEMEVAL_DATA", raising=False)
    dataset = write_dataset(tmp_path / "mini.json")
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps({"A": {"label": "embedding"}, "B": {"x": ["memory.retriever=bm25"], "config": {"agent.window_messages": 32}}}))
    argv = ["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--localmelo-sha", server.git_sha(repo)[:8], "--base-url", models.base_url,
            "--model", "chat-model", "--embedding-model", "embed-model", "--data", str(dataset), "--arms-file", str(arms), "--dry-run",
            "--gpu-lock", str(tmp_path / "no-lock"), "--questions", "q1aa,q2bb_abs", "--repeats", "2", "--control-prompts", "default",
            "--control-homes", "1"]
    assert run.main(argv) == 0
    out = capsys.readouterr().out
    assert "2 questions (abstention 1, single-session-user 1) x 2 arms x 2 repeats = 8 question cells, 2 control cells, 2 ingests" in out
    assert "Arm B: melo serve -X memory.retriever=bm25 [config.toml: agent.window_messages=32]" in out and "Preflight: ok" in out
    assert "the GPU lock is absent" in out and not (tmp_path / "run").exists()
    assert run.main([*argv, "--arms", "A"]) == 0
    with pytest.raises(SystemExit):
        run.main([*argv, "--arms", "Z9"])
    with pytest.raises(SystemExit):
        run.main([*argv, "--questions", "nope"])


def test_preflight_names_what_is_wrong(tmp_path, repo, models):
    ok = make_public_study(tmp_path, repo, models)
    assert run.preflight(ok) == []
    bad_arm = make_public_study(tmp_path, repo, models, arms={"X": run.Arm("X", "x", x=("memory.nope=1",), config={"memory.recall_records": 8})})
    assert run.preflight(bad_arm) == ["this localmelo build does not know the setting memory.nope",
                                      "this localmelo build does not know the setting memory.recall_records"]
    assert any("does not exist" in p for p in run.preflight(make_public_study(tmp_path, tmp_path / "nowhere", models, expect_sha="abc")))
    assert any("judge model 'other'" in p for p in run.preflight(make_public_study(tmp_path, repo, models, judge_model="other")))
    assert any("not in the file" in p for p in run.preflight(make_public_study(tmp_path, repo, models, ids=("q1aa", "nope"))))
    assert any("not deadbeef" in p for p in run.preflight(make_public_study(tmp_path, repo, models, expect_sha="deadbeef")))
    bare = FakeModelServer(["chat-model", "embed-model"], status_routes=False)
    try:
        cold = make_public_study(tmp_path, repo, bare)
        cold.cold_start = True
        assert any("--admin-token-cmd" in p for p in run.preflight(cold))
        cold.cold_start = False
        assert run.preflight(cold) == []
    finally:
        bare.stop()
