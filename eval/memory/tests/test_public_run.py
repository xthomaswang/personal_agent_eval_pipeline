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


def test_ingest_passes_the_runs_extra_pairs_and_only_then_names_them(tmp_path, monkeypatch):
    seen = []

    class Done:
        returncode, stderr = 0, ""
        stdout = json.dumps({"sessions": 1, "ids": {}})

    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or Done())
    monkeypatch.setattr(run, "template_home", lambda study, qdir, url=None: tmp_path / "home")
    study = run.Study(run_dir=tmp_path, questions=mini_questions(), ids=("q1aa",), arms={}, arm_names=(), repeats=1,
                      localmelo=tmp_path, expect_sha="", base_url="", model="", embedding_model="", judge_model="",
                      ingest_x=("ingest=compress", "memory.compress_cache=/c.jsonl"))
    (tmp_path / "q").mkdir()
    run.ingest(study, mini_questions()[0], tmp_path / "q")
    assert seen[0][-6:] == ["-X", "notes=rules", "-X", "ingest=compress", "-X", "memory.compress_cache=/c.jsonl"]
    assert study.identity()["ingest"] == {"notes": "rules", "x": ["ingest=compress", "memory.compress_cache=/c.jsonl"]}
    plain = run.Study(**{**study.__dict__, "ingest_x": ()})
    assert plain.identity()["ingest"] == {"notes": "rules"}  # the identity of a run without them is as before


def test_encode_share_ingests_the_oldest_sessions_with_the_pairs_and_the_rest_plain(tmp_path, monkeypatch):
    seen = []

    class Done:
        returncode, stderr = 0, ""

    def fake_run(cmd, **kw):
        sessions = json.loads(Path(cmd[3]).read_text())["sessions"]
        seen.append((cmd, [x["id"] for x in sessions]))
        done = Done()
        compress = "ingest=compress" in cmd
        done.stdout = json.dumps({"sessions": len(sessions), "turns": 2 * len(sessions), "ids": {x["id"]: "st-" + x["id"] for x in sessions},
                                  "error": "", **({"ingest": "compress", "model_calls": 3, "compress": {"w": 1}} if compress else {})})
        return done

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(run, "template_home", lambda study, qdir, url=None: tmp_path / "home")
    study = run.Study(run_dir=tmp_path, questions=mini_questions(), ids=("q1aa", "q2bb_abs"), arms={}, arm_names=(), repeats=1,
                      localmelo=tmp_path, expect_sha="", base_url="", model="", embedding_model="", judge_model="",
                      ingest_x=("ingest=compress", "memory.x=1"), encode_share=0.5)
    (tmp_path / "q").mkdir()
    report = run.ingest(study, mini_questions()[0], tmp_path / "q")
    assert [ids for _, ids in seen] == [["sess_a", "sess_b"], ["sess_c"]]  # ceil(0.5 x 3) = 2, oldest first
    assert seen[0][0][-4:] == ["-X", "ingest=compress", "-X", "memory.x=1"] and seen[1][0][-2:] == ["-X", "memory.x=1"]
    assert report["sessions"] == 3 and report["turns"] == 6 and report["model_calls"] == 3 and report["ingest"] == "compress"
    assert report["ids"] == {"sess_a": "st-sess_a", "sess_b": "st-sess_b", "sess_c": "st-sess_c"}
    assert (report["encoded_sessions"], report["plain_sessions"], report["error"]) == (2, 1, "")
    assert not list((tmp_path / "q").glob("ingest_*.json"))  # both ingest files are removed
    seen.clear()
    run.ingest(study, mini_questions()[1], tmp_path / "q")
    assert [ids for _, ids in seen] == [["sess_y"], ["sess_x"]]  # by date, not the file's order
    assert study.identity()["ingest"]["encode_share"] == 0.5
    assert "encode_share" not in run.Study(**{**study.__dict__, "encode_share": None}).identity()["ingest"]
    assert run.encode_share(None, ()) is None and run.encode_share(0.5, ["ingest=compress"]) == 0.5
    for share, pairs in ((0.5, []), (1.0, ["ingest=compress"]), (0.0, ["ingest=compress"])):
        with pytest.raises(SystemExit):
            run.encode_share(share, pairs)


def test_ingest_settings_go_into_the_ingest_home_only(tmp_path):
    assert run.parse_settings(["memory.compress_cache=/r/c.json", "memory.compress_at_chars=8000", "a.b=true"]) == {
        "memory.compress_cache": "/r/c.json", "memory.compress_at_chars": 8000, "a.b": True}
    with pytest.raises(SystemExit):
        run.parse_settings(["memory.compress_cache"])
    study = run.Study(run_dir=tmp_path, questions=mini_questions(), ids=("q1aa",), arms={}, arm_names=(), repeats=1,
                      localmelo=tmp_path, expect_sha="", base_url="http://127.0.0.1:1/v1", model="m", embedding_model="e",
                      judge_model="", ingest_settings={"memory.compress_cache": "/r/c.json"},
                      ingest_extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    home = run.template_home(study, tmp_path / "q", "http://127.0.0.1:2/v1")
    config = tomllib.loads((home / ".cache" / "localmelo" / "config.toml").read_text())
    assert config["memory"]["compress_cache"] == "/r/c.json" and config["memory"]["retriever"] == "embedding"
    assert config["vllm"]["chat_url"] == "http://127.0.0.1:2/v1"  # the ingest's proxy
    assert config["vllm"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert study.identity()["ingest"] == {"notes": "rules", "settings": {"memory.compress_cache": "/r/c.json"},
                                          "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert run.Study(**{**study.__dict__, "ingest_settings": {}, "ingest_extra_body": {}}).identity()["ingest"] == {"notes": "rules"}


def test_an_encoder_at_its_token_cap_or_failing_windows_stops_the_run():
    window = {"exchanges": 3, "chars": 9000, "cached": False, "failed": False, "cut": False, "dropped": 0}
    compress = {"s1": {"windows": [window, {**window, "failed": True}], "gate": {"forced": [4]}, "items": [{}, {}]},
                "s2": {"windows": [{**window, "cached": True}] * 8, "gate": {"forced": []}, "items": [{}]}}
    summary = run.compress_summary(compress)
    assert summary == {"windows": 10, "failed": 1, "cached": 8, "cut": 0, "items": 3, "forced": 1} and run.compress_summary(None) is None
    rows = [{"path": "/v1/chat/completions", "finish_reason": "stop", "completion_tokens": 900},
            {"path": "/v1/chat/completions", "finish_reason": "length", "completion_tokens": 2000},
            {"path": "/v1/embeddings"}, {"path": "/v1/chat/completions"}]
    assert run.ingest_requests(rows) == {"chat": 3, "length": 1, "finish_unknown": 1, "completion_tokens_max": 2000}
    assert run.ingest_problem({"compress_summary": summary, "requests": {"length": 0}}) == ""  # 10% is allowed
    assert "token cap" in run.ingest_problem({"compress_summary": summary, "requests": {"length": 1}})
    many = {**summary, "failed": 2}
    assert run.ingest_problem({"compress_summary": many, "requests": {"length": 0}}) == "2 of 10 encoder windows failed (more than 10%)"
    assert run.ingest_problem({"requests": {"chat": 0, "length": 0}}) == ""  # a turns ingest has no encoder


def test_a_splitting_encoder_counts_only_failed_windows():
    window = {"exchanges": 3, "chars": 9000, "cached": False, "failed": False, "cut": False, "dropped": 0,
              "split": 0, "failed_parts": 0, "part": None}
    compress = {"s1": {"windows": [window, {**window, "split": 2, "failed_parts": 1}], "gate": {"forced": []}, "items": [{}]}}
    summary = run.compress_summary(compress)
    assert summary["split"] == 2 and summary["failed_parts"] == 1 and summary["failed"] == 0
    assert run.ingest_problem({"compress_summary": summary, "requests": {"length": 2}}) == ""  # the capped replies were split
    assert "failed" in run.ingest_problem({"compress_summary": {**summary, "failed": 1}, "requests": {"length": 0}})


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


def test_the_copy_of_a_repeated_answer_session_is_a_hit():
    q = mini_questions()[0]
    q["haystack_session_ids"] = ["sess_a", "sess_b", "sess_a"]
    ids = {"sess_a": "st1", "sess_b": "st2", "sess_a#2": "st3"}
    row = run.retrieval(q, {"used": [{"kind": "chat", "session_id": "st3", "text": "x"}]}, [], ids)
    assert row["auto_hit"] is True and row["used"][0]["file_id"] == "sess_a"


def test_evidence_seen_and_all_are_the_shares_of_the_evidence_turns_words_the_model_was_shown():
    q = mini_questions()[0]
    q["haystack_sessions"][0][0]["has_answer"] = True  # "I hold a degree in Business Administration."
    other = {"used": [{"kind": "chat", "session_id": "st1", "text": "melo: Congratulations on the degree!"}]}
    assert run.evidence_seen(q, other, []) == {"evidence_seen": 0.25, "evidence_all": 0.25}  # only "degree" of four words
    shown = {"used": [{"kind": "record", "text": "The user holds a degree in Business Administration."}]}
    assert run.evidence_seen(q, shown, [])["evidence_seen"] == 0.75  # "hold" is not "holds"
    assert run.evidence_seen(q, None, search_events("1. User: I hold a degree in Business Administration."))["evidence_all"] == 1.0
    q["haystack_sessions"][0][1]["has_answer"] = True  # a second evidence turn, "Congratulations on the degree!", half shown
    assert run.evidence_seen(q, shown, []) == {"evidence_seen": 0.75, "evidence_all": 0.5}
    assert run.evidence_seen(mini_questions()[0], shown, []) == {"evidence_seen": None, "evidence_all": None}
    assert run.retrieval(q, shown, [], {"sess_a": "st1"})["evidence_all"] == 0.5
    cut = {"used": [{"kind": "chat", "text": "User: I hold a degree in Business Administration.", "shown": "the user said “I hold a …”"}]}
    assert run.evidence_seen(q, cut, [])["evidence_seen"] == 0.25  # the line the model read counts, not the stored head


def test_opened_turns_are_counted_and_their_text_counts_as_shown():
    q = mini_questions()[0]
    q["haystack_sessions"][0][0]["has_answer"] = True  # "I hold a degree in Business Administration."
    ids = {"sess_a": "st1", "sess_b": "st2"}
    listed = {"used": [{"kind": "chat", "session_id": "st1", "text": "“Which degree…” · cosine 0.61"}],
              "opened": [{"session_id": "st2", "turn": 1, "id": 7}, {"session_id": "st1", "turn": 1, "id": 3}]}
    opened = [{"type": "tool.started", "data": {"name": "open_memory", "arguments": {"ids": ["t7", "t3"]}}},
              {"type": "tool.finished", "data": {"name": "open_memory", "output": "User: I hold a degree in Business Administration."}}]
    row = run.retrieval(q, listed, opened, ids)
    assert (row["opens"], row["open_hit"], row["evidence_all"]) == (2, True, 1.0)
    assert run.retrieval(q, {"used": [], "opened": [{"session_id": "st2"}]}, [], ids)["open_hit"] is False
    assert (run.retrieval(q, {"used": []}, [], ids)["opens"], run.retrieval(q, {"used": []}, [], ids)["open_hit"]) == (0, None)


def test_a_first_turn_digest_is_counted_by_the_chats_it_names():
    q = mini_questions()[0]
    ids = {"sess_a": "st1", "sess_b": "st2"}
    lines = [{"session_id": "st2", "turn": 3, "line": "a chat · 4 min ago: …"}, {"session_id": "st1", "turn": 1, "line": "…"}]
    row = run.retrieval(q, {"used": [], "digest": {"minutes": 30, "items": lines}}, [], ids)
    assert (row["digest_items"], row["digest_hit"]) == (2, True)
    assert run.retrieval(q, {"used": [], "digest": {"minutes": 30, "items": lines[:1]}}, [], ids)["digest_hit"] is False
    assert {k: run.retrieval(q, {"used": []}, [], ids)[k] for k in ("digest_items", "digest_hit")} == {"digest_items": 0, "digest_hit": None}


def compress_report():
    """A compress ingest's -j section for mini_questions()[0]: sess_a's exchange 0 holds the evidence."""
    return {"model_calls": 3, "cache_hits": 0,
            "sess_a": {"gate": {"kept": [0], "ignored": [], "unmentioned": [], "forced": []},
                       "items": [{"id": 11, "title": "Degree", "text": "You hold a degree in Business Administration.", "kind": "profile",
                                  "sources": [["sess_a", 0]]}],
                       "routes": [{"item": 11, "action": "NEW", "target": None, "reason": "", "title": "Degree"}]},
            "sess_b": {"gate": {"kept": [], "ignored": [0], "unmentioned": [], "forced": []}, "items": [],
                       "routes": [{"item": None, "action": "IGNORE", "target": None, "reason": "small talk", "title": "Hobby"}]}}


def test_compress_evidence_follows_the_evidence_through_the_gate_and_the_items():
    q = mini_questions()[0]
    q["haystack_sessions"][0][0]["has_answer"] = True
    q["answer"] = "Business Administration"
    found = run.compress_evidence(q, compress_report())
    assert found == {"evidence": 1, "kept": 1, "gated_out": False, "items": [11], "answer_in_item": True, "routes": ["NEW"]}
    lossy = compress_report()
    lossy["sess_a"]["items"][0]["text"] = "You studied at a university."
    assert run.compress_evidence(q, lossy)["answer_in_item"] is False
    ignored = compress_report()
    ignored["sess_a"]["gate"] = {"kept": [], "ignored": [0], "unmentioned": [], "forced": []}
    ignored["sess_a"]["items"] = []
    assert run.compress_evidence(q, ignored)["gated_out"] is True
    assert run.compress_evidence(q, None) is None


def test_item_hits_and_opens_name_the_chats_of_their_sources():
    q = mini_questions()[0]
    ids = {"sess_a": "st1", "sess_b": "st2"}
    memory = {"used": [{"kind": "item", "item": 11, "text": "Degree", "sources": [["st1", 1]]}],
              "opened": [{"item": 11, "sources": [["st1", 1]]}]}
    ret = run.retrieval(q, memory, [], ids)
    assert ret["used"][0]["file_ids"] == ["sess_a"] and ret["auto_hit"] is True
    assert (ret["opens"], ret["open_hit"], ret["opened_items"]) == (1, True, [11])


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


def test_search_hit_from_the_memory_events_searched_items_wins_over_the_tool_text():
    q = mini_questions()[0]
    ids = {"sess_a": "st1", "sess_b": "st2"}
    other = "1. User: My favourite hobby is bouldering at the gym. melo: That sounds like fun. (a chat, May 21)"
    memory = {"used": [], "searched": [{"session_id": "st1", "turn": 2, "kind": "chat", "id": 7}]}
    hit = run.retrieval(q, memory, search_events(other), ids)
    assert (hit["search_hit"], hit["search_via"]) == (True, "memory")
    miss = run.retrieval(q, {"used": [], "searched": [{"session_id": "st2", "turn": 1}]}, search_events(other), ids)
    assert (miss["search_hit"], miss["search_via"]) == (False, "memory")
    assert run.retrieval(q, {"used": [], "searched": []}, [], ids)["search_hit"] is False  # a search that found nothing

def test_a_build_that_sends_searched_is_never_matched_by_words(tmp_path):
    q = mini_questions()[0]
    ids = {"sess_a": "st1", "sess_b": "st2"}
    words = search_events("1. User: I hold a degree in Business Administration. melo: Congratulations on the degree! (a chat, May 20)")
    assert run.retrieval(q, {"used": []}, words, ids)["search_via"] == "text"  # an older build
    exact = run.retrieval(q, {"used": []}, words, ids, exact=True)  # searched left out: the searches found nothing
    assert (exact["search_hit"], exact["search_via"]) == (False, "memory")
    found = run.retrieval(q, {"used": [], "searched": [{"session_id": "st1"}]}, words, ids, exact=True)
    assert (found["search_hit"], found["search_via"]) == (True, "memory")
    assert run.retrieval(q, {"used": []}, [], ids, exact=True)["search_hit"] is None  # no search call
    doc = tmp_path / "docs" / "reference" / "events.md"
    assert run.sends_searched(tmp_path) is False
    doc.parent.mkdir(parents=True)
    doc.write_text("| `turn.memory` | ... `searched` (`[{kind, id, text, source, session_id, turn, at}]`, ...) |\n")
    assert run.sends_searched(tmp_path) is True


def test_prompt_tokens_come_from_the_first_model_call():
    res = client.TurnResult("s", "t", 1, "completed", usage={"prompt_tokens": 900.0, "completion_tokens": 30.0})
    assert run.first_prompt_tokens(res) == 900
    res.events = [{"type": "assistant.message", "data": {"usage": {"prompt_tokens": 412, "completion_tokens": 9}}},
                  {"type": "assistant.message", "data": {"usage": {"prompt_tokens": 488}}}]
    assert run.first_prompt_tokens(res) == 412
    assert run.first_prompt_tokens(client.TurnResult("s", "t", 1, "completed")) is None


def test_reply_notices_count_the_replies_asked_again_and_a_turn_that_ended_on_a_text_tool_call():
    def notice(kind):
        return {"type": "notice", "data": {"kind": kind, "message": "m", "step": 1, "attempt": 1, "detail": {}}}
    reply = {"type": "assistant.message", "data": {}}
    events = [reply, notice("tool_call_as_text"), reply, notice("empty_reply"), notice("retry"), reply, notice("tool_call_as_text")]
    assert run.reply_notices(events) == {
        "notices": {"tool_call_as_text": 2, "empty_reply": 1, "retry": 1}, "reply_retries": 2, "final_text_tool_call": True}
    assert run.reply_notices([reply, notice("empty_reply"), reply]) == {
        "notices": {"empty_reply": 1}, "reply_retries": 1, "final_text_tool_call": False}
    assert run.reply_notices([reply]) == {"notices": {}, "reply_retries": 0, "final_text_tool_call": False}
    assert run.turn_fields(None, "boom")["notices"] == {}


def test_proxy_view_reads_the_agent_request():
    log = [
        {"path": "/v1/embeddings", "t_start": 100.0, "t_first_byte": 100.1},
        {"path": "/v1/chat/completions", "tools": None, "t_start": 100.2, "t_first_byte": 100.3, "req_bytes": 90, "stream": False},
        {"path": "/v1/chat/completions", "tools": True, "t_start": 100.5, "t_first_byte": 101.0, "req_bytes": 4000, "stream": True},
        {"path": "/v1/chat/completions", "tools": True, "t_start": 500.0, "t_first_byte": 501.0, "req_bytes": 1, "stream": True},
    ]
    view = run.proxy_view(log, 99.0, 102.0)
    assert view == {"ttft_ms": 500.0, "req_bytes": 4000, "streamed": True, "chat_requests": 2, "embed_requests": 1,
                    "prompt_tokens_max": None, "cached_share": None, "reasoning_tokens": None}  # an old log: no usage numbers
    assert run.proxy_view([], 1.0, 2.0)["ttft_ms"] is None
    log[1].update(prompt_tokens=400, cached_tokens=None, reasoning_tokens=5)
    log[2].update(prompt_tokens=1600, cached_tokens=1500, reasoning_tokens=None)
    log[3].update(prompt_tokens=9000, cached_tokens=9000, reasoning_tokens=70)  # after the turn: not counted
    view = run.proxy_view(log, 99.0, 102.0)
    assert (view["prompt_tokens_max"], view["cached_share"], view["reasoning_tokens"]) == (1600, 0.75, 5)


def test_repeat_tools_counts_the_same_call_again():
    def started(name, **arguments):
        return {"type": "tool.started", "data": {"name": name, "arguments": arguments}}

    events = [started("search_memory", query="rent"), {"type": "tool.finished", "data": {"name": "search_memory"}},
              started("open_memory", id=3), started("search_memory", query="rent"), started("search_memory", query="rent 2024"),
              started("open_memory", id=3)]
    assert run.repeat_tools(events) == 2
    assert run.repeat_tools([]) == 0


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
    monkeypatch.setattr(run, "gateway_problem", lambda melo, home: None)  # the fake checkout has no python of its own
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


def test_a_fixed_judge_on_its_own_server_is_checked_there_and_recorded(tmp_path, repo, models):
    """The judge can be a model the chat server does not have: it is looked for on its own server, and the run's identity names that server."""
    judge_server = FakeModelServer(["judge-model"])
    try:
        study = make_public_study(tmp_path, repo, models, judge_model="judge-model", judge_base_url=judge_server.base_url)
        assert run.preflight(study) == []
        assert study.identity()["judge_base_url"] == judge_server.base_url
        missing = make_public_study(tmp_path, repo, models, judge_model="other", judge_base_url=judge_server.base_url)
        assert run.preflight(missing) == [f"judge model 'other' is not listed by {judge_server.base_url}/models"]
    finally:
        judge_server.stop()
    gone = make_public_study(tmp_path, repo, models, judge_model="judge-model", judge_base_url=judge_server.base_url)
    assert any("/models is not readable" in p for p in run.preflight(gone))
    same = make_public_study(tmp_path, repo, models)  # the judge on the chat server: the identity is as before
    assert "judge_base_url" not in same.identity()
    assert "judge_base_url" not in make_public_study(tmp_path, repo, models, judge_base_url=models.base_url).identity()


def test_preflight_names_a_checkout_that_cannot_start_its_gateway(tmp_path, repo, models):
    broken = make_public_study(tmp_path, repo, models, serve_check=lambda melo, home: f"{melo.name} cannot start melo serve")
    assert run.preflight(broken) == ["melo cannot start melo serve"]
    python = repo / ".venv" / "bin" / "python"
    python.write_text("#!/bin/sh\necho 'ImportError: Gateway requires fastapi and uvicorn' >&2\nexit 1\n")
    python.chmod(0o755)
    problem = run.gateway_problem(repo / ".venv" / "bin" / "melo", tmp_path)
    assert "cannot start melo serve (ImportError: Gateway requires fastapi and uvicorn)" in problem and "--all-extras" in problem
    python.write_text("#!/bin/sh\nexit 0\n")
    assert run.gateway_problem(repo / ".venv" / "bin" / "melo", tmp_path) is None
    python.unlink()
    assert "cannot be run" in run.gateway_problem(repo / ".venv" / "bin" / "melo", tmp_path)


def test_optional_settings_only_on_builds_that_know_them(tmp_path):
    knows = tmp_path / "melo-new"
    knows.write_text("#!/bin/sh\n[ \"$1 $2 $3\" = \"config get approvals.auto_asks\" ] && echo true && exit 0\nexit 2\n")
    knows.chmod(0o755)
    older = tmp_path / "melo-old"
    older.write_text("#!/bin/sh\nexit 2\n")
    older.chmod(0o755)
    from memory.across import server
    assert dict(server.optional_settings(knows)) == {"approvals.auto_asks": False}
    assert server.optional_settings(older) == ()
    assert server.optional_settings(tmp_path / "missing") == ()
