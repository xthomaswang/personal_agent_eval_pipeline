"""Teaching once and asking from a copy (the teach_only and teach_from arm options), the query adapter built between the
teaching and the questions (the adapter option), and the checks and the schedule that go with them, on the fake gateway."""
import hashlib
import json
import tempfile

import pytest
from memory.across import data_v2, report, run
from memory.tests.across_fake import FakeModelServer
from memory.tests.across_support import Launcher, git, make_repo, make_study, v2_launcher, v2_study
from memory.tests.test_across_sleep import sleep_raw

DEV = {"ui.mode": "developer"}
BM25 = {"x": ["memory.retriever=bm25"], "embedding": False}
EMBED = {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2"}


@pytest.fixture(autouse=True)
def real_home(tmp_path, monkeypatch):
    home = tmp_path / "realhome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    yield home
    assert list(home.iterdir()) == []


@pytest.fixture
def models():
    server_ = FakeModelServer(["chat-model", "embed-model"])
    yield server_
    server_.stop()


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


def write_arms(tmp_path, rows):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(rows))
    return path


def teach_arms(tmp_path, base=BM25):
    """T teaches; A asks from a copy of it; B does too and builds the query adapter first."""
    return run.load_arms(write_arms(tmp_path, {
        "T": {**base, "teach_only": True},
        "A": {**base, "teach_from": "T"},
        "B": {**base, "teach_from": "T", "config": DEV, "adapter": True},
    }))


def journal(study, cell):
    return run.Journal(study.run_dir / "cells" / cell / "turns.jsonl").read()


def turns(study, cell):
    return [r for r in journal(study, cell) if r["event"] == "turn"]


def cell_info(study, cell):
    return json.loads((study.run_dir / "cells" / cell / "cell.json").read_text())


def posts(gw, *paths):
    """The POSTs a gateway got to these paths (chats are created and the night is run by POST), in order."""
    return [p for m, p in gw.requests if m == "POST" and p in paths]


class WorkWriter(Launcher):
    """Leaves a file in the cell's work folder when the gateway starts, unless it is there already (a copy brought it)."""

    def __call__(self, spec):
        made = spec.home.parent / "work" / "made.txt"
        if not made.exists():
            made.write_text(spec.home.parent.name)
        return super().__call__(spec)


# --- arms file, identity, checks ---


def test_load_arms_reads_the_teaching_options(tmp_path):
    arms = teach_arms(tmp_path)
    assert (arms["T"].teach_only, arms["T"].teach_from, arms["T"].adapter) == (True, "", False)
    assert (arms["A"].teach_only, arms["A"].teach_from, arms["A"].adapter) == (False, "T", False)
    assert (arms["B"].teach_only, arms["B"].teach_from, arms["B"].adapter, arms["B"].settings()) == (False, "T", True, DEV)
    assert [arms[a].phase for a in "TAB"] == ["teach", "ask", "ask"] and run.ARMS["M2"].phase == "all"
    assert arms["T"].identity()["teach_only"] is True and "teach_from" not in arms["T"].identity()
    assert arms["A"].identity()["teach_from"] == "T" and "adapter" not in arms["A"].identity()
    assert arms["B"].identity()["adapter"] is True and arms["B"].identity()["teach_from"] == "T"
    assert not {"teach_only", "teach_from", "adapter"} & set(run.ARMS["M2"].identity())
    assert run.teach_sources(arms, ("A", "B", "T")) == {"A": "T", "B": "T"} and run.teach_sources(run.ARMS, ("M2", "N0")) == {}


def test_the_options_change_the_identity_of_the_study(tmp_path, repo, models):
    plain = run.load_arms(write_arms(tmp_path, {"T": {**BM25, "teach_only": True}, "A": BM25}))
    copied = run.load_arms(write_arms(tmp_path, {"T": {**BM25, "teach_only": True}, "A": {**BM25, "teach_from": "T"}}))
    one = make_study(tmp_path / "a", repo, models, arms=plain, arm_names=("T", "A")).identity()
    two = make_study(tmp_path / "b", repo, models, arms=copied, arm_names=("T", "A")).identity()
    assert one != two and two["arms"]["A"]["teach_from"] == "T" and "teach_from" not in one["arms"]["A"]


@pytest.mark.parametrize("rows, problem", [
    ({"A": {**BM25, "teach_from": "T"}}, "not an arm of this arms file"),
    ({"T": BM25, "A": {**BM25, "teach_from": "T"}}, "does not have teach_only"),
    ({"T": {**BM25, "teach_only": True}, "A": {**BM25, "teach_only": True, "teach_from": "T"}}, "cannot be combined"),
    ({"T": {**BM25, "teach_only": True, "teach_from": "T"}}, "cannot be combined"),
    ({"T": {**BM25, "teach_only": True, "sleep": "rules"}}, "would never run"),
    ({"T": {**BM25, "teach_only": True, "adapter": True}}, "would never run"),
])
def test_bad_teaching_options_are_refused_when_the_arms_load(tmp_path, rows, problem):
    with pytest.raises(ValueError, match=problem):
        run.load_arms(write_arms(tmp_path, rows))


def test_the_source_arm_must_be_in_the_run(tmp_path, repo, models):
    study = make_study(tmp_path, repo, models, Launcher(), arms=teach_arms(tmp_path), arm_names=("A",))
    with pytest.raises(ValueError, match="not among this run's arms"):
        run.run_study(study)
    assert not study.run_dir.exists()  # refused before anything was written


def test_an_adapter_arm_is_refused_without_keep_ask_chats(tmp_path, repo, models):
    launcher = Launcher()
    study = make_study(tmp_path, repo, models, launcher, arms=teach_arms(tmp_path), arm_names=("T", "B"))
    with pytest.raises(ValueError, match="adapter needs --keep-ask-chats"):
        run.run_study(study)
    assert launcher.gateways == [] and not study.run_dir.exists()
    study.keep_ask_chats = True
    run.check_study(study)  # fine with it


def test_a_teaching_arm_cannot_run_long_range_items(tmp_path, repo, models):
    arms = run.load_arms(write_arms(tmp_path, {"T": {**EMBED, "teach_only": True}, "A": {**EMBED, "teach_from": "T"}}))
    study = v2_study(tmp_path, repo, models, arms=arms, arm_names=("T", "A"))  # M2's row lists the long-range kind
    with pytest.raises(ValueError, match="arm T: teach_only cannot run long-range items"):
        run.run_study(study)
    assert not study.run_dir.exists()
    mixed = run.load_arms(write_arms(tmp_path, {"T": {**EMBED, "base": "B0", "teach_only": True}, "A": {**EMBED, "teach_from": "T"}}))
    with pytest.raises(ValueError, match="arm A: teach_from cannot run long-range items"):
        run.check_study(v2_study(tmp_path, repo, models, arms=mixed, arm_names=("T", "A")))
    run.check_study(v2_study(tmp_path, repo, models, arms=arms, arm_names=("T", "A"), question_ids=("P1", "P2")))  # followups only: fine


def test_the_command_line_refuses_the_same_and_the_dry_run_plans_the_copies(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    arms = write_arms(tmp_path, {
        "T": {**BM25, "teach_only": True}, "A": {**BM25, "teach_from": "T"}, "B": {**BM25, "teach_from": "T", "config": DEV, "adapter": True}})
    argv = ["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--localmelo-sha", git(repo, "rev-parse", "HEAD")[:8],
            "--base-url", models.base_url, "--model", "chat-model", "--embedding-model", "embed-model", "--dry-run",
            "--gpu-lock", str(tmp_path / "no-lock"), "--repeats", "1", "--arms-file", str(arms), "--arms", "A,B,T"]
    with pytest.raises(SystemExit, match="adapter needs --keep-ask-chats"):
        run.main(argv)
    assert run.main([*argv, "--keep-ask-chats"]) == 0
    out = capsys.readouterr().out
    assert out.index("r1__T") < out.index("r1__A") < out.index("r1__B")  # the source runs first although it was named last
    assert "[teaching only]" in out and "[teaching and HOME copied from T]" in out and "[query adapter before the questions]" in out
    assert "163 turns in all" in out  # T teaches the 43 facts of the shipped file, A and B ask its 60 questions each
    assert not (tmp_path / "run").exists()
    bad = write_arms(tmp_path, {"A": {**BM25, "teach_from": "T"}})
    with pytest.raises(SystemExit, match="not an arm of this arms file"):
        run.main([*argv[:-4], "--arms-file", str(bad), "--arms", "A"])


# --- schedule ---


def test_the_source_arm_runs_first_in_every_repeat():
    copiers = {"A": "T", "B": "T"}
    cells = run.schedule(("A", "B", "T"), 3, copiers)
    assert [c["cell_id"] for c in cells] == ["r1__T", "r1__A", "r1__B", "r2__T", "r2__B", "r2__A", "r3__T", "r3__A", "r3__B"]
    assert [c["position"] for c in cells[:3]] == [1, 2, 3] and [c["index"] for c in cells] == list(range(1, 10))
    assert [c["arm"] for c in cells if c["position"] == 1] == ["T", "T", "T"]


def test_the_rest_of_the_rotation_is_kept():
    cells = run.schedule(("T", "A", "Z"), 3, {"A": "T"})
    by_repeat = [[c["arm"] for c in cells if c["repeat"] == r] for r in (1, 2, 3)]
    assert by_repeat == [["T", "A", "Z"], ["T", "A", "Z"], ["Z", "T", "A"]]  # T joins A at the front; Z keeps its turn
    plain = run.schedule(("M2", "L1", "B0", "N0"), 3)
    assert run.schedule(("M2", "L1", "B0", "N0"), 3, {}) == plain == run.schedule(("M2", "L1", "B0", "N0"), 3, None)
    assert run.sources_first(["B", "A", "T"], {"A": "T"}) == ["B", "T", "A"]
    assert run.sources_first(["A"], {"A": "T"}) == ["A"]  # a source outside the run is not invented here (check_study refuses it)


# --- teach_only and teach_from, v1 ---


def test_a_teach_only_cell_sends_only_the_teaching_and_finishes_with_a_home_that_can_be_copied(tmp_path, repo, models):
    launcher = Launcher(store=True, delay=0.002)
    study = make_study(tmp_path, repo, models, launcher, arms=teach_arms(tmp_path), arm_names=("T",))
    assert run.run_study(study) == 0
    assert [(r["phase"], r["item"]) for r in turns(study, "r1__T")] == [("teach", f"F{n}") for n in (1, 2, 3, 4)]
    gw = launcher.gateways[0]
    assert posts(gw, "/api/sessions") == ["/api/sessions"] * 4 and not any(m == "DELETE" for m, _ in gw.requests)
    assert posts(gw, "/api/memory/sleep", "/api/memory/adapter") == []
    info = cell_info(study, "r1__T")
    assert info["state"] == "finished" and info["gateway_exit"] == 0 and info["counts"]["teach"]["turns"] == 4
    assert info["counts"]["ask"]["turns"] == 0 and "rss" in info and "after_teach" in info["rss"]
    home = study.run_dir / "cells" / "r1__T" / "home"
    assert not (home / ".cache" / "localmelo" / "gateway" / "auth").exists()  # the token is gone: safe to copy
    assert (home / ".cache" / "localmelo" / "memory" / "records.db").exists()
    assert json.loads(gw.memory_path.read_text())["records"] and "teach_from" not in info


def test_a_teach_from_cell_starts_from_the_copy_and_sends_only_the_questions(tmp_path, repo, models):
    launcher = WorkWriter(store=True, delay=0.002)
    study = make_study(tmp_path, repo, models, launcher, arms=teach_arms(tmp_path), arm_names=("A", "B", "T"), keep_ask_chats=True)
    assert run.run_study(study) == 0
    assert launcher.cells == ["r1__T", "r1__A", "r1__B"]
    gws = dict(zip(launcher.cells, launcher.gateways, strict=True))
    src = study.run_dir / "cells" / "r1__T"
    sha = hashlib.sha256((src / "home" / ".cache" / "localmelo" / "memory" / "records.db").read_bytes()).hexdigest()
    taught = [r for r in journal(study, "r1__T") if r.get("phase") == "teach"]
    assert len(taught) == 8 and all("copied_from" not in r for r in taught)  # a start and a turn for each of 4 facts
    for cell in ("r1__A", "r1__B"):
        cdir = study.run_dir / "cells" / cell
        rows = journal(study, cell)
        copied = [r for r in rows if r.get("copied_from")]
        assert copied == [{**r, "copied_from": "r1__T"} for r in taught]  # the same rows, in order, marked
        assert [(r["phase"], r["item"]) for r in rows if r["event"] == "turn" and not r.get("copied_from")] == [("ask", q.id) for q in study.asked()]
        assert not [r for r in rows if r["event"] == "start" and r["phase"] == "teach" and not r.get("copied_from")]
        assert posts(gws[cell], "/api/sessions").count("/api/sessions") == 5  # the five questions, no teaching
        info = cell_info(study, cell)
        assert info["teach_from"] == {"cell": "r1__T", "records_sha256": sha, "turns": 4} and info["state"] == "finished"
        assert info["counts"]["teach"]["turns"] == 4 and info["counts"]["ask"]["turns"] == 5
        assert (cdir / "work" / "made.txt").read_text() == "r1__T"  # the work folder came along
        records = json.loads(gws[cell].memory_path.read_text())["records"]
        assert len(records) == 3 and [r["text"] for r in records] == [r["text"] for r in json.loads((src / "home" / ".cache" / "localmelo" / "memory" / "memory.json").read_text())["records"]]
        asked = {r["item"]: r for r in rows if r["event"] == "turn" and r["phase"] == "ask"}
        assert asked["Q1"]["answer"].startswith("From my notes") and asked["Q4"]["answer"] == "From my notes: I live in Utrecht."
        assert not (cdir / "home" / ".cache" / "localmelo" / "gateway" / "auth").exists()
    assert (study.run_dir / "cells" / "r1__A" / "home" / ".cache" / "localmelo" / "memory" / "records.db").read_bytes() == (src / "home" / ".cache" / "localmelo" / "memory" / "records.db").read_bytes()
    assert (src / "work" / "made.txt").read_text() == "r1__T"
    rep = report.analyse(study.run_dir, study.tasks)  # the report still finds the teaching turns of a copying cell
    cells = {c["id"]: c for c in report.load_run(study.run_dir)["cells"]}
    assert sum(t["phase"] == "teach" for t in cells["r1__A"]["turns"]) == 4 and {r["arm"] for r in rep["rows"]} == {"A", "B"}


def test_a_copying_cell_that_stops_and_resumes_copies_nothing_again(tmp_path, repo, models):
    calls = []

    def guard():
        calls.append(1)
        return "STOP file found" if len(calls) == 7 else None  # T's 4 teaching turns, then 2 questions of B

    launcher = Launcher(store=True, delay=0.002)
    study = make_study(tmp_path, repo, models, launcher, arms=teach_arms(tmp_path), arm_names=("T", "B"), keep_ask_chats=True, turn_guard=guard)
    assert run.run_study(study) == 3
    assert len(turns(study, "r1__B")) == 4 + 2
    first = cell_info(study, "r1__B")
    assert first["state"] == "paused" and "adapter" in first and posts(launcher.gateways[-1], "/api/memory/adapter") == ["/api/memory/adapter"]
    # a file the stopped cell made after the copy must survive the resume: nothing is copied over it again
    mark = study.run_dir / "cells" / "r1__B" / "home" / "mark.txt"
    mark.write_text("kept")
    study.turn_guard = lambda: None
    assert run.run_study(study) == 0
    assert mark.read_text() == "kept" and launcher.cells == ["r1__T", "r1__B", "r1__B"]
    rows = journal(study, "r1__B")
    assert len([r for r in rows if r.get("copied_from")]) == 8 and len({r["key"] for r in rows if r["event"] == "turn"}) == 9
    info = cell_info(study, "r1__B")
    assert info["resumed"] == 1 and info["teach_from"] == first["teach_from"] and info["adapter"] == first["adapter"]
    assert sum(len(g.adapter_bodies) for g in launcher.gateways) == 1  # built once, not again after the resume


def test_a_teach_from_cell_needs_a_finished_source(tmp_path, repo, models, capsys):
    arms = teach_arms(tmp_path)
    launcher = Launcher(store=True, delay=0.002)
    cell = {"index": 2, "repeat": 1, "arm": "A", "position": 2, "cell_id": "r1__A"}
    direct = make_study(tmp_path / "direct", repo, models, launcher, arms=arms, arm_names=("T", "A"))
    with pytest.raises(run.TeachSourceError, match="r1__T, which is missing"):
        run.run_cell(direct, cell)
    assert not (direct.run_dir / "cells" / "r1__A").exists() and launcher.gateways == []  # nothing was left behind
    study = make_study(tmp_path, repo, models, launcher, arms=arms, arm_names=("T", "A"))
    calls = []
    study.cell_guard = lambda: calls.append(1) or ("STOP file found" if len(calls) == 2 else None)  # T runs, A does not start
    assert run.run_study(study) == 3
    path = study.run_dir / "cells" / "r1__T" / "cell.json"
    path.write_text(json.dumps({**cell_info(study, "r1__T"), "state": "paused"}))  # as if the teaching had been cut short
    study.cell_guard = lambda: None
    assert run.run_study(study) == 2
    assert "r1__T, which is paused" in capsys.readouterr().err
    assert json.loads((study.run_dir / "across.json").read_text())["cells"]["r1__A"]["state"] == "error"
    assert not (study.run_dir / "cells" / "r1__A").exists()


# --- the query adapter ---


def test_the_adapter_is_built_once_between_the_teaching_and_the_questions(tmp_path, repo, models):
    arms = run.load_arms(write_arms(tmp_path, {"G": {**BM25, "config": DEV, "adapter": True}, "H": {**BM25, "config": DEV}}))
    launcher = Launcher(store=True, delay=0.002)
    study = make_study(tmp_path, repo, models, launcher, arms=arms, arm_names=("G", "H"), keep_ask_chats=True)
    assert run.run_study(study) == 0
    gws = dict(zip(launcher.cells, launcher.gateways, strict=True))
    order = posts(gws["r1__G"], "/api/sessions", "/api/memory/adapter")
    assert order.index("/api/memory/adapter") == 4 and len(order) == 4 + 1 + 5 and order.count("/api/memory/adapter") == 1
    assert gws["r1__G"].adapter_bodies == [{}] and gws["r1__H"].adapter_bodies == []
    cdir = study.run_dir / "cells" / "r1__G"
    doc = json.loads((cdir / "adapter.json").read_text())
    assert set(doc) == {"report", "wall_s", "error"} and doc["report"]["built"] is True and doc["error"] is None and doc["wall_s"] >= 0
    assert (cdir / "adapter.npz").read_bytes() == b"fake adapter 1"
    info = cell_info(study, "r1__G")
    assert info["adapter"]["npz"] is True and info["adapter"]["report"]["built"] is True and info["adapter"]["wall_s"] == doc["wall_s"]
    assert "error" not in info["adapter"] and "sleep" not in info
    assert not (study.run_dir / "cells" / "r1__H" / "adapter.json").exists() and "adapter" not in cell_info(study, "r1__H")
    assert not any(m == "DELETE" for m, _ in gws["r1__G"].requests)  # the question chats are kept: a delete would rebuild the adapter


def test_the_adapter_follows_the_sleep_and_a_failed_build_is_recorded_not_raised(tmp_path, repo, models):
    arms = run.load_arms(write_arms(tmp_path, {
        "S": {**BM25, "config": DEV, "sleep": "rules", "adapter": True}, "P": {**BM25, "sleep": "rules", "adapter": True}}))
    launcher = Launcher(store=True, delay=0.002)
    study = make_study(tmp_path, repo, models, launcher, arms=arms, arm_names=("S", "P"), keep_ask_chats=True)
    assert run.run_study(study) == 0
    gws = dict(zip(launcher.cells, launcher.gateways, strict=True))
    assert posts(gws["r1__S"], "/api/memory/sleep", "/api/memory/adapter") == ["/api/memory/sleep", "/api/memory/adapter"]
    info = cell_info(study, "r1__P")  # no developer mode: both routes answer 404
    assert info["sleep"]["error"].startswith("GatewayError") and info["adapter"]["error"].startswith("GatewayError") and info["adapter"]["npz"] is False
    doc = json.loads((study.run_dir / "cells" / "r1__P" / "adapter.json").read_text())
    assert doc["report"] is None and doc["error"].startswith("GatewayError") and not (study.run_dir / "cells" / "r1__P" / "adapter.npz").exists()
    assert len(turns(study, "r1__P")) == 9  # the questions were asked all the same


def test_an_adapter_build_waits_longer_than_the_client_s_ten_seconds(tmp_path):
    class SlowGateway:
        def __init__(self):
            self.calls = []

        def call(self, method, path, body=None, ok=(200,), timeout=None):
            self.calls.append((method, path, body, timeout))
            return {"built": True, "records": 3}

    gw, home = SlowGateway(), tmp_path / "home"
    home.mkdir()
    (home / run.ADAPTER_NPZ).parent.mkdir(parents=True)
    (home / run.ADAPTER_NPZ).write_bytes(b"npz")
    summary = run.build_adapter(gw, tmp_path, home)
    assert gw.calls == [("POST", "/api/memory/adapter", {}, run.SLEEP_TIMEOUT_S)] and run.SLEEP_TIMEOUT_S >= 600
    assert summary["npz"] is True and summary["report"] == {"built": True, "records": 3} and (tmp_path / "adapter.npz").read_bytes() == b"npz"


# --- v2 ---


def sleep_arms(tmp_path):
    return run.load_arms(write_arms(tmp_path, {
        "T": {**EMBED, "teach_only": True},
        "A": {**EMBED, "teach_from": "T"},
        "B": {**EMBED, "teach_from": "T", "config": DEV, "sleep": "rules", "adapter": True},
    }))


def test_the_sleep_kind_teaches_in_one_cell_and_asks_in_the_copies(tmp_path, repo, models):
    tasks = data_v2.parse(sleep_raw(), None)
    launcher = Launcher(store=True, delay=0.002)
    study = make_study(tmp_path, repo, models, launcher, tasks=tasks, arms=sleep_arms(tmp_path), arm_names=("B", "A", "T"), keep_ask_chats=True)
    assert run.run_study(study) == 0
    assert launcher.cells == ["r1__T", "r1__B", "r1__A"]  # rotated to B, A, T; T moves up to just before B
    gws = dict(zip(launcher.cells, launcher.gateways, strict=True))
    teach = [("teach", f"{i}.f{n}") for n in (1, 2) for i in ("D1", "C1")]  # day 1, round by round
    assert [(r["phase"], r["item"]) for r in turns(study, "r1__T")] == teach
    assert posts(gws["r1__T"], "/api/sessions", "/api/memory/sleep", "/api/memory/adapter") == ["/api/sessions"] * 4
    for cell in ("r1__A", "r1__B"):
        rows = turns(study, cell)
        assert [(r["phase"], r["item"]) for r in rows if r.get("copied_from")] == teach
        assert sorted(r["item"] for r in rows if not r.get("copied_from")) == ["A1.q1", "C1.q1", "D1.q1"]
        assert cell_info(study, cell)["teach_from"]["turns"] == 4
    assert posts(gws["r1__A"], "/api/sessions", "/api/memory/sleep", "/api/memory/adapter") == ["/api/sessions"] * 3
    assert posts(gws["r1__B"], "/api/sessions", "/api/memory/sleep", "/api/memory/adapter") == [
        "/api/memory/sleep", "/api/memory/adapter", "/api/sessions", "/api/sessions", "/api/sessions"]  # the night, then day 2
    assert gws["r1__B"].sleeps == ["rules"] and (study.run_dir / "cells" / "r1__B" / "adapter.npz").exists()
    assert "sleep" not in cell_info(study, "r1__A") and "adapter" not in cell_info(study, "r1__A")


def test_a_sleep_teach_only_cell_stops_after_day_one(tmp_path, repo, models):
    sent = []

    def send(phase, item, text, **kw):
        sent.append(item)
        return {"event": "turn", "phase": phase, "item": item}

    study = make_study(tmp_path, repo, models, tasks=data_v2.parse(sleep_raw(), None), arms=sleep_arms(tmp_path), arm_names=("T",))
    nights = []
    live = run.Live(study, study.arms["T"], {}, None, None, tmp_path, {"turns": {}, "deleted": set()}, send, tmp_path, lambda: nights.append(1), "teach")
    items = [study.tasks.item(i) for i in ("D1", "C1", "A1")]
    assert run.run_sleep(live, items) is False and sent == ["D1.f1", "C1.f1", "D1.f2", "C1.f2"] and nights == []
    sent.clear()
    live.phase = "ask"  # the copy: no day 1, then the night and the questions
    live.study = type("S", (), {"tasks": study.tasks, "turn_guard": staticmethod(lambda: None), "keep_ask_chats": True, "sleep": staticmethod(lambda s: None)})()
    assert run.run_sleep(live, items) is False and nights == [1] and [s for s in sent if s.endswith(".f1") or s.endswith(".f2")] == []
    assert sorted(sent) == ["A1.q1", "C1.q1", "D1.q1"]
    sent.clear()
    live.phase = "teach"
    assert run.run_recall(live, items) is False and sent == []


def test_a_followup_kind_teaches_in_one_cell_and_the_copy_builds_the_adapter_before_its_questions(tmp_path, repo, models):
    arms = run.load_arms(write_arms(tmp_path, {
        "T": {**EMBED, "teach_only": True}, "B": {**EMBED, "teach_from": "T", "config": DEV, "adapter": True}}))
    launcher = v2_launcher()
    study = v2_study(tmp_path, repo, models, launcher, arms=arms, arm_names=("T", "B"), question_ids=("P1", "P2"), keep_ask_chats=True)
    assert run.run_study(study) == 0
    gws = dict(zip(launcher.cells, launcher.gateways, strict=True))
    order = [i.id for i in study.items("T")]  # the task file's fixed shuffle
    assert [r["item"] for r in turns(study, "r1__T")] == [f"{i}.f{n}" for i in order for n in (1, 2)]
    assert posts(gws["r1__T"], "/api/sessions", "/api/memory/adapter") == ["/api/sessions"] * 4
    rows = turns(study, "r1__B")
    assert [r["item"] for r in rows if not r.get("copied_from")] == [f"{i}.q{n}" for i in order for n in (1, 2)]
    assert [r["item"] for r in rows if r.get("copied_from")] == [f"{i}.f{n}" for i in order for n in (1, 2)]
    assert posts(gws["r1__B"], "/api/sessions", "/api/memory/adapter") == ["/api/memory/adapter", "/api/sessions", "/api/sessions"]  # two ask chats
    assert gws["r1__B"].adapter_bodies == [{}] and rows[-1]["state"] == "finished"
    assert "Ortiz" in next(r for r in rows if r["item"] == "P1.q1")["answer"]  # B never taught it: the copied notes answer


def test_a_long_chat_refuses_a_teaching_phase(tmp_path):
    live = run.Live(None, run.ARMS["FC"], {}, None, None, tmp_path, {"turns": {}, "deleted": set()}, lambda *a, **k: {}, phase="ask")
    with pytest.raises(ValueError, match="cannot run in a teach_only or teach_from cell"):
        run.run_longrange(live, [])
