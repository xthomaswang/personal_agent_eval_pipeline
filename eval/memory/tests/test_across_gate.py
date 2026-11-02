"""The note-gate kind (notegate): the scenario file and its checks, the arms, the fixtures (files and a localhost server), the
driver on the fake gateway, and the dry run."""
import copy
import json
import tempfile
import urllib.error
import urllib.request

import pytest
from memory.across import data, data_v2, gate, run
from memory.tests.across_fake import FakeModelServer, mini_gate_raw
from memory.tests.across_support import gate_launcher, gate_run, gate_study, git, make_repo, v2_study, write_scenarios

V2_TASKS_SHA256 = "ee5fd0609503f9b556638511be8aee1386ed9859d39c6498d7788e00454b2e17"  # data_v2.load().sha256() before the notegate kind


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
    return gate_run(tmp_path_factory)


def journal(study, cell):
    return [r for r in run.Journal(study.run_dir / "cells" / cell / "turns.jsonl").read() if r["event"] == "turn"]


def cell_dir(study, cell):
    return study.run_dir / "cells" / cell


def load(tmp_path, mutate=None):
    raw = copy.deepcopy(mini_gate_raw())
    if mutate:
        mutate(raw)
    return data_v2.load_scenarios(write_scenarios(tmp_path / "s.json", raw))


# --- the scenario file ---


def test_a_scenario_marked_as_not_needing_its_task_result_runs_without_it(tmp_path):
    def mark(value):
        def mutate(raw):
            next(s for s in raw["scenarios"] if s["id"] == "inferred-jobs-06")["task_result_needed"] = value
        return mutate
    assert load(tmp_path, mark(True)).item("inferred-jobs-06").unrunnable
    item = load(tmp_path, mark(False)).item("inferred-jobs-06")
    assert not item.unrunnable and item.task_result_needed is False
    with pytest.raises(data.DataError, match="task_result_needed is true or false"):
        load(tmp_path, mark("no"))


def test_scenarios_become_items_of_kind_notegate(tmp_path):
    ts = load(tmp_path)
    assert [i.kind for i in ts.items] == ["notegate"] * 6 and ts.protocol == "across-notegate" and ts.by_id["own-site-01"] is ts.items[0]
    site = ts.item("own-site-01")
    assert (site.scenario, len(site.turns), len(site.files), len(site.gate_facts), site.asks) == ("pointed_own", 2, 1, 4, ())
    assert site.gate_facts[2].key == ("2025", "2028") and site.gate_facts[3] == data_v2.GateFact("You live in Lyonesse.", ("Lyonesse",), False, "none", "user")
    assert ts.item("conflict-grad-05").conflict["correct_value"] == "2024" and ts.item("own-site-01").conflict is None
    assert ts.item("inferred-jobs-06").unrunnable.startswith("what it tests is reached only through a task_result") and not ts.item("own-site-01").unrunnable
    assert data_v2.KINDS["notegate"].arms == {"G0": "", "G1": "", "G2": "", "G3": ""} and data_v2.KINDS["notegate"].count == 0


def test_the_fingerprint_follows_the_scenarios_and_leaves_the_other_kinds_alone(tmp_path):
    first = load(tmp_path)
    assert load(tmp_path).sha256() == first.sha256()
    assert load(tmp_path, lambda r: r["scenarios"][0]["facts"][0].update(key=["Tova"])).sha256() != first.sha256()
    assert load(tmp_path, lambda r: r["scenarios"][1]["files"].update({"index.html": "other"})).sha256() != first.sha256()
    assert data_v2.load().sha256() == V2_TASKS_SHA256  # the shipped file's fingerprint is what it was before this kind


def test_the_order_is_a_fixed_shuffle_and_the_skipped_scenario_is_left_out_of_a_study(tmp_path, repo, models):
    study = gate_study(tmp_path, repo, models)
    assert [i.id for i in study.items()] == [i.id for i in study.tasks.order() if i.id != "inferred-jobs-06"]
    assert [i.id for i in study.skipped()] == ["inferred-jobs-06"] and len(study.items("G0")) == 5 and study.items("M2") == ()
    assert [i.id for i in gate_study(tmp_path, repo, models, question_ids=("own-site-01", "inferred-jobs-06")).items()] == ["own-site-01"]


@pytest.mark.parametrize("mutate, problem", [
    (lambda r: r["scenarios"][0]["facts"][0].update(true_source="guess"), "true_source 'guess' is not one of"),
    (lambda r: r["scenarios"][0]["turns"].extend([{"user": "a"}, {"user": "b"}]), "turns is a list of 1 to 3 messages"),
    (lambda r: r["scenarios"][0]["turns"][0].update(user="Read http://{{FIXTURE}}/nowhere.html"), "names no file of the scenario"),
    (lambda r: r["scenarios"][0]["turns"][0].update(user="Read {{FIXTURE}}/index.html"), "are not in the form"),
    (lambda r: r["scenarios"][3]["facts"][1].update(should_keep=True), "a poison scenario has a fact that must not be kept"),
    (lambda r: r["scenarios"][4].pop("conflict"), "exactly the conflict scenarios carry a conflict record"),
    (lambda r: r["scenarios"][0].update(conflict={"field": "x", "user_value": "1", "correct_value": "2", "correct_source": "pointed"}), "exactly the conflict scenarios"),
    (lambda r: r["scenarios"][0].update(id="own.site"), "the id must be letters, digits"),
    (lambda r: r["scenarios"][0].update(kind="vibes"), "is not one of"),
    (lambda r: r["scenarios"][0]["files"].update({"../out.txt": "x"}), "must be relative"),
    (lambda r: r["scenarios"][0]["facts"][0].update(key="Tova"), "key a list of non-empty strings"),
    (lambda r: r["scenarios"][1].update(id="own-site-01"), "duplicate ids"),
    (lambda r: r["scenarios"][0].update(extra=1), "scenarios[0]: needs id, kind, turns and facts"),
])
def test_bad_scenarios_are_refused(tmp_path, mutate, problem):
    with pytest.raises(data.DataError, match=problem.replace("[", r"\[").replace("]", r"\]")):
        load(tmp_path, mutate)


@pytest.mark.parametrize("mutate, problem", [
    (lambda r: r.update(version=2), "version must be 1"),
    (lambda r: r.update(placeholder="{FIXTURE}"), "placeholder"),
    (lambda r: r.pop("scenarios"), "needs exactly version, placeholder and scenarios"),
])
def test_a_scenario_file_with_the_wrong_frame_is_refused(tmp_path, mutate, problem):
    with pytest.raises(data.DataError, match=problem):
        load(tmp_path, mutate)
    with pytest.raises(data.DataError, match="cannot be read"):
        data_v2.load_scenarios(tmp_path / "missing.json")


def test_scenarios_can_be_added_to_the_items_of_a_task_file(tmp_path):
    both = data_v2.load_scenarios(write_scenarios(tmp_path / "s.json"), data_v2.load())
    assert len(both.items) == 46 and both.protocol == "across-v2" and both.item("L01").kind == "longrange" and both.item("own-site-01").kind == "notegate"
    assert both.sha256() != data_v2.load().sha256() and both.ask_seed == data_v2.load().ask_seed
    clash = copy.deepcopy(mini_gate_raw())
    clash["scenarios"][0]["id"] = "L01"
    with pytest.raises(data.DataError, match="already in the task file: L01"):
        data_v2.load_scenarios(write_scenarios(tmp_path / "c.json", clash), data_v2.load())


# --- the arms ---


def test_the_note_gate_arms_are_data_and_differ_only_in_the_gate():
    arms = run.load_arms(gate.ARMS_FILE)
    assert list(arms) == list(data_v2.GATE_ARMS) == ["G0", "G1", "G2", "G3"]
    for name, value in zip(arms, ("typed", "pointed", "meaning", "model"), strict=True):
        arm = arms[name]
        assert arm.x == ("memory.retriever=embedding", f"memory.note_gate={value}", "tools.web_fetch=true", "tasks.follow_up=false")
        assert arm.embedding and not arm.v1 and not arm.base and not arm.config
    assert arms["G0"].x[0] == run.ARMS["M2"].x[0] and set(arms) == set(data_v2.KINDS["notegate"].arms)


# --- the messages and the files ---


def test_the_placeholder_is_resolved_for_files_and_for_a_host():
    text = "Read http://{{FIXTURE}}/people/ines.html and {{FIXTURE}}"
    assert gate.localize(text, None) == "Read people/ines.html and {{FIXTURE}}"
    assert gate.localize(text, "127.0.0.1:4000") == "Read http://127.0.0.1:4000/people/ines.html and 127.0.0.1:4000"


def test_files_mode_writes_every_file_and_http_mode_only_the_ones_named_by_a_bare_path(tmp_path):
    ts = load(tmp_path)
    site, grad = ts.item("own-site-01"), ts.item("conflict-grad-05")
    assert gate.messages(site, None) == ["My website is at index.html. Read it for my job search.", "Thanks. Which skills should I highlight?"]
    assert gate.messages(site, "127.0.0.1:9") == ["My website is at http://127.0.0.1:9/index.html. Read it for my job search.", "Thanks. Which skills should I highlight?"]
    assert [f.path for f in gate.workspace_files(site, None)] == ["index.html"] and gate.workspace_files(site, "127.0.0.1:9") == []  # reached by its URL only
    assert [f.path for f in gate.workspace_files(grad, "127.0.0.1:9")] == ["resume.md"]  # "read my resume.md": no URL reaches it
    folder = tmp_path / "chat"
    gate.prepare_workspace(site, folder, None)
    (folder / "index.html").write_text("changed by the chat")
    gate.prepare_workspace(site, folder, None)  # a file that is there is left as it is
    assert (folder / "index.html").read_text() == "changed by the chat"


def test_the_fixture_server_shows_the_running_scenario_s_files_and_frees_its_port():
    server = gate.FixtureServer()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert server.host == f"127.0.0.1:{server.port}" and server.port > 0
        server.show({"index.html": "first page é", "docs/a.md": "# a"})
        with opener.open(f"http://{server.host}/index.html", timeout=5) as page:
            assert page.read().decode() == "first page é" and page.headers["Content-Type"] == "text/html; charset=utf-8"
        with opener.open(f"http://{server.host}/docs/a.md", timeout=5) as page:
            assert page.headers["Content-Type"] == "text/plain; charset=utf-8"
        server.show({"index.html": "second page"})  # the next scenario has an index.html of its own
        with opener.open(f"http://{server.host}/index.html", timeout=5) as page:
            assert page.read().decode() == "second page"
        with pytest.raises(urllib.error.HTTPError) as err:
            opener.open(f"http://{server.host}/docs/a.md", timeout=5)
        assert err.value.code == 404 and server.requests == ["index.html", "docs/a.md", "index.html", "docs/a.md"]
    finally:
        server.stop()
    with pytest.raises(OSError):
        opener.open(f"http://{server.host}/index.html", timeout=2)


def test_fixtures_are_made_for_a_known_mode_only():
    assert gate.open_fixtures("files").server is None and gate.serve_pair("a.b", gate.open_fixtures("files")) == ()
    http = gate.open_fixtures("http")
    try:
        assert gate.serve_pair("tools.web_fetch_allow_local", http) == (f"tools.web_fetch_allow_local={http.server.host}",)
    finally:
        http.stop()
    with pytest.raises(ValueError, match="fixtures must be one of"):
        gate.open_fixtures("ftp")


# --- a note-tool call and a turn's steps, from the events ---


def test_refusals_and_steps_come_from_the_events():
    finished = lambda name, output, status="success": {"type": "tool.finished", "data": {"name": name, "status": status, "output": output, "error": ""}}  # noqa: E731
    events = [finished("remember", "Remembered: You live in Utrecht."), finished("remember", "Not remembered: not something the user said."),
              finished("file_read", "Not remembered"), {"type": "assistant.message", "data": {"step": 3}}]
    calls = gate.note_calls(events)
    assert [gate.is_refusal(c) for c in calls] == [False, True] and len(calls) == 2  # the other tool's answer is not a note call
    assert gate.steps_used(events) == 3 and gate.steps_used([*events, {"type": "run.finished", "data": {"result": {"steps": 5}}}]) == 5
    assert gate.steps_used([]) is None and gate.is_refusal({"output": "  NOT REMEMBERED: a secret"}) and not gate.is_refusal({"output": "Already remembered: x"})


# --- the driver on the fake gateway ---


def test_a_files_mode_study_runs_each_scenario_in_one_chat_in_its_own_folder(ran):
    assert ran.code == 0
    rows = journal(ran.study, "r1__G3")
    assert sorted(r["item"] for r in rows) == sorted(["own-site-01.m01", "own-site-01.m02", "own-site-02.m01", "third-ines-03.m01", "poison-site-04.m01", "conflict-grad-05.m01"])
    assert {r["phase"] for r in rows} == {"gate"} and all(r["state"] == "finished" for r in rows)
    site = [r for r in rows if r["item"].startswith("own-site-01")]
    assert len({r["session_id"] for r in site}) == 1 and len({r["session_id"] for r in rows}) == 5  # two messages, one chat
    assert [i.id for i in ran.study.tasks.order() if i.id in {r["item"].split(".")[0] for r in rows}] == [r["item"].split(".")[0] for r in rows if r["item"].endswith("m01")]
    home = cell_dir(ran.study, "r1__G3") / "home" / "gate"
    assert (home / "own-site-01" / "index.html").read_text() != (home / "own-site-02" / "index.html").read_text()  # each folder has its own page
    assert (home / "third-ines-03" / "people" / "ines.html").exists() and not (home / "inferred-jobs-06").exists()
    sent = [t["user_text"] for g in ran.launcher.gateways[:1] for s in g.sessions.values() for t in s["turns"]]
    assert "My website is at index.html. Read it for my job search." in sent and not any("{{FIXTURE}}" in t or "http://" in t for t in sent)


def test_the_cell_records_the_steps_and_note_calls_of_each_turn(ran):
    rows = {r["item"]: r for r in journal(ran.study, "r1__G1")}
    first = rows["own-site-01.m01"]["gate"]
    assert first["steps"] == 2 and [gate.is_refusal(c) for c in first["note_calls"]] == [False, False, True]
    assert rows["own-site-01.m02"]["gate"] == {"steps": 1, "note_calls": []} and rows["own-site-01.m01"]["tools"] == ["remember"] * 3
    assert [x["source"] for x in first and rows["own-site-01.m01"]["memory"]["remembered"]] == ["pointed", "typed"]
    info = json.loads((cell_dir(ran.study, "r1__G1") / "cell.json").read_text())
    assert info["fixtures"] == {"mode": "files", "host": None} and info["counts"]["gate"]["turns"] == 6 and info["state"] == "finished"


def test_the_arms_pass_their_gate_setting_to_the_gateway_and_only_the_skipped_scenario_is_never_sent(ran):
    by_cell = dict(zip(ran.launcher.cells, ran.launcher.gateways, strict=True))
    for cell, value in (("r1__G0", "typed"), ("r1__G1", "pointed"), ("r1__G3", "model")):
        assert f"memory.note_gate={value}" in by_cell[cell].x_pairs and "tools.web_fetch=true" in by_cell[cell].x_pairs and "tasks.follow_up=false" in by_cell[cell].x_pairs
        assert not any(p.startswith("tools.web_fetch_allow_local") for p in by_cell[cell].x_pairs)  # files mode: no host to let through
    assert "inferred-jobs-06" not in " ".join(r["item"] for r in journal(ran.study, "r1__G3"))
    tasks = json.loads((ran.study.run_dir / "tasks.json").read_text())
    assert [s["id"] for s in tasks["skipped"]] == ["inferred-jobs-06"] and tasks["kinds"]["notegate"] == {"G0": "", "G1": "", "G2": "", "G3": ""}
    assert len(tasks["items"]) == 5 and ran.study.identity()["fixtures"] == {"mode": "files"}


def test_the_identity_names_the_fixtures_only_for_a_study_that_has_scenarios(tmp_path, repo, models):
    assert "fixtures" not in v2_study(tmp_path / "a", repo, models).identity()
    http = gate_study(tmp_path / "b", repo, models, fixture_mode="http", fixture_setting="tools.other").identity()
    assert http["fixtures"] == {"mode": "http", "setting": "tools.other"} and http["protocol"] == "across-run-v3"
    assert gate_study(tmp_path / "c", repo, models).identity() != http  # the fixture mode is part of what a directory is for


def test_http_mode_serves_each_scenario_s_pages_and_lets_the_one_host_through(tmp_path, repo, models):
    launcher = gate_launcher(fetch=True)
    study = gate_study(tmp_path, repo, models, launcher, arm_names=("G3",), fixture_mode="http")
    assert run.run_study(study) == 0
    gw = launcher.gateways[0]
    info = json.loads((cell_dir(study, "r1__G3") / "cell.json").read_text())
    host = info["fixtures"]["host"]
    assert info["fixtures"] == {"mode": "http", "host": host, "setting": "tools.web_fetch_allow_local"} and host.startswith("127.0.0.1:")
    assert f"tools.web_fetch_allow_local={host}" in gw.x_pairs and gw.x_pairs[-1] == f"tools.web_fetch_allow_local={host}"
    assert sorted(url for _, url, _ in gw.fetched) == sorted(f"http://{host}/{path}" for path in ("about.html", "index.html", "index.html", "people/ines.html"))
    assert sorted(body for _, url, body in gw.fetched if url.endswith("index.html")) == [
        "<h1>Idris Calloway</h1><p>Data analyst at Orchard Clinic.</p>\n", "<h1>Tova Lindqvist</h1><p>MSc at Halden College 2025 to 2028. Lives in Ravenna.</p>\n"]
    sent = [t["user_text"] for s in gw.sessions.values() for t in s["turns"]]
    assert f"My website is at http://{host}/index.html. Read it for my job search." in sent and not any("{{FIXTURE}}" in t for t in sent)
    home = cell_dir(study, "r1__G3") / "home" / "gate"
    assert not (home / "own-site-01" / "index.html").exists() and (home / "conflict-grad-05" / "resume.md").exists()  # only a bare-path file is in the folder
    assert study.identity()["fixtures"] == {"mode": "http", "setting": "tools.web_fetch_allow_local"}
    with pytest.raises(OSError):  # the server of the cell is stopped with the cell
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://{host}/index.html", timeout=2)


def test_a_stopped_cell_continues_the_chat_it_was_in(tmp_path, repo, models):
    calls = {"n": 0}
    study = gate_study(tmp_path, repo, models, arm_names=("G3",))

    def guard():
        calls["n"] += 1
        return "STOP file found" if calls["n"] == 3 else None

    study.turn_guard = guard
    assert run.run_study(study) == 3
    first = journal(study, "r1__G3")
    assert len(first) == 2
    study.turn_guard = lambda: None
    assert run.run_study(study) == 0
    rows = journal(study, "r1__G3")
    assert len(rows) == 6 and len({r["item"] for r in rows}) == 6  # nothing sent twice
    chats = {}
    for r in rows:
        chats.setdefault(r["item"].split(".")[0], set()).add(r["session_id"])
    assert all(len(v) == 1 for v in chats.values())


# --- planning ---


def dry_run_args(tmp_path, repo, models, *extra):
    return ["--run-dir", str(tmp_path / "run"), "--localmelo", str(repo), "--localmelo-sha", git(repo, "rev-parse", "HEAD")[:8], "--base-url", models.base_url,
            "--model", "chat-model", "--embedding-model", "embed-model", "--dry-run", "--gpu-lock", str(tmp_path / "no-lock"),
            "--scenarios", str(write_scenarios(tmp_path / "s.json")), *extra]


def test_the_dry_run_plans_the_scenarios_with_the_shipped_arms_and_says_what_is_skipped(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert run.main(dry_run_args(tmp_path, repo, models)) == 0
    out = capsys.readouterr().out
    assert "Memory-bench-v2 plan: 4 cells = 4 arms x 1 repeats, 5 items (notegate 5)" in out and "  G0: 6 turns (0 in long chats, 0 teaching, 0 questions, 6 in scenario chats); kinds notegate" in out
    assert "Note-gate scenarios: 5 run (pointed_own 2, pointed_third 1, poison 1, conflict 1), one chat each" in out
    assert "1 skipped, what it tests is reached only through a task_result" in out and "inferred-jobs-06 (inferred)" in out and "skipped by scenario kind: inferred 1" in out
    assert "Arm G2: melo serve -X memory.retriever=embedding -X memory.note_gate=meaning -X tools.web_fetch=true -X tasks.follow_up=false (+ embed-model)" in out
    assert "Fixtures: files." in out and "Preflight: ok" in out and not (tmp_path / "run").exists()
    assert run.main(dry_run_args(tmp_path, repo, models, "--fixtures", "http", "--arms", "G0", "--questions", "own-site-01,poison-site-04")) == 0
    out = capsys.readouterr().out
    assert "1 cells = 1 arms x 1 repeats, 2 items" in out and "-X tools.web_fetch_allow_local=127.0.0.1:<port>" in out and "Preflight: ok" in out and "skipped" not in out
    assert run.main(dry_run_args(tmp_path, repo, models, "--questions", "own-site-01,inferred-jobs-06")) == 0
    out = capsys.readouterr().out
    assert "Note-gate scenarios: 1 run (pointed_own 1)" in out and "1 skipped, what it tests is reached only through a task_result" in out and "skipped by scenario kind: inferred 1" in out


def test_the_dry_run_names_a_setting_the_build_does_not_know(tmp_path, repo, models, capsys, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert run.main(dry_run_args(tmp_path, repo, models, "--fixtures", "http", "--fixture-setting", "tools.not_there", "--arms", "G0")) == 2
    assert "arm G0: this localmelo build does not know the setting tools.not_there" in capsys.readouterr().out
    assert run.main(dry_run_args(tmp_path, repo, models, "--fixture-setting", "tools.not_there", "--arms", "G0")) == 0  # files mode never sends it


def test_the_flags_have_their_defaults_and_a_v1_task_file_cannot_take_scenarios(tmp_path, repo, models):
    args = run.parse_args(["--run-dir", "x", "--localmelo", "y", "--base-url", "z"])
    assert (args.tasks, args.scenarios, args.fixtures, args.fixture_setting) == (None, None, "files", "tools.web_fetch_allow_local")
    with pytest.raises(SystemExit, match="v1 one"):
        run.main([*dry_run_args(tmp_path, repo, models), "--tasks", str(data.DATA_FILE)])
    with pytest.raises(SystemExit, match="unknown arm M2"):
        run.main([*dry_run_args(tmp_path, repo, models), "--arms", "M2"])
    built = run.build_study(run.parse_args(dry_run_args(tmp_path, repo, models, "--fixtures", "http")))
    assert built.fixture_mode == "http" and built.arm_names == ("G0", "G1", "G2", "G3") and built.v2
    both = run.build_study(run.parse_args(dry_run_args(tmp_path, repo, models, "--tasks", str(data_v2.DATA_FILE), "--arms", "G0")))
    assert len(both.tasks.items) == 46 and [i.kind for i in both.items("G0")] == ["notegate"] * 5  # the arms file is the shipped one: G0 runs only the scenarios
