"""The recall kind (questions on what memory holds before the cell starts), the seeded HOME of an arm and the memory
import step, on the fake gateway."""
import json

import pytest
from memory.across import data, data_v2, run
from memory.tests.across_fake import FakeModelServer
from memory.tests.across_support import make_repo, make_study


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


def recall_raw(**extra_items):
    def item(id_, category, text, answer, **kw):
        return {"id": id_, "kind": "recall", "lang": "en", "category": category, "asks": [{"text": text, "answer": answer, **kw}]}

    return {
        "version": 2, "protocol": "mini-recall", "ask_seed": 3, "fixtures": [], "filler": [], "sets": {},
        "items": [
            item("R1", "habit", "Which tool manages Python environments in a new project?", "uv"),
            item("R2", "abstain", "What is my favourite food?", ""),
            {**item("R3", "project", "What is the app's code name?", "Lark"), "workspace": "Developer/app"},
            item("R4", "harm", "What is 12 times 12?", "144", reject=["uv"]),
            *extra_items.values(),
        ],
    }


def test_recall_items_parse_and_an_abstain_question_is_graded_as_nothing_to_recall():
    ts = data_v2.parse(recall_raw(), None)
    r1, r2, r3, r4 = (ts.item(i) for i in ("R1", "R2", "R3", "R4"))
    assert (r1.category, r1.asks[0].kind, r1.workspace) == ("habit", "recall", "")
    assert (r2.asks[0].kind, r2.asks[0].answer) == ("nothing", "")
    assert r3.workspace == "Developer/app" and r4.asks[0].reject == ("uv",)


@pytest.mark.parametrize("row, problem", [
    ({"category": "abstain", "asks": [{"text": "What is my favourite food?", "answer": "pizza"}]}, "an abstain question has no answer"),
    ({"category": "habit", "asks": [{"text": "Which shell do I use?", "answer": ""}]}, "must be non-empty"),
    ({"category": "harm", "asks": [{"text": "What is 3 + 4?", "answer": "7"}]}, "lists the remembered trivia"),
    ({"category": "project", "workspace": "../out", "asks": [{"text": "Where is the repo?", "answer": "here"}]}, "below the cell's HOME"),
    ({"category": "gossip", "asks": [{"text": "Who said it?", "answer": "Ann"}]}, "is not one of"),
])
def test_bad_recall_items_are_refused(row, problem):
    raw = recall_raw(bad={"id": "R9", "kind": "recall", "lang": "en", **row})
    with pytest.raises(data.DataError, match=problem):
        data_v2.parse(raw, None)


def test_the_shipped_task_file_has_no_recall_items_and_still_loads():
    ts = data_v2.load()
    assert not [i for i in ts.items if i.kind == "recall"] and data_v2.V2["kinds"]["recall"] == 0


def make_seed(root):
    seed = root / "seed"
    (seed / ".claude" / "projects" / "@app" / "memory").mkdir(parents=True)
    (seed / ".claude" / "CLAUDE.md").write_text("@~/rules.md\n")
    (seed / "rules.md").write_text("- Use uv for new Python projects.\n")
    (seed / ".claude" / "projects" / "@app" / "memory" / "app.md").write_text("The app's code name is Lark.\n")
    (seed / "seed.json").write_text(json.dumps({"workspaces": {"@app": "Developer/app"}}))
    return seed


def test_seed_home_copies_the_files_and_names_the_project_folder_after_the_workspace(tmp_path):
    seed, home = make_seed(tmp_path), tmp_path / "cell" / "home"
    home.mkdir(parents=True)
    written = run.seed_home(home, seed)
    folder = run.project_folder_name(home.resolve() / "Developer" / "app")
    assert folder.startswith("-") and "/" not in folder and "_" not in folder
    assert sorted(written) == sorted([".claude/CLAUDE.md", "rules.md", f".claude/projects/{folder}/memory/app.md"])
    assert (home / "Developer" / "app").is_dir() and not (home / "seed.json").exists()
    assert (home / ".claude" / "projects" / folder / "memory" / "app.md").read_text() == "The app's code name is Lark.\n"


def test_the_seed_fingerprint_follows_the_bytes(tmp_path):
    seed = make_seed(tmp_path)
    before = run.seed_sha256(seed)
    (seed / "rules.md").write_text("- Use pip.\n")
    assert run.seed_sha256(seed) != before


def test_an_arms_file_reads_the_seed_relative_to_itself_and_the_import_switch(tmp_path):
    make_seed(tmp_path)
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({
        "I1": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "seed": "seed", "import": True},
        "E0": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "seed": "seed"},
    }))
    arms = run.load_arms(path)
    assert arms["I1"].seed == str((tmp_path / "seed").resolve()) and arms["I1"].import_all and not arms["E0"].import_all
    assert arms["I1"].identity()["import"] is True and arms["I1"].identity()["seed_sha256"] == run.seed_sha256(tmp_path / "seed")
    assert "import" not in arms["E0"].identity() and "seed_sha256" not in run.ARMS["M2"].identity()


class StubGateway:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def call(self, method, path, body=None, ok=(200,)):
        self.calls.append((method, path, body))
        if self.fail == path:
            raise run.client.GatewayError(400, "invalid_request")
        if path.endswith("/scan"):
            return {"batch_id": "b7", "partial": False, "sources": [{"id": "claude-1", "items_kept": 2}, {"id": "codex-2", "items_kept": 1}]}
        return {"batch_id": "b7", "written_ids": [1, 2, 3], "skipped": 0}


def test_import_memory_brings_in_every_source_and_keeps_the_scan(tmp_path):
    gw = StubGateway()
    summary = run.import_memory(gw, tmp_path)
    assert gw.calls[1] == ("POST", "/api/memory/import", {"batch_id": "b7", "source_ids": ["claude-1", "codex-2"]})
    assert {k: summary[k] for k in ("sources", "kept", "written", "skipped", "partial")} == {"sources": 2, "kept": 3, "written": 3, "skipped": 0, "partial": False}
    doc = json.loads((tmp_path / "import.json").read_text())
    assert doc["scan"]["batch_id"] == "b7" and doc["apply"]["written_ids"] == [1, 2, 3] and "error" not in doc


def test_a_failed_import_is_recorded_not_raised(tmp_path):
    summary = run.import_memory(StubGateway(fail="/api/memory/import"), tmp_path)
    assert summary["written"] == 0 and summary["error"].startswith("GatewayError")
    assert "error" in json.loads((tmp_path / "import.json").read_text())


def test_a_recall_study_seeds_both_arms_imports_in_one_and_asks_each_question_in_its_own_chat(tmp_path, models):
    make_seed(tmp_path)
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps({
        "I1": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "seed": "seed", "import": True},
        "E0": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "seed": "seed"},
    }))
    launcher = run_launcher()
    study = make_study(tmp_path, make_repo(tmp_path), models, launcher, tasks=data_v2.parse(recall_raw(), None),
                       arms=run.load_arms(arms_file), arm_names=("I1", "E0"))
    assert run.run_study(study) == 0
    by_cell = dict(zip(launcher.cells, launcher.gateways, strict=True))
    for cell, imports in (("r1__I1", True), ("r1__E0", False)):
        gw, cdir = by_cell[cell], study.run_dir / "cells" / cell
        info = json.loads((cdir / "cell.json").read_text())
        assert ".claude/CLAUDE.md" in info["seed"]["files"] and (cdir / "home" / "Developer" / "app").is_dir()
        assert (("POST", "/api/memory/import") in gw.requests) is imports and ("import" in info) is imports
        rows = [r for r in run.Journal(cdir / "turns.jsonl").read() if r["event"] == "turn"]
        assert sorted(r["item"] for r in rows) == ["R1.q1", "R2.q1", "R3.q1", "R4.q1"] and len({r["session_id"] for r in rows}) == 4
        r3 = next(r for r in rows if r["item"] == "R3.q1")
        assert ("DELETE", f"/api/sessions/{r3['session_id']}") in gw.requests
    info = json.loads((study.run_dir / "cells" / "r1__I1" / "cell.json").read_text())
    assert info["import"]["sources"] == 2 and info["import"]["written"] == 2  # CLAUDE.md's one line + the notes folder's one


def run_launcher():
    from memory.tests.across_support import Launcher
    return Launcher(store=True, delay=0.002)


def test_a_project_question_is_asked_in_its_workspace_under_the_cell_home_and_a_done_one_is_not_asked_again(tmp_path, models):
    study = make_study(tmp_path, make_repo(tmp_path), models, tasks=data_v2.parse(recall_raw(), None), arm_names=("M2",), keep_ask_chats=True)
    cell = {"cell_id": "c", "repeat": 1, "arm": "M2"}
    sent = []

    def send(phase, item, text, **kw):
        sent.append((item, kw.get("workspace")))
        return {"session_id": f"s-{item}"}

    home = tmp_path / "cellhome"
    done = {"turns": {("ask", "R1.q1"): {"session_id": "s-old"}}, "deleted": set()}
    live = run.Live(study, run.ARMS["M2"], cell, None, run.Journal(tmp_path / "j.jsonl"), tmp_path, done, send, home)
    assert run.run_recall(live, [study.tasks.item(i) for i in ("R1", "R3", "R4")]) is False
    assert sent == [("R3.q1", home / "Developer" / "app"), ("R4.q1", None)]


def test_the_report_has_the_import_table_and_recall_accuracy_by_category(tmp_path, models):
    from memory.across import report_v2
    make_seed(tmp_path)
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps({
        "I1": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "seed": "seed", "import": True}}))
    tasks = data_v2.parse(recall_raw(), None)
    study = make_study(tmp_path, make_repo(tmp_path), models, run_launcher(), tasks=tasks, arms=run.load_arms(arms_file), arm_names=("I1",))
    assert run.run_study(study) == 0
    rep = report_v2.analyse(study.run_dir, tasks)
    assert set(rep["kinds"]) == {"recall"} and rep["imports"]["r1__I1"]["written"] == 2
    acc = rep["accuracy"]["recall"]["I1"]
    assert set(acc) == {"all", "habit", "project", "abstain", "harm"} and acc["all"]["n"] == 4
    rows = {r["qid"]: r for r in rep["rows"]}
    assert rows["R2.q1"]["category"] == "abstain" and all("used_sources" in r for r in rows.values())
    text = report_v2.render(rep)
    assert "## Memory import" in text and "**recall**" in text and "| abstain |" in text
