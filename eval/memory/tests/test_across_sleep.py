"""The sleep kind (day 1 teaching round by round, one memory sleep, day 2 questions), the sleep arm option on v1
studies too, and the report's sleep section, on the fake gateway."""
import json

import pytest
from memory.across import data, data_v2, report_v2, run
from memory.tests.across_fake import FakeModelServer
from memory.tests.across_support import Launcher, make_repo, make_study


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


def sleep_raw(**extra):
    def item(id_, category, facts, text, answer, **kw):
        return {"id": id_, "kind": "sleep", "lang": "en", "category": category, **({"facts": facts} if facts else {}),
                "asks": [{"text": text, "answer": answer, **kw}]}

    return {
        "version": 2, "protocol": "mini-sleep", "ask_seed": 4, "fixtures": [], "filler": [], "sets": {},
        "items": [
            item("D1", "duplicate", ["I live in Utrecht.", "Home is Utrecht these days."], "Where do I live?", "Utrecht"),
            item("C1", "change", ["My gym day is Monday.", "By the way my gym day moved to Friday, can you list three stretches?"],
                 "Which day is my gym day?", "Friday", reject=["Monday"]),
            item("A1", "abstain", [], "What is my favourite food?", ""),
            *extra.values(),
        ],
    }


def arms_file(tmp_path):
    path = tmp_path / "arms.json"
    dev = {"ui.mode": "developer"}
    path.write_text(json.dumps({
        "S0": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "config": dev},
        "S1": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "config": dev, "sleep": "rules"},
    }))
    return path


def test_sleep_items_parse():
    ts = data_v2.parse(sleep_raw(), None)
    assert ts.item("C1").facts[0] == "My gym day is Monday." and ts.item("A1").asks[0].kind == "nothing"


@pytest.mark.parametrize("row, problem", [
    ({"category": "duplicate", "facts": ["I live in Lisbon."], "asks": [{"text": "Where?", "answer": "Lisbon"}]}, "says it twice"),
    ({"category": "change", "facts": ["My swim day is Tuesday.", "Swim day moved to Sunday."], "asks": [{"text": "Swim day?", "answer": "Sunday"}]},
     "rejects the old value"),
    ({"category": "change", "facts": ["Swim day moved to Sunday.", "My swim day is Tuesday."], "asks": [{"text": "Swim day?", "answer": "Sunday", "reject": ["Tuesday"]}]},
     "in its last fact"),
    ({"category": "lost", "facts": ["Summarise this. Also, my dentist is Dr. Lin."], "asks": [{"text": "Dentist?", "answer": "Dr. Wu"}]}, "in none of the facts"),
    ({"category": "abstain", "facts": ["I like tea."], "asks": [{"text": "Favourite drink?", "answer": ""}]}, "teaches nothing"),
    ({"category": "messy", "facts": ["x y z."], "asks": [{"text": "q?", "answer": "z"}]}, "is not one of"),
])
def test_bad_sleep_items_are_refused(row, problem):
    with pytest.raises(data.DataError, match=problem):
        data_v2.parse(sleep_raw(bad={"id": "B9", "kind": "sleep", "lang": "en", **row}), None)


def test_day_one_teaches_round_by_round_then_one_sleep_then_day_two_and_only_the_sleep_arm_sleeps(tmp_path, models):
    launcher = Launcher(store=True, delay=0.002, sleep_changes=[{"op": "O1", "before_ids": [1, 2], "after_id": 2, "text": "You live in Utrecht.", "source_turn": 1}])
    tasks = data_v2.parse(sleep_raw(), None)
    study = make_study(tmp_path, make_repo(tmp_path), models, launcher, tasks=tasks, arms=run.load_arms(arms_file(tmp_path)), arm_names=("S0", "S1"))
    assert run.run_study(study) == 0
    gws = dict(zip(launcher.cells, launcher.gateways, strict=True))
    order = [i.id for i in tasks.order() if i.facts]
    for cell, sleeps in (("r1__S0", []), ("r1__S1", ["rules"])):
        cdir = study.run_dir / "cells" / cell
        rows = [r for r in run.Journal(cdir / "turns.jsonl").read() if r["event"] == "turn"]
        assert [r["item"] for r in rows if r["phase"] == "teach"] == [f"{i}.f1" for i in order] + [f"{i}.f2" for i in order]
        assert sorted(r["item"] for r in rows if r["phase"] == "ask") == ["A1.q1", "C1.q1", "D1.q1"]
        gw = gws[cell]
        assert gw.sleeps == sleeps
        info = json.loads((cdir / "cell.json").read_text())
        assert ("sleep" in info) is bool(sleeps)
        if sleeps:
            reqs = [p for m, p in gw.requests if m == "POST" and p in ("/api/sessions", "/api/memory/sleep")]
            night = reqs.index("/api/memory/sleep")
            assert night == 4 and len(reqs) == 4 + 1 + 3  # after the 4 teaching chats, before the 3 question chats
            assert info["sleep"] == {**info["sleep"], "variant": "rules", "changes": 1, "by_op": {"O1": 1}, "model_calls": 0}
            assert (cdir / "sleep.json").exists() and (cdir / "after_sleep_records.db").exists()
    rep = report_v2.analyse(study.run_dir, tasks)
    assert set(rep["sleeps"]) == {"r1__S1"} and rep["sleeps"]["r1__S1"]["changes"][0]["op"] == "O1"
    assert set(rep["sleeps"]["r1__S1"]["items"]) == {"D1", "C1"}
    assert set(rep["accuracy"]["sleep"]["S1"]) == {"all", "duplicate", "change", "abstain"}
    text = report_v2.render(rep)
    assert "## Memory sleep" in text and "**r1__S1** after the night" in text


def test_a_v1_study_sleeps_between_the_teaching_and_the_questions(tmp_path, models):
    launcher = Launcher(store=True, delay=0.002)
    study = make_study(tmp_path, make_repo(tmp_path), models, launcher, arms=run.load_arms(arms_file(tmp_path)), arm_names=("S1",))
    assert run.run_study(study) == 0
    gw = launcher.gateways[0]
    assert gw.sleeps == ["rules"]
    rows = [r for r in run.Journal(study.run_dir / "cells" / "r1__S1" / "turns.jsonl").read() if r["event"] == "turn"]
    n_teach = sum(r["phase"] == "teach" for r in rows)
    reqs = [p for m, p in gw.requests if m == "POST" and p in ("/api/sessions", "/api/memory/sleep")]
    assert reqs.index("/api/memory/sleep") == n_teach  # every teaching chat first, then the night, then the questions


def test_a_sleep_without_developer_mode_is_recorded_as_failed_not_raised(tmp_path, models):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({"S1": {"x": ["memory.retriever=embedding"], "embedding": True, "base": "M2", "sleep": "rules"}}))
    study = make_study(tmp_path, make_repo(tmp_path), models, Launcher(store=True, delay=0.002), tasks=data_v2.parse(sleep_raw(), None),
                       arms=run.load_arms(path), arm_names=("S1",))
    assert run.run_study(study) == 0
    info = json.loads((study.run_dir / "cells" / "r1__S1" / "cell.json").read_text())
    assert info["sleep"]["error"].startswith("GatewayError") and info["sleep"]["changes"] == 0


def test_the_shipped_sleep_file_and_its_arms_load():
    from collections import Counter
    ts = data_v2.load(data_v2.DATA_FILE.parent / "tasks_sleep.yaml")
    assert Counter(i.category for i in ts.items) == {c: 5 for c in data_v2.SLEEP_CATEGORIES} and {i.kind for i in ts.items} == {"sleep"}
    assert 0.30 <= sum(i.lang == "zh" for i in ts.items) / len(ts.items) <= 0.40
    arms = run.load_arms(data_v2.DATA_FILE.parent / "arms_sleep.json")
    assert (arms["S0"].sleep, arms["S1"].sleep) == ("", "rules") and all(a.settings() == {"ui.mode": "developer"} for a in arms.values())
