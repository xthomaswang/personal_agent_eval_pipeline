"""The v2 report from a run against the fake gateway: validity, accuracy with intervals, miss classes, sources, pairs,
timings, arm variants, the comparison of two runs, and the cases that make an item invalid."""
import json
import shutil
import sqlite3

import pytest
import yaml
from memory.across import report, report_v2, run, stores
from memory.tests.across_fake import FakeModelServer, mini_raw_v2
from memory.tests.across_support import make_repo, v2_launcher, v2_run, v2_study

FACT = "Today's call is in room 4B."


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return v2_run(tmp_path_factory)


@pytest.fixture(autouse=True)
def real_home(tmp_path, monkeypatch):
    home = tmp_path / "realhome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    yield home
    assert list(home.iterdir()) == []


@pytest.fixture
def rep(ran):
    return report_v2.analyse(ran.study.run_dir, ran.tasks)


@pytest.fixture
def copy(ran, tmp_path):
    """A private copy of the session's run directory, for tests that change it."""
    target = tmp_path / "copy"
    shutil.copytree(ran.study.run_dir, target)
    return target


def rate(rep, kind, arm, name):
    r = rep["accuracy"][kind][arm][name]
    return r["k"], r["n"]


def cell_edit(run_dir, cell, fn):
    path = run_dir / "cells" / cell / "turns.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        fn(row)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- the shipped run ---


def test_every_item_is_valid_for_its_arm_and_the_arms_run_their_kinds(rep):
    assert rep["arms"] == ["N0", "M2", "B0", "FC"]
    assert rep["ran"] == {"N0": {"items": 4, "valid": 4}, "M2": {"items": 4, "valid": 4}, "B0": {"items": 2, "valid": 2}, "FC": {"items": 2, "valid": 2}}
    assert all(not v for v in rep["invalid"].values())
    windows = {(c["arm"], c["item"]): c["window"] for c in rep["chats"] if c["kind"] == "longrange"}
    assert not windows[("N0", "L1")]["fact_in_window"] and windows[("N0", "L1")]["cap"] == 64 and windows[("N0", "L1")]["messages_before"] > 64
    assert windows[("FC", "L1")]["fact_in_window"] and windows[("FC", "L1")]["cap"] == 256 and windows[("M2", "L2")]["fact_turn"] == 3


def test_accuracy_with_wilson_intervals_by_kind_arm_and_note_expected(rep):
    assert rate(rep, "longrange", "N0", "all") == (0, 2) and rate(rep, "longrange", "M2", "all") == (1, 2) and rate(rep, "longrange", "FC", "all") == (2, 2)
    assert rate(rep, "longrange", "M2", "note not expected") == (0, 1) and rate(rep, "longrange", "M2", "note expected") == (1, 1)
    assert rate(rep, "longrange", "FC", "note not expected") == (1, 1)
    assert [rate(rep, "followup", a, n) for a in ("M2", "B0") for n in ("Q1", "Q2", "Q2 given Q1 right")] == [(2, 2), (0, 2), (0, 2), (2, 2), (0, 2), (0, 2)]
    assert rate(rep, "followup", "N0", "Q1") == (0, 2) and "FC" not in rep["accuracy"]["followup"]
    two_of_two = rep["accuracy"]["longrange"]["FC"]["all"]
    assert two_of_two["lo"] == pytest.approx(0.342, abs=1e-3) and two_of_two["hi"] == 1.0
    assert report_v2.wilson(0, 0) is None and report_v2.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-3)


def test_every_wrong_answer_gets_a_miss_class(rep):
    assert rep["misses"]["longrange"] == {"N0": {"out of window": 2}, "M2": {"not saved": 1}, "B0": {}, "FC": {}}
    assert rep["misses"]["followup"] == {"N0": {"memory off": 4}, "M2": {"not recalled": 2}, "B0": {"not recalled": 2}, "FC": {}}
    wrong = [r for r in rep["rows"] if r["valid"] and not r["correct"]]
    assert wrong and all(r["miss"] for r in wrong) and all(not r["miss"] for r in rep["rows"] if r["correct"])


def test_where_the_right_answers_came_from(rep):
    assert rep["sources"] == {"N0": {}, "M2": {"note": 1}, "B0": {}, "FC": {"own-chat turn": 2}}


def test_each_question_records_the_recall_and_the_stale_flag(rep):
    row = next(r for r in rep["rows"] if r["arm"] == "M2" and r["qid"] == "L2.q1")
    assert row["gold_hit"] and row["gold_rank"] == 1 and row["saved"] is True and row["n_used"] == 1 and row["correct"] and row["source"] == "note"
    assert row["remembered"] == [] and row["recall_ms"] is not None and row["ttfe_ms"] is not None and row["wall_ms"] > 0 and row["stale"] is False
    stale = next(r for r in rep["rows"] if r["arm"] == "M2" and r["qid"] == "L1.q1")
    assert not stale["correct"] and stale["saved"] is False and not stale["gold_hit"] and stale["miss"] == "not saved" and stale["context"] is None
    fact_turn = next(c for c in rep["chats"] if c["arm"] == "M2" and c["item"] == "L2")
    assert fact_turn["fact_made_note"] is True and fact_turn["fact_note_saved"] is True
    assert next(c for c in rep["chats"] if c["arm"] == "M2" and c["item"] == "L1")["fact_made_note"] is False  # note_expected: false, and none was made
    q2 = next(r for r in rep["rows"] if r["arm"] == "M2" and r["qid"] == "P1.q2")
    assert q2["saved"] is True and q2["miss"] == "not recalled" and q2["label"] == "followup.q2" and not q2["gold_hit"]


def test_the_recall_block_is_an_estimate_until_the_build_reports_it(rep):
    assert any("recall-block characters below are an estimate" in n for n in rep["notes"])
    lat = rep["latency"]["longrange"]["M2"]
    assert lat["block_estimated"]["n"] == 2 and lat["block_real"]["n"] == 0
    assert "recall block (chars, ESTIMATE)" in report_v2.render(rep)


def test_request_bytes_and_times_per_kind_and_arm(rep):
    longrange = rep["latency"]["longrange"]
    assert longrange["FC"]["req_bytes"]["median"] > longrange["N0"]["req_bytes"]["median"] > 0  # the 256-message window is sent whole
    assert all(longrange[a]["wall_ms"]["n"] == 2 for a in ("N0", "M2", "FC")) and "FC" not in rep["latency"]["followup"]
    chats = {(c["arm"], c["item"]): c for c in rep["chats"]}
    row = next(r for r in rep["rows"] if r["arm"] == "FC" and r["qid"] == "L1.q1")
    assert chats[("FC", "L1")]["req_bytes_max"] >= row["req_bytes"] > 0


def test_pairs_of_arms_per_kind(rep):
    assert sorted(rep["paired"]["longrange"]) == ["M2-FC", "N0-FC", "N0-M2"] and sorted(rep["paired"]["followup"]) == ["M2-B0", "N0-B0", "N0-M2"]
    nf = rep["paired"]["longrange"]["N0-FC"]
    assert {k: nf["overall"][k] for k in ("n", "both", "only_a", "only_b", "neither")} == {"n": 2, "both": 0, "only_a": 0, "only_b": 2, "neither": 0}
    assert nf["overall"]["p"] == pytest.approx(0.5) and nf["overall"]["diff_pp"] == -100.0
    followup = rep["paired"]["followup"]["N0-M2"]["by_kind"]
    assert sorted(followup) == ["followup.q1", "followup.q2"] and followup["followup.q1"]["only_b"] == 2 and followup["followup.q2"]["neither"] == 2


def test_tools_and_notes_of_the_long_chats(rep):
    tools = {c["arm"]: c["tools"] for c in rep["chats"] if c["item"] == "L1"}
    assert tools["M2"] == {"file_read": 16, "list_files": 6} == tools["FC"]
    assert all(c["denied"] == 0 for c in rep["chats"])


def test_the_report_renders_every_section(rep):
    text = report_v2.render(rep)
    for heading in ("# Memory bench v2: ", "## Which arm runs what", "## Items run and invalid items", "## Accuracy", "## Misses", "## Notes made in the long chats",
                    "## Tools the long chats used", "## Paired comparison of the arms", "## Time, request size and recall block"):
        assert heading in text
    assert "| all | 0% (0/2) [0-66] | 50% (1/2) [9-91] | - | 100% (2/2) [34-100] |" in text
    assert "| Q2 given Q1 right | - | 0% (0/2) [0-66] | 0% (0/2) [0-66] | - |" in text
    assert "| out of window | 2 | 0 | 0 | 0 |" in text and "| note | 0 | 1 | 0 | 0 |" in text and "| own-chat turn | 0 | 0 | 0 | 2 |" in text
    assert "N0: out, M2: out, FC: in" in text and "M2: none, B0: none, N0: none" in text
    assert "| FC | memory.across_chats=false | agent.window_messages=256 | no |" in text and "| M2 | memory.retriever=embedding | - | yes |" in text
    json.dumps(rep)  # the report is plain data


def test_report_main_writes_the_files_for_a_v2_run(ran, copy, tmp_path):
    tasks_file = tmp_path / "mini_v2.yaml"
    tasks_file.write_text(yaml.safe_dump(mini_raw_v2(), allow_unicode=True), encoding="utf-8")
    assert report.main([str(copy), "--tasks", str(tasks_file)]) == 0
    assert (copy / "report.md").read_text().startswith("# Memory bench v2: copy") and json.loads((copy / "report.json").read_text())["arms"] == ["N0", "M2", "B0", "FC"]
    with pytest.raises(SystemExit):
        report.main([str(copy), "--tasks", str(tasks_file), "--compare", str(copy), "M2", "extra"])


# --- why an item is invalid ---


def invalid(rep, arm):
    return {i["item"]: i["reason"] for i in rep["invalid"][arm]}


def test_a_full_context_chat_over_the_prompt_cap_is_invalid(copy, ran):
    path = copy / "cells" / "r1__FC" / "proxy.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["req_bytes"] = 150_000
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = report_v2.analyse(copy, ran.tasks)
    assert set(invalid(rep, "FC")) == {"L1", "L2"} and all("over the prompt cap" in why for why in invalid(rep, "FC").values())
    assert "FC" not in rep["accuracy"]["longrange"] and rep["ran"]["FC"] == {"items": 2, "valid": 0}  # left out of the accuracy
    assert "- FC L1 (repeat 1): a request of 150,000 bytes, over the prompt cap" in report_v2.render(rep)


def test_a_fact_that_is_still_in_the_window_makes_an_out_arm_invalid(copy, ran):
    db = copy / "cells" / "r1__N0" / "home" / stores.SESSIONS_DB
    conn = sqlite3.connect(db)
    sid = next(r["session_id"] for r in map(json.loads, (copy / "cells" / "r1__N0" / "turns.jsonl").read_text().splitlines()) if r.get("item") == "L1.q1" and r["event"] == "turn")
    conn.execute("UPDATE windows SET data = ? WHERE session_id = ?", (json.dumps({"v": 1, "messages": [{"role": "user", "content": FACT}]}), sid))
    conn.commit()
    conn.close()
    rep = report_v2.analyse(copy, ran.tasks)
    assert invalid(rep, "N0") == {"L1": "the fact turn is still in the window"}
    assert rep["ran"]["N0"] == {"items": 4, "valid": 3} and rate(rep, "longrange", "N0", "all") == (0, 1)


def test_no_window_row_an_unfinished_cell_and_a_failed_fact_turn_are_invalid(copy, ran):
    conn = sqlite3.connect(copy / "cells" / "r1__FC" / "home" / stores.SESSIONS_DB)
    conn.execute("DELETE FROM windows")
    conn.commit()
    conn.close()
    info_path = copy / "cells" / "r1__M2" / "cell.json"
    info = json.loads(info_path.read_text())
    info["state"] = "paused"
    info_path.write_text(json.dumps(info))
    cell_edit(copy, "r1__N0", lambda r: r.update(state="failed", status="timeout") if r.get("item") == "L2.t03" else None)  # L2's fact turn
    rep = report_v2.analyse(copy, ran.tasks)
    assert set(invalid(rep, "FC").values()) == {"no window row for the chat"}
    assert set(invalid(rep, "M2").values()) == {"the cell has not finished, so its stores are not read"} and set(invalid(rep, "M2")) == {"L1", "L2"}
    assert invalid(rep, "N0") == {"L2": "the fact turn did not finish"}
    assert rep["misses"]["followup"]["M2"] == {"unchecked": 2}  # a paused cell's notes are not read: saved or not is not known
    assert any("Not finished: r1__M2" in n for n in rep["notes"])


def test_a_chat_that_never_reached_its_question_is_invalid(copy, ran):
    path = copy / "cells" / "r1__FC" / "turns.jsonl"
    rows = [line for line in path.read_text().splitlines() if '"L2.q1"' not in line]
    path.write_text("\n".join(rows) + "\n")
    assert invalid(report_v2.analyse(copy, ran.tasks), "FC") == {"L2": "the chat did not reach the question"}


def test_wrong_answers_of_a_memory_arm_with_the_note_recalled_and_of_full_context_get_the_other_classes(copy, ran):
    cell_edit(copy, "r1__M2", lambda r: r.update(answer="I do not know.") if r.get("item") == "L2.q1" else None)
    cell_edit(copy, "r1__FC", lambda r: r.update(answer="I do not know.") if r.get("item") == "L1.q1" else None)
    rep = report_v2.analyse(copy, ran.tasks)
    assert rep["misses"]["longrange"]["M2"] == {"not saved": 1, "recalled but not used": 1}
    assert rep["misses"]["longrange"]["FC"] == {"in window but not used": 1}


def test_an_own_chat_turn_recalled_without_a_note_is_recalled_but_not_used_when_the_answer_is_wrong(copy, ran):
    """M2 on a build that recalls a chat's own evicted turn: L1 has no note ("not saved" before), the turn is in `used`."""
    def own_turn(row):
        if row.get("item") == "L1.q1" and row["event"] == "turn":
            row["memory"] = {"used": [{"kind": "chat", "text": f"User: {FACT}\nmelo: Noted.", "source": "this chat, Oct 31", "session_id": row["session_id"]}],
                             "remembered": [], "refused": ""}
            row["answer"] = "I do not know."
    cell_edit(copy, "r1__M2", own_turn)
    rep = report_v2.analyse(copy, ran.tasks)
    row = next(r for r in rep["rows"] if r["arm"] == "M2" and r["qid"] == "L1.q1")
    assert row["saved"] is False and row["gold_hit"] and row["miss"] == "recalled but not used"
    assert rep["misses"]["longrange"]["M2"] == {"recalled but not used": 1}
    cell_edit(copy, "r1__M2", lambda r: r.update(answer="In room 4B.") if r.get("item") == "L1.q1" else None)  # and right: the chat's own turn
    right = report_v2.analyse(copy, ran.tasks)
    assert right["sources"]["M2"] == {"note": 1, "own-chat turn": 1} and rate(right, "longrange", "M2", "all") == (2, 2)


def test_a_failed_question_turn_is_wrong_and_classed_as_failed(copy, ran):
    cell_edit(copy, "r1__M2", lambda r: r.update(state="failed", status="timeout", answer="", timed_out=True) if r.get("item") == "L2.q1" else None)
    rep = report_v2.analyse(copy, ran.tasks)
    assert rep["misses"]["longrange"]["M2"] == {"not saved": 1, "turn failed": 1} and rate(rep, "longrange", "M2", "all") == (0, 2)


# --- variants and newer builds ---


def variant_arms(tmp_path):
    path = tmp_path / "arms.json"
    base = ["memory.retriever=embedding"]
    path.write_text(json.dumps({
        "M2-off": {"label": "context off", "x": [*base, "memory.context_query=off"], "embedding": True, "base": "M2"},
        "M2-on": {"label": "context on", "x": [*base, "memory.context_query=on"], "embedding": True, "base": "M2"},
        "N0": {"label": "off", "x": list(run.MEMORY_OFF_X), "embedding": False}}))
    return run.load_arms(path)


def test_variants_of_m2_follow_the_arm_table_of_m2_and_report_the_recall_context(tmp_path):
    arms = variant_arms(tmp_path)
    assert arms["M2-on"].base == "M2" and arms["N0"].base == "" and arms["M2-off"].embedding
    models = FakeModelServer(["chat-model", "embed-model"])
    try:
        study = v2_study(tmp_path, make_repo(tmp_path), models, v2_launcher(new_build=True), arms=arms, arm_names=("M2-off", "M2-on", "N0"))
        assert study.items("M2-off") == study.items("M2") == study.items("M2-on") and [i.id for i in study.items("M2-on")][:2] != []
        assert run.preflight(study) == [] and study.identity()["arms"]["M2-on"]["base"] == "M2" and "base" not in study.identity()["arms"]["N0"]
        assert run.run_study(study) == 0
    finally:
        models.stop()
    rep = report_v2.analyse(study.run_dir, study.tasks)
    assert rep["arms"] == ["M2-off", "M2-on", "N0"] and rep["bases"] == {"M2-off": "M2", "M2-on": "M2"}
    assert rep["ran"] == {"M2-off": {"items": 4, "valid": 4}, "M2-on": {"items": 4, "valid": 4}, "N0": {"items": 4, "valid": 4}}  # the long-range rule of M2 held for both
    assert rate(rep, "followup", "M2-off", "Q2") == (0, 2) and rate(rep, "followup", "M2-on", "Q2") == (2, 2)  # the follow-up finds its note with the context on
    assert rate(rep, "followup", "M2-on", "Q2 given Q1 right") == (2, 2)
    assert rep["contexts"] == {"M2-off": {"off": 6}, "M2-on": {"used": 6}, "N0": {"not reported": 6}}
    assert all(r["context"] == "used" for r in rep["rows"] if r["arm"] == "M2-on") and not any(r["block_estimated"] for r in rep["rows"] if r["block_chars"] is not None)
    assert not any("recall-block characters below are an estimate" in n for n in rep["notes"])  # the build reports block_chars
    assert "M2-off-M2-on" in rep["paired"]["followup"] and "M2-off-M2-on" in rep["paired"]["longrange"]
    pair = rep["paired"]["followup"]["M2-off-M2-on"]["by_kind"]["followup.q2"]
    assert (pair["only_a"], pair["only_b"]) == (0, 2)
    text = report_v2.render(rep)
    assert "Variants follow their base arm's row: M2-off follows M2, M2-on follows M2." in text and "## Recall context" in text and "| used | 0 | 6 | 0 |" in text
    real = rep["latency"]["followup"]["M2-on"]
    assert real["block_real"]["n"] == 4 and real["block_estimated"]["n"] == 0


def test_a_variant_of_an_arm_no_kind_lists_has_nothing_to_run(tmp_path):
    arms = {"X": run.Arm("X", "x", ("memory.retriever=bm25",), False, base="L1")}
    models = FakeModelServer(["chat-model", "embed-model"])
    try:
        study = v2_study(tmp_path, make_repo(tmp_path), models, arms=arms, arm_names=("X",))
        assert study.items("X") == () and "arm X has nothing to run here" in run.preflight(study)[0]
    finally:
        models.stop()


# --- the comparison of two runs ---


@pytest.fixture(scope="module")
def other_run(tmp_path_factory, ran):
    """M2 once more on the same tasks, as a build that finds the follow-up's note would run it (context_query on)."""
    base = tmp_path_factory.mktemp("other")
    arms = {"M2": run.Arm("M2", "mixed, context on", ("memory.retriever=embedding", "memory.context_query=on"), True)}
    patch = pytest.MonkeyPatch()
    (base / "home").mkdir()
    patch.setenv("HOME", str(base / "home"))
    models = FakeModelServer(["chat-model", "embed-model"])
    try:
        study = v2_study(base, make_repo(base), models, v2_launcher(model_calls=True, new_build=True), arms=arms, arm_names=("M2",))
        assert run.run_study(study) == 0
    finally:
        models.stop()
        patch.undo()
    return study


def test_two_runs_are_paired_ask_by_ask(ran, other_run, rep):
    other = report_v2.analyse(other_run.run_dir, ran.tasks)
    cmp = report_v2.compare(rep, other, ran.tasks, "M2")
    assert list(cmp["arms"]) == ["M2"] and cmp["notes"] == [] and cmp["arms"]["M2"]["paired"] == 6  # L1, L2, and two asks of each follow-up item
    q2 = cmp["arms"]["M2"]["kinds"]["followup"]["by_kind"]["followup.q2"]
    assert (q2["n"], q2["only_a"], q2["only_b"], q2["both"], q2["neither"]) == (2, 0, 2, 0, 0)
    assert cmp["arms"]["M2"]["kinds"]["longrange"]["overall"]["both"] == 1 and cmp["arms"]["M2"]["kinds"]["longrange"]["overall"]["neither"] == 1
    differences = cmp["arms"]["M2"]["differences"]
    assert [d["qid"] for d in differences] == ["P1.q2", "P2.q2"] and all(not d["this"] and d["other"] and d["this_miss"] == "not recalled" and not d["other_miss"] for d in differences)
    # without an arm: every arm both runs have (only M2 here)
    assert list(report_v2.compare(rep, other, ran.tasks)["arms"]) == ["M2"]
    with pytest.raises(ValueError, match="arm FC is not in both runs"):
        report_v2.compare(rep, other, ran.tasks, "FC")


def test_report_compare_writes_the_comparison_into_the_report(ran, other_run, copy, tmp_path):
    tasks_file = tmp_path / "mini_v2.yaml"
    tasks_file.write_text(yaml.safe_dump(mini_raw_v2(), allow_unicode=True), encoding="utf-8")
    assert report.main([str(copy), "--tasks", str(tasks_file), "--compare", str(other_run.run_dir), "M2"]) == 0
    text = (copy / "report.md").read_text()
    assert "## Compared with run" in text and "### M2: 6 asks paired" in text and "| followup.q2 | 2 | 0 | 0 | 2 | 0 | -100.0 | 0.500 |" in text
    assert "- P1.q2 (followup.q2): right in the other run only." in text
    assert json.loads((copy / "report.json").read_text())["compare"]["arms"]["M2"]["paired"] == 6
    with pytest.raises(SystemExit, match="arm FC is not in both runs"):
        report.main([str(copy), "--tasks", str(tasks_file), "--compare", str(other_run.run_dir), "FC"])


def test_compare_is_for_v2_runs_only(tmp_path):
    (tmp_path / "across.json").write_text(json.dumps({"config": {"protocol": "across-run-v2"}}))
    with pytest.raises(SystemExit, match="v2 task file"):
        report.main([str(tmp_path), "--compare", str(tmp_path)])
