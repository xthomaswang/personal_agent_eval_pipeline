"""The across-chats report, from synthetic run directories and from a run against the fake gateway."""
import json

import pytest
import yaml
from memory.across import data, report, run, score
from memory.tests.across_fake import FakeModelServer, mini_raw
from memory.tests.across_support import Launcher, make_repo, make_study

TASKS = data.parse(mini_raw(), None)
SESSION = {"F1": "sF1", "F2": "sF2", "F3": "sF3", "F4": "sF4"}
MB = 1024  # rss is recorded in kB
POOL_MB = 1024 * 1024  # and the server's pool memory in bytes


def used(*sessions):
    return {"used": [{"kind": "record", "id": i, "text": "t", "source": "s", "session_id": s, "turn": 1, "at": 0} for i, s in enumerate(sessions)],
            "remembered": [], "refused": ""}


def turn(cell, phase, item, answer="", memory=None, recall=None, wall=300.0, state="finished", session=None, **extra):
    return {"event": "turn", "cell": cell, "phase": phase, "item": item, "key": f"k-{cell}-{item}", "state": state, "status": "completed",
            "session_id": session or (SESSION.get(item) if phase == "teach" else f"ask-{cell}-{item}"), "answer": answer, "wall_ms": wall,
            "ttfe_ms": 5.0, "wall_source": "stream", "server_ms": {"memory": 3.0, "run_started": 40.0, "first_message": 280.0, "finished": 290.0},
            "memory": memory, "recall": recall, "timed_out": False, "approvals_denied": 0, **extra}


def timings(embed=0, search=2, select=0, tokens=0, fallback="none"):
    """A `recall` as localmelo 5f1498f4 reports it: integer ms, select_tokens a dict, fallback "none" when none."""
    return {"retriever": "x", "fallback": fallback, "embed_ms": embed, "search_ms": search, "select_ms": select,
            "select_tokens": {"prompt": tokens, "completion": 12 if tokens else 0}, "candidates": 3, "used": 1}


def m2_turns(cell, repeat):
    def r(item):
        slow = repeat == 2 and item == "Q3"  # one question whose chat request an embeddings request overlaps
        return timings(80 if slow else 20, 2, 0, fallback="timeout" if slow else "none")
    return [
        turn(cell, "ask", "Q1", "Your cat is called Biscuit.", used("sF1"), r("Q1")),
        turn(cell, "ask", "Q2", "Biscuit.", used("sF1"), r("Q2")),
        turn(cell, "ask", "Q3", "Your roommate is 小满 (Xiaoman).", used("sF2"), r("Q3")),
        turn(cell, "ask", "Q4", "You live in Utrecht.", used("sF4"), r("Q4")),
        turn(cell, "ask", "Q5", "I don't know your favourite board game.", used(), r("Q5")),
    ]


def l1_turns(cell, repeat):
    r = lambda: timings(0, 2, 900 + 100 * (repeat - 1), 640)
    return [
        turn(cell, "ask", "Q1", "Biscuit.", used("sF1"), r(), wall=1500.0),
        turn(cell, "ask", "Q2", "I don't know.", used(), timings(0, 2, 0, 0), wall=1500.0),
        turn(cell, "ask", "Q3", "Your roommate is Alex.", used(), timings(0, 2, 0, 0), wall=1500.0),
        turn(cell, "ask", "Q4", "You lived in Lisbon and now Utrecht.", used("sF3", "sF4"), r(), wall=1500.0),
        turn(cell, "ask", "Q5", "Your favourite is Catan.", used("sF1"), r(), wall=1500.0),
    ]


def stage_rss(mb, loaded, pool_mb):
    return {"rss_kb": mb * MB, "loaded": loaded, "status": {"model_memory_used": pool_mb * POOL_MB}}


def proxy_requests(arm, repeat, item, start):
    """What the timing proxy logged for one turn that began at ``start`` (the turn ends at start + 5)."""
    def req(path, a, first, end, **kw):
        return {"method": "POST", "path": path, "model": "m", "stream": kw.get("stream"), "tools": kw.get("tools"), "t_start": a,
                "t_headers": first, "t_first_byte": first, "t_end": end, "status": 200, "bytes": 100, "req_bytes": 50, "error": ""}
    rows = []
    if arm == "M2":
        rows.append(req("/v1/embeddings", start + 0.1, start + 0.2, start + 0.2))  # the query
        slow = repeat == 2 and item == "Q3"
        rows.append(req("/v1/chat/completions", start + 0.3, start + 0.3 + (0.9 if slow else 0.3), start + 4, stream=False, tools=True))
        if slow:
            rows.append(req("/v1/embeddings", start + 0.25, start + 0.6, start + 0.6))  # in flight as the chat request starts
        rows.append(req("/v1/embeddings", start + 5.2, start + 5.3, start + 5.3))  # the turn is indexed after it ends
    if arm == "L1":
        rows.append(req("/v1/chat/completions", start + 0.1, start + 0.9, start + 0.95, stream=False, tools=False))  # recall selection
        rows.append(req("/v1/chat/completions", start + 1.0, start + 2.2, start + 4, stream=False, tools=True))
    return rows


def write_run(root, specs, state="finished", tasks_sha=None, schedule=None, proxy=True, cold=True):
    """specs: (arm, repeat, ask turns, {stage: rss}, warm-up). Teaching turns are added; two of them were intervened on."""
    cells_state = {}
    for arm, repeat, asks, rss, warm in specs:
        cell = f"r{repeat}__{arm}"
        note = {"used": [], "remembered": [{"id": 1, "text": "x", "topic": "about"}], "refused": ""}
        teach = [turn(cell, "teach", f, "Noted.", note if arm != "N0" else None, session=SESSION[f]) for f in ("F1", "F2", "F3", "F4")]
        if (arm, repeat) == ("M2", 1):
            teach[1].update(approvals_denied=2, denied_tools=["file_write", "python_exec"])
        if (arm, repeat) == ("L1", 2):  # the third teaching message hit the timeout
            teach[2].update(state="failed", status="timeout", timed_out=True, answer="", memory=None)
        d = root / "cells" / cell
        d.mkdir(parents=True)
        journal = []
        for k, t in enumerate([*teach, *asks]):
            t.update(started_epoch=1000.0 + 10 * k, ended_epoch=1005.0 + 10 * k)
            journal.append(t)
        (d / "turns.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in journal) + "\n")
        if proxy:
            log = [row for t in journal for row in proxy_requests(arm, repeat, t["item"], t["started_epoch"])]
            (d / "proxy.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\nnot json\n")
        info = {"cell": {"repeat": repeat, "arm": arm, "cell_id": cell}, "arm": arm, "state": state, "wall_s": 100.0 + repeat, "rss": rss, "warmup": warm}
        if cold:  # the embedding model was left loaded by every cell but the first (M2, repeat 1) and unloaded by the driver
            left = (arm, repeat) != ("M2", 1)
            info["cold_start"] = {"model": "embed", "was_loaded": left, "unload_status": 200 if left else None, "attempts": int(left),
                                  "wait_ms": 700.0 if left else 0.0, "loaded_after": False}
        (d / "cell.json").write_text(json.dumps(info))
        cells_state[cell] = {"state": state, "arm": arm, "repeat": repeat}
    arms = sorted({s[0] for s in specs}, key=["M2", "L1", "B0", "N0"].index)
    (root / "across.json").write_text(json.dumps({"config": {
        "protocol": "across-run-v2", "tasks_sha256": tasks_sha or TASKS.sha256(), "arms": {a: {} for a in arms}, "repeats": 2,
        "questions": [q.id for q in TASKS.questions], "localmelo_sha": "5f1498f41234567", "model": "chat", "embedding_model": "embed",
        "keep_ask_chats": False}, "schedule_sha256": "x", "cells": cells_state, "complete": True}))
    (root / "schedule.json").write_text(json.dumps(schedule or [{"cell_id": c} for c in cells_state]))
    (root / "environment.json").write_text(json.dumps({"start": {"sw_vers": "ProductName: test\nBuildVersion: 1", "xcodebuild_version": "Xcode 1"}}))
    return root


@pytest.fixture
def synthetic(tmp_path):
    specs = []
    for repeat in (1, 2):
        m2 = {"before": stage_rss(10240, ["chat"], 8000), "after_warmup": stage_rss(10752, ["chat", "embed"], 8600),
              "after_teach": stage_rss(10800, ["chat", "embed"], 8600), "after": stage_rss(10880, ["chat", "embed"], 8600)}
        l1 = {"before": stage_rss(10240, ["chat"], 8000), "after_warmup": stage_rss(10240, ["chat"], 8000),
              "after_teach": stage_rss(10260, ["chat"], 8000), "after": stage_rss(10300, ["chat"], 8000)}
        specs.append(("M2", repeat, m2_turns(f"r{repeat}__M2", repeat), m2, {"chat_ms": 4000.0, "embed_ms": 900.0}))
        specs.append(("L1", repeat, l1_turns(f"r{repeat}__L1", repeat), l1, {"chat_ms": 3000.0}))
    return write_run(tmp_path / "run", specs)


def test_accuracy_by_kind_and_arm(synthetic):
    rep = report.analyse(synthetic, TASKS)
    acc = rep["accuracy"]
    assert rep["arms"] == ["M2", "L1"]
    assert acc["all"] == {"M2": (10, 10), "L1": (2, 10)}
    assert acc["exact"] == {"M2": (4, 4), "L1": (2, 4)}
    assert acc["crosslang"]["L1"] == (0, 2) and acc["superseded"]["L1"] == (0, 2) and acc["nothing"]["M2"] == (2, 2)
    assert acc["lookalike"]["M2"] == (0, 0)  # no such question in the mini set


def test_grades_follow_the_scorer(synthetic):
    rows = {(r["arm"], r["repeat"], r["qid"]): r for r in report.analyse(synthetic, TASKS)["rows"]}
    assert rows[("L1", 1, "Q4")]["outcome"] == score.MIXED and rows[("L1", 1, "Q2")]["outcome"] == score.ABSTAIN
    assert rows[("L1", 1, "Q5")]["outcome"] == score.WRONG and rows[("M2", 1, "Q3")]["outcome"] == score.CORRECT


def test_recall_hits_wrong_and_false_recall(synthetic):
    rc = report.analyse(synthetic, TASKS)["recall"]
    assert rc["M2"]["hit_at_3"] == (8, 8) and rc["L1"]["hit_at_3"] == (4, 8) and rc["L1"]["hit_at_1"] == (2, 8)
    assert rc["M2"]["wrong_recall"] == (0, 2) and rc["L1"]["wrong_recall"] == (2, 2)
    assert rc["M2"]["false_recall"] == (0, 2) and rc["L1"]["false_recall"] == (2, 2) and rc["L1"]["false_recall_taught"] == (2, 2)
    assert rc["L1"]["hit_at_3_by_kind"]["exact"] == (2, 4) and rc["L1"]["hit_at_3_by_kind"]["crosslang"] == (0, 2)


def test_latency_median_p90_and_select_tokens(synthetic):
    lt = report.analyse(synthetic, TASKS)["latency"]
    m2 = lt["M2"]["recall_ms"]  # 22 x9 and one slow embedding, 82
    assert (m2["n"], m2["median"], m2["p90"]) == (10, 22.0, pytest.approx(28.0))
    l1 = lt["L1"]["recall_ms"]  # 2 x4 (no candidates), 902 x3 (repeat 1), 1002 x3 (repeat 2)
    assert (l1["n"], l1["median"], l1["p90"]) == (10, 902.0, 1002.0)
    assert lt["M2"]["wall_ms"]["median"] == 300.0 and lt["L1"]["wall_ms"]["p90"] == 1500.0
    assert lt["L1"]["select_prompt"] == {"n": 6, "median": 640.0, "p90": 640.0, "mean": 640.0, "total": 3840}
    assert lt["L1"]["select_completion"]["total"] == 72 and lt["L1"]["select_completion"]["median"] == 12.0
    assert lt["M2"]["select_prompt"]["n"] == 0 and lt["M2"]["select_prompt"]["total"] == 0  # no selection is made
    assert lt["M2"]["fallbacks"] == (1, 10) and lt["L1"]["fallbacks"] == (0, 10)  # "none" is not a fallback
    assert lt["M2"]["pre_run_ms"]["median"] == 40.0 and lt["M2"]["ttfe_ms"]["median"] == 5.0


def test_cost_of_the_second_model(synthetic):
    cost = report.analyse(synthetic, TASKS)["cost"]
    assert cost["M2"]["rss_mb"] == {"before": 10240.0, "after_warmup": 10752.0, "after_teach": 10800.0, "after": 10880.0}
    assert cost["M2"]["rss_delta_mb"] == {"warmup": 512.0, "cell": 640.0} and cost["L1"]["rss_delta_mb"] == {"warmup": 0.0, "cell": 60.0}
    assert cost["M2"]["pool_delta_mb"] == 600.0 and cost["L1"]["pool_delta_mb"] == 0.0
    # every cell starts without the embedding model; M2 loads it, L1 never asks for it
    assert cost["M2"]["embedding_loaded"] == {"before": (0, 2), "after_warmup": (2, 2), "after_teach": (2, 2), "after": (2, 2)}
    assert cost["L1"]["embedding_loaded"] == {"before": (0, 2), "after_warmup": (0, 2), "after_teach": (0, 2), "after": (0, 2)}
    assert cost["M2"]["cold"] == cost["L1"]["cold"] == 2 and cost["L1"]["unload_wait_ms"] == 700.0 and cost["M2"]["unload_wait_ms"] == 700.0
    assert cost["M2"]["chat_loaded"]["after"] == (2, 2)
    assert cost["L1"]["rss_mb"]["after"] == 10300.0 and cost["M2"]["embed_warmup_ms"] == 900.0 and cost["L1"]["embed_warmup_ms"] is None
    assert cost["M2"]["chat_warmup_ms"] == 4000.0 and cost["M2"]["cell_wall_s"] == 101.5


def test_the_report_says_whether_every_cell_started_cold(synthetic, tmp_path):
    rep = report.analyse(synthetic, TASKS)
    assert rep["cold_start"] == {"cells": 4, "cold": 4, "unloaded_by_driver": 3, "skipped": 0} and not rep["notes"]
    asks = [turn("r1__M2", "ask", "Q1", "Biscuit.", used("sF1"), timings())]
    stages = {"before": stage_rss(1, ["chat"], 1), "after": stage_rss(2, ["chat", "embed"], 1)}
    never = write_run(tmp_path / "never", [("M2", 1, asks, stages, None)], cold=False)  # a run that did not unload
    text = report.render(report.analyse(never, TASKS))
    assert "Only 0 of 1 cells started with the embedding model unloaded" in " ".join(report.analyse(never, TASKS)["notes"])
    assert "**Not every cell started with the embedding model unloaded**: 0 of 1 did." in text
    # the driver said it unloaded the model, but the server's own list just after still had it: not cold
    liar = write_run(tmp_path / "liar", [("M2", 1, asks, {**stages, "before": stage_rss(1, ["chat", "embed"], 1)}, None)])
    assert report.analyse(liar, TASKS)["cold_start"]["cold"] == 0
    skipped = tmp_path / "skipped"
    write_run(skipped, [("M2", 1, asks, stages, None)])
    info = json.loads((skipped / "cells" / "r1__M2" / "cell.json").read_text())
    info["cold_start"] = {"model": "embed", "skipped": True}
    (skipped / "cells" / "r1__M2" / "cell.json").write_text(json.dumps(info))
    assert report.analyse(skipped, TASKS)["cold_start"] == {"cells": 1, "cold": 0, "unloaded_by_driver": 0, "skipped": 1}


def test_chat_time_to_first_byte_from_the_proxy_log(synthetic):
    rep = report.analyse(synthetic, TASKS)
    tf = rep["ttft"]
    m2 = tf["M2"]
    assert (m2["ask"]["n"], m2["ask"]["median"], m2["ask"]["p90"]) == (10, 300.0, pytest.approx(360.0))
    assert (m2["teach"]["n"], m2["teach"]["median"]) == (8, 300.0) and m2["streamed"] == (0, 18)
    assert (m2["overlapped"]["n"], m2["overlapped"]["median"]) == (1, 900.0) and (m2["clear"]["n"], m2["clear"]["median"]) == (17, 300.0)
    assert (m2["chat_requests"], m2["embeddings"], m2["embeddings_in_turns"]) == (18, 37, 19)  # queries, one neighbour, 18 indexed after their turn
    assert (m2["with_embedding"]["n"], m2["without_embedding"]["n"]) == (18, 0)
    assert m2["spearman_embed_ms"] == pytest.approx(1.0) and m2["pairs"] == 10
    l1 = tf["L1"]  # the recall selection (no tools) is not the chat call; the agent's call is, at 1.2 s
    assert (l1["ask"]["median"], l1["ask"]["n"], l1["teach"]["n"]) == (1200.0, 10, 7)  # the timed-out teaching turn is left out
    assert (l1["chat_requests"], l1["embeddings"], l1["with_embedding"]["n"], l1["without_embedding"]["n"]) == (36, 0, 0, 17)
    assert l1["spearman_embed_ms"] is None and l1["overlapped"]["n"] == 0


def test_turns_without_a_window_or_a_proxy_log_have_no_first_byte(tmp_path):
    asks = [turn("r1__M2", "ask", "Q1", "Biscuit.", used("sF1"), timings())]
    bare = {"before": stage_rss(1, None, 1), "after": stage_rss(2, None, 1)}
    root = write_run(tmp_path / "run", [("M2", 1, asks, bare, None)], proxy=False)
    rep = report.analyse(root, TASKS)
    assert rep["ttft"]["M2"]["ask"]["n"] == 0 and rep["ttft"]["M2"]["chat_requests"] == 0
    assert any("no proxy log" in n.lower() or "proxy log" in n for n in rep["notes"]) and "No proxy log." in report.render(rep)
    assert rep["cost"]["M2"]["embedding_loaded"]["before"] == (0, 0)  # the server did not say what was loaded


def test_main_call_choice_and_missing_first_byte(tmp_path):
    asks = [turn("r1__M2", "ask", "Q1", "Biscuit.", used("sF1"), timings())]
    root = write_run(tmp_path / "run", [("M2", 1, asks, {"before": stage_rss(1, [], 1), "after": stage_rss(1, [], 1)}, None)])
    log = report.proxy.read_log(root / "cells" / "r1__M2" / "proxy.jsonl")
    assert any(r["path"].endswith("/embeddings") for r in log)  # the unreadable last line was skipped
    # the chat request of every turn loses its first byte (the connection broke): no TTFT, no crash
    path = root / "cells" / "r1__M2" / "proxy.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
    for row in rows:
        if row["path"].endswith("/chat/completions"):
            row["t_first_byte"] = None
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rep = report.analyse(root, TASKS)
    assert rep["ttft"]["M2"]["ask"]["n"] == 0 and rep["ttft"]["M2"]["chat_requests"] == 5


def test_paired_comparison_by_question(synthetic):
    pair = report.analyse(synthetic, TASKS)["paired"]
    assert (pair["a"], pair["b"]) == ("M2", "L1")
    assert {k: pair["overall"][k] for k in ("n", "both", "only_a", "only_b", "neither")} == {"n": 10, "both": 2, "only_a": 8, "only_b": 0, "neither": 0}
    assert pair["overall"]["p"] == pytest.approx(2 / 256) and pair["overall"]["diff_pp"] == 80.0
    assert pair["by_kind"]["exact"]["only_a"] == 2 and pair["by_kind"]["nothing"]["only_a"] == 2
    low, high = pair["ci_diff_pp"]
    assert low <= 80.0 <= high and low > 0
    assert pair["wall_ms"] == {"n": 5, "median_a": 300.0, "median_b": 1500.0, "median_diff": -1200.0}
    assert pair["recall_ms"] == {"n": 5, "median_a": 22.0, "median_b": 952.0, "median_diff": -930.0}
    assert [q for q, _ in pair["only_a_always"]] == ["Q2", "Q3", "Q4", "Q5"] and pair["only_b_always"] == []


def test_interventions_are_counted_per_arm_and_phase(synthetic):
    iv = report.analyse(synthetic, TASKS)["interventions"]
    assert iv["M2"]["teach"] == {"turns": 8, "failed": 0, "made_a_note": 8, "refused": 0, "approvals_denied": 2, "turns_with_denied": 1, "timed_out": 0}
    assert iv["L1"]["teach"] == {"turns": 8, "failed": 1, "made_a_note": 7, "refused": 0, "approvals_denied": 0, "turns_with_denied": 0, "timed_out": 1}
    assert iv["M2"]["ask"]["timed_out"] == 0 and iv["L1"]["ask"]["approvals_denied"] == 0 and iv["L1"]["ask"]["turns"] == 10


def test_render_markdown(synthetic):
    text = report.render(report.analyse(synthetic, TASKS))
    assert text.startswith("# Memory across chats: run")
    assert "| all | 100% (10/10) | 20% (2/10) |" in text
    assert "| hit@3 | 100% (8/8) | 50% (4/8) |" in text
    assert "| false recall | 0% (0/2) | 100% (2/2) |" in text
    assert "| recall | 22.0 / 28.0 (n=10) | 902 / 1002 (n=10) |" in text
    assert "| 10 | 2 | 8 | 0 | 0 | +80.0 | 0.008 |" in text
    assert "## M2 against L1, by question" in text and "Right with M2, wrong with L1, in every repeat: Q2 (What is my cat called these days?)" in text
    assert "| rss after cell | 10,880 | 10,300 |" in text and "| embedding warm-up ms (M2: the load) | 900 | - |" in text
    assert "Every cell started with the embedding model unloaded (4 of 4 cells; the driver unloaded it in 3, the chat model never)." in text
    assert "| embedding model loaded before the cell | 0/2 | 0/2 |" in text and "| unload wait ms (when it had to unload) | 700 | 700 |" in text and "| rss added by the cell | +640 | +60 |" in text
    assert "| question turns | 300 / 360 (n=10) | 1200 / 1200 (n=10) |" in text and "| Spearman: turn embed ms and chat first byte | +1.00 (n=10) | - |" in text
    assert "| teaching: turns with a denied approval / requests denied | 1 / 2 | 0 / 0 |" in text
    first = text.index("## ")  # the first table of the report is the check of the "do not save it in a file" instruction
    assert text[first:].startswith("## Teaching turns that asked for tool approval") and text.index("## Accuracy") > first
    assert "| teaching turns that asked for approval | 12% (1/8) | 0% (0/8) |" in text and "| approval requests denied (teaching) | 2 | 0 |" in text
    assert "| teaching turns stopped by the timeout | 0% (0/8) | 12% (1/8) |" in text and "| question turns that asked for approval | 0% (0/10) | 0% (0/10) |" in text
    assert "| teaching: turns stopped by the timeout | 0 | 1 |" in text and "| prompt median / p90 | - / - | 640 / 640 |" in text
    assert "localmelo `5f1498f412`" in text and "ProductName: test BuildVersion: 1" in text


def test_notes_for_unfinished_cells_changed_tasks_and_old_builds(tmp_path):
    asks = [turn("r1__M2", "ask", "Q1", "Biscuit.", used("sF1")), turn("r1__M2", "ask", "Q5", "I don't know.")]
    root = write_run(tmp_path / "run", [("M2", 1, asks, {"before": stage_rss(1, [], 1), "after": stage_rss(3, [], 1)}, None)], state="paused", tasks_sha="old",
                     schedule=[{"cell_id": "r1__M2"}, {"cell_id": "r1__L1"}])
    rep = report.analyse(root, TASKS)
    text = " ".join(rep["notes"])
    assert "task file differs" in text and "Not finished: r1__M2, r1__L1" in text
    assert rep["paired"] is None and "Both arms are needed." in report.render(rep)
    assert rep["cost"]["M2"]["embed_warmup_ms"] is None and rep["latency"]["M2"]["recall_ms"]["median"] is None
    assert rep["cost"]["M2"]["rss_mb"]["after"] == 3.0


def test_aborted_cells_are_ignored(synthetic):
    (synthetic / "cells" / "r1__M2.aborted-1").mkdir()
    (synthetic / "cells" / "r1__M2.aborted-1" / "cell.json").write_text("{}")
    assert report.analyse(synthetic, TASKS)["accuracy"]["all"]["M2"] == (10, 10)


def test_a_gold_fact_that_was_never_taught_cannot_be_recalled(tmp_path):
    cell = "r1__M2"
    asks = [turn(cell, "ask", "Q1", "Biscuit.", used("sF1"))]
    root = write_run(tmp_path / "run", [("M2", 1, asks, {"before": stage_rss(1, [], 1), "after": stage_rss(2, [], 1)}, None)])
    path = root / "cells" / cell / "turns.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows if r.get("item") != "F1") + "\n")  # F1's teaching turn is gone
    row = report.analyse(root, TASKS)["rows"][0]
    assert row["correct"] and row["recall"]["hit_at_3"] is False


def test_statistics():
    assert report.percentile([], 50) is None and report.percentile([4.0], 90) == 4.0
    assert report.percentile([1, 2, 3, 4], 50) == 2.5 and report.percentile([1, 2, 3, 4, 5], 90) == 4.6
    assert report.mcnemar_p(0, 0) == 1.0 and report.mcnemar_p(5, 5) == 1.0
    assert report.mcnemar_p(8, 0) == pytest.approx(2 / 256) and report.mcnemar_p(0, 8) == report.mcnemar_p(8, 0)
    assert report.bootstrap_ci([]) is None
    low, high = report.bootstrap_ci([1.0, 1.0, 0.0, 1.0])
    assert 0.0 <= low <= 0.75 <= high <= 1.0
    assert report.recall_ms({"embed_ms": 1, "search_ms": 2, "select_ms": None}) == 3.0 and report.recall_ms(None) is None
    assert report.recall_ms({"embed_ms": None, "search_ms": None}) is None


def test_report_of_a_run_against_the_fake_gateway(tmp_path):
    models = FakeModelServer(["chat-model", "embed-model"], first_delay=0.02, embed_delay=0.02, slow_embed_delay=0.4)
    try:
        study = make_study(tmp_path, make_repo(tmp_path), models, Launcher(model_calls=True), arm_names=("M2", "L1", "N0"), repeats=2)
        assert run.run_study(study) == 0
    finally:
        models.stop()
    tasks_file = tmp_path / "mini.yaml"
    tasks_file.write_text(yaml.safe_dump(mini_raw(), allow_unicode=True), encoding="utf-8")
    assert report.main([str(study.run_dir), "--tasks", str(tasks_file)]) == 0
    rep = json.loads((study.run_dir / "report.json").read_text(encoding="utf-8"))
    assert rep["arms"] == ["M2", "L1", "N0"] and rep["accuracy"]["all"]["N0"][1] == 10
    assert rep["recall"]["N0"]["hit_at_3"] == [0, 8] and rep["recall"]["L1"]["hit_at_1"][0] >= 2
    assert rep["latency"]["L1"]["select_prompt"]["total"] > 0 and rep["latency"]["M2"]["recall_ms"]["n"] > 0
    assert rep["cost"]["M2"]["rss_mb"]["before"] == 2.0 and rep["cost"]["M2"]["embedding_loaded"]["after"] == [0, 2]  # the study's probe lists only the chat model
    assert rep["interventions"]["N0"]["teach"]["made_a_note"] == 0 and rep["interventions"]["L1"]["teach"]["made_a_note"] == 8
    assert rep["paired"]["overall"]["n"] == 10
    tf = rep["ttft"]
    # one chat call per turn goes through the proxy in every arm; only M2 sends embeddings (queries and the indexing of each turn)
    assert tf["N0"]["chat_requests"] == tf["M2"]["chat_requests"] == 18 and tf["N0"]["embeddings"] == 0 and tf["L1"]["embeddings"] == 0
    assert tf["L1"]["chat_requests"] == 36
    assert 18 < tf["M2"]["embeddings"] <= 36  # 18 queries, and the indexing of turns (the last ones land after the proxy stops)
    assert tf["M2"]["ask"]["n"] == 10 and tf["M2"]["ask"]["median"] >= 20 and tf["M2"]["streamed"] == [18, 18]
    assert tf["M2"]["overlapped"]["n"] >= 1  # the 0.4 s indexing of a teaching turn is still running when the next one calls the chat model
    text = (study.run_dir / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# Memory across chats: run") and "## M2 against L1, by question" in text and "## Chat time to first byte" in text
    assert "predates them" not in text  # the fake reports recall timings
