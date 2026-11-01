"""The public-benchmark report: accuracy with bootstrap intervals, paired differences, hits, ingest, judge and control."""
import inspect
import json

import pytest
from memory.public import report, run
from memory.tests.public_support import (
    PublicLauncher,
    judge_server,
    make_public_study,
    make_repo,
    result_row,
    write_run,
)

QUESTIONS = [
    {"question_id": qid, "question_type": qtype, "abstention": qid.endswith("_abs")}
    for qid, qtype in (("q1", "single-session-user"), ("q2", "single-session-user"), ("q3", "single-session-assistant"),
                       ("q4", "single-session-assistant"), ("q5", "multi-session"), ("q6_abs", "temporal-reasoning"))
]
TYPE = {q["question_id"]: q["question_type"] for q in QUESTIONS}
# per question: the verdicts of repeats 1 and 2 for arm A and arm B
VERDICTS = {
    "q1": ([1, 1], [1, 0]), "q2": ([1, 0], [0, 0]), "q3": ([0, 1], [0, 0]), "q4": ([1, 1], [1, 0]), "q5": ([1, 1], [1, 1]), "q6_abs": ([0, 1], [1, 1]),
}


def rows():
    out = []
    for qid, (a, b) in VERDICTS.items():
        for arm, verdicts in (("A", a), ("B", b)):
            for repeat, ok in enumerate(verdicts, 1):
                auto = arm == "A" and not (qid == "q3" and repeat == 1)
                out.append(result_row(
                    qid, TYPE[qid], arm, repeat, bool(ok), abstention=qid.endswith("_abs"), auto=auto, cover=1.0 if auto else 0.0,
                    searches=1 if arm == "B" else 0, search_hit=(qid != "q2") if arm == "B" else None, via="text" if arm == "B" else None,
                    judge_flags=("retry",) if (qid, arm, repeat) == ("q1", "A", 2) else ("unparsed",) if (qid, arm, repeat) == ("q2", "B", 1) else (),
                    prompt_tokens=1000 + 100 * repeat, ttft_ms=300.0 + 50 * repeat))
    out.append(result_row("q5", "multi-session", "A", 3, None, judge_flags=("error",)))  # a reply the judge never answered
    out.append(result_row("q5", "multi-session", "B", 3, False, state="failed"))  # a turn that did not finish
    return out


def ingest(turns, cut, **kw):
    return {"sessions": 5, "turns": turns, "docs": turns, "notes": 7, "embedded_chars": 1000, "cut": cut, "skipped": 0, "trimmed": 0,
            "seconds": 3.0, "model": "embed-model", "error": "", "ids": {}, **kw}


def controls():
    def row(arm, qid, used, tokens, ttft, state="finished"):
        return {"question_id": qid, "arm": arm, "index": 0, "prompt": "p", "state": state, "used": [{}] * used, "false_recall": used > 0,
                "prompt_tokens": tokens, "ttft_ms": ttft}
    return [row("A", "q1", 2, 100, 200.0), row("A", "q2", 0, 120, 250.0), row("A", "q3", 1, 140, 300.0), row("A", "q4", 0, None, None, "failed"),
            row("B", "q1", 0, 90, 180.0), row("B", "q2", 0, 95, 190.0)]


@pytest.fixture
def run_dir(tmp_path):
    write_run(tmp_path / "run", rows(), arms=("A", "B"), questions=QUESTIONS, repeats=2, controls=controls(),
              ingests={"q1": ingest(100, 2), "q3": ingest(100, 30), "q4": ingest(50, 10, error="embedding failed"), "q5": ingest(200, 4)},
              extra_bodies={"B": {"temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}}})
    return tmp_path / "run"


def test_bootstrap_interval():
    assert report.bootstrap_ci([]) is None
    assert report.bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    lo, hi = report.bootstrap_ci([0.0, 1.0] * 50)
    assert 0.38 < lo < 0.5 < hi < 0.62
    assert report.bootstrap_ci([0.0, 1.0] * 50) == (lo, hi)  # seed 0, the same every time
    spread = [0.1 * n for n in range(20)]
    assert report.bootstrap_ci(spread, 50, seed=0) == report.bootstrap_ci(spread, 50, seed=0) != report.bootstrap_ci(spread, 50, seed=1)
    assert report.bootstrap_ci([0.0, 1.0]) == (0.0, 1.0)
    assert (report.BOOTSTRAP, report.SEED) == (10000, 0)


def test_accuracy_per_type_with_intervals_and_repeat_spread(run_dir):
    rep = report.analyse(run_dir, resamples=500)
    assert rep["arms"] == ["A", "B"] and rep["questions"] == 6
    acc = rep["accuracy"]
    user_a, user_b = acc["single-session-user"]["A"], acc["single-session-user"]["B"]
    assert (user_a["questions"], user_a["cells"], user_a["right"], user_a["accuracy"]) == (2, 4, 3, 0.75)
    assert user_a["repeats"] == {1: 1.0, 2: 0.5} and user_a["spread"] == [0.5, 1.0] and user_a["ci"] == [0.5, 1.0]
    assert user_b["accuracy"] == 0.25 and acc["single-session-assistant"]["A"]["accuracy"] == 0.75
    assert acc["abstention"]["A"]["accuracy"] == 0.5 and acc["abstention"]["B"]["accuracy"] == 1.0
    assert acc["abstention"]["A"]["questions"] == 1 and acc["single-session-preference"]["A"]["questions"] == 0
    assert acc["single-session-preference"]["A"]["accuracy"] is None and acc["single-session-preference"]["A"]["ci"] is None
    total = acc["all"]["A"]
    assert total["questions"] == 6 and total["accuracy"] == pytest.approx(sum([1.0, 0.5, 0.5, 1.0, 1.0, 0.5]) / 6)
    assert total["ci"][0] <= total["accuracy"] <= total["ci"][1]
    # a reply the judge never answered is out (A's third q5 row); a turn that did not finish is a wrong answer (B's)
    assert acc["multi-session"]["A"]["cells"] == 2 and acc["multi-session"]["B"]["cells"] == 3 and acc["multi-session"]["B"]["right"] == 2
    assert acc["multi-session"]["B"]["accuracy"] == pytest.approx(2 / 3)


def test_the_interval_is_the_same_every_time_and_default_is_10000(run_dir):
    one = report.analyse(run_dir, resamples=300)
    two = report.analyse(run_dir, resamples=300)
    assert one == two
    assert report.analyse(run_dir, resamples=100, seed=1)["accuracy"] != one["accuracy"]  # the seed is the interval's
    defaults = inspect.signature(report.analyse).parameters
    assert (defaults["resamples"].default, defaults["seed"].default) == (10000, 0)


def test_paired_differences(run_dir):
    paired = {(p["a"], p["b"], p["group"]): p for p in report.analyse(run_dir, resamples=500)["paired"]}
    assert len(paired) == 8 and {k[:2] for k in paired} == {("A", "B")}  # one pair of arms, a row per group
    user = paired[("A", "B", "single-session-user")]
    assert (user["questions"], user["diff"], user["ci"], user["a_better"], user["b_better"]) == (2, 0.5, [0.5, 0.5], 2, 0)
    total = paired[("A", "B", "all")]
    assert total["questions"] == 6 and total["diff"] == pytest.approx((4 * 0.5 + (1 - 2 / 3) - 0.5) / 6)
    assert total["ci"][0] <= total["diff"] <= total["ci"][1] and (total["a_better"], total["b_better"]) == (5, 1)
    assert paired[("A", "B", "abstention")]["diff"] == -0.5 and paired[("A", "B", "abstention")]["b_better"] == 1
    empty = paired[("A", "B", "single-session-preference")]
    assert empty["questions"] == 0 and empty["diff"] is None and empty["ci"] is None


def test_retrieval_apart_for_auto_and_search(run_dir):
    table = report.analyse(run_dir, resamples=100)["retrieval"]
    user = table["single-session-user"]
    assert user["A"]["turns"] == 4 and (user["A"]["auto_hit"]["k"], user["A"]["auto_hit"]["n"]) == (4, 4)
    assert user["A"]["searched"]["k"] == 0 and user["A"]["search_hit"]["n"] == 0 and user["A"]["search_hit"]["p"] is None
    assert (user["B"]["auto_hit"]["k"], user["B"]["searched"]["k"], user["B"]["searched"]["n"]) == (0, 4, 4)
    assert (user["B"]["search_hit"]["k"], user["B"]["search_hit"]["n"]) == (2, 4)  # q2 is never found by the search
    assert (user["B"]["either"]["k"], user["B"]["via_text"], user["B"]["via_items"]) == (2, 4, 0)
    assistant = table["single-session-assistant"]["A"]
    assert (assistant["auto_hit"]["k"], assistant["auto_hit"]["n"], assistant["auto_all"]["k"]) == (3, 4, 3)
    ci = user["B"]["search_hit"]
    assert ci["lo"] < 0.5 < ci["hi"]  # a Wilson interval, not a bootstrap
    assert table["all"]["A"]["turns"] == 13


def test_ingest_cut_rate_with_the_assistant_type_on_its_own(run_dir):
    ing = report.analyse(run_dir, resamples=100)["ingest"]
    assistant = ing["single-session-assistant"]
    assert (assistant["ingests"], assistant["turns"], assistant["cut"]) == (2, 150, 40) and assistant["cut_rate"]["p"] == pytest.approx(40 / 150)
    assert assistant["errors"] == 1 and assistant["notes"]["median"] == 7 and assistant["seconds"]["median"] == 3.0
    rest = ing["all but single-session-assistant"]
    assert (rest["ingests"], rest["turns"], rest["cut"]) == (2, 300, 6) and rest["cut_rate"]["p"] == pytest.approx(0.02)
    assert ing["all"]["turns"] == 450 and ing["single-session-user"]["ingests"] == 1 and ing["temporal-reasoning"]["ingests"] == 0


def test_judge_retry_unparsed_and_control(run_dir):
    rep = report.analyse(run_dir, resamples=100)
    a, b = rep["judge"]["A"], rep["judge"]["B"]
    assert (a["judge_retry"], a["judge_unparsed"], a["judge_error"], a["judged"], a["rows"]) == (1, 0, 1, 13, 13)
    assert (b["judge_retry"], b["judge_unparsed"], b["not_judged"], b["rows"]) == (1, 1, 1, 13)
    ct = rep["control"]
    assert (ct["A"]["cells"], ct["A"]["failed"]) == (4, 1) and (ct["A"]["false_recall"]["k"], ct["A"]["false_recall"]["n"]) == (2, 3)
    assert ct["A"]["prompt_tokens"]["median"] == 120 and ct["A"]["ttft_ms"]["median"] == 250.0 and ct["A"]["items_used"]["median"] == 1.0
    assert ct["B"]["false_recall"]["k"] == 0 and ct["B"]["false_recall"]["n"] == 2
    assert ct["A"]["question_prompt_tokens"]["median"] == 1150.0 and ct["A"]["question_ttft_ms"]["median"] == 375.0


def test_replies_asked_again_per_arm(tmp_path):
    def row(arm, repeat, notices, retries, final=False):
        r = result_row("q1", "single-session-user", arm, repeat, True)
        if notices is not None:
            r.update(notices=notices, reply_retries=retries, final_text_tool_call=final)
        return r
    rows = [row("A", 1, {"tool_call_as_text": 2, "retry": 1}, 1, final=True), row("A", 2, {"empty_reply": 1}, 1),
            row("A", 3, None, 0), row("B", 1, {}, 0)]
    write_run(tmp_path / "run", rows, arms=("A", "B"), questions=QUESTIONS[:1], repeats=3)
    rep = report.analyse(tmp_path / "run", resamples=10)
    assert rep["replies"]["A"] == {
        "rows": 3, "not_recorded": 1, "turns_asked_again": 2, "asked_again": 2, "final_text_tool_call": 1, "request_retry": 1,
        "empty_reply": 1, "unfinished_answer": 0, "tool_call_as_text": 2}
    assert rep["replies"]["B"]["turns_asked_again"] == 0 and rep["replies"]["B"]["not_recorded"] == 0
    text = report.render(rep)
    assert "## Replies asked again" in text and "| turns asked again | 2 / 2 | 0 / 1 |" in text
    assert "| turn ended on a text tool call | 1 | 0 |" in text and "| rows not recorded | 1 | 0 |" in text


def test_opens_per_arm_and_on_the_controls(tmp_path):
    rows = [result_row("q1", "single-session-user", "A", r, True) for r in (1, 2)]
    rows[0]["retrieval"].update(opens=2, open_hit=True)
    rows[1]["retrieval"].update(opens=0, open_hit=None)
    controls = [{"question_id": "q1", "arm": "A", "index": 0, "prompt": "p", "state": "finished", "used": [], "false_recall": False,
                 "opens": 1, "prompt_tokens": 100, "ttft_ms": 200.0}]
    write_run(tmp_path / "run", rows, arms=("A",), questions=QUESTIONS[:1], repeats=2, controls=controls)
    rep = report.analyse(tmp_path / "run", resamples=10)
    cell = rep["retrieval"]["all"]["A"]
    assert (cell["opened"]["k"], cell["opened"]["n"], cell["open_hit"]["k"], cell["open_hit"]["n"]) == (1, 2, 1, 1)
    assert (rep["control"]["A"]["opened"]["k"], rep["control"]["A"]["opened"]["n"]) == (1, 1)
    text = report.render(rep)
    assert "turns that opened a listed turn" in text and "control prompts that opened a listed turn" in text


def test_the_rendered_report(run_dir):
    text = report.render(report.analyse(run_dir, resamples=200))
    for heading in ("## Arms", "## Accuracy", "## Paired arm differences", "## Retrieval", "## Ingest", "## Judge", "## Control: ordinary prompts"):
        assert heading in text
    assert "Sampling: temperature 0.3; thinking off" in text and "Sampling: the server defaults for the model (no sampling fields sent)" in text
    assert "| **single-session-assistant** | 2 | 150 | 40 |" in text and "all but single-session-assistant" in text
    assert "| single-session-user | 75.0 [50.0, 100.0] · r 50.0-100.0 (n=2) |" in text
    assert "| A - B | all | 6 |" in text and "judge_retry" in text and "judge_unparsed" in text
    assert "localmelo `0123456789`" in text and "BuildVersion: TEST" in " ".join(text.split())
    assert "200 resamples" in text
    assert "> 1 replies were not judged" in text and "> The embedding model failed during some ingests" in text


def test_the_command_line_writes_the_report_files(run_dir, capsys):
    assert report.main([str(run_dir), "--resamples", "100"]) == 0
    doc = json.loads((run_dir / "report.json").read_text())
    assert doc["resamples"] == 100 and doc["accuracy"]["all"]["A"]["questions"] == 6
    assert (run_dir / "report.md").read_text().startswith("# LongMemEval-S: run")
    assert "report.md" in capsys.readouterr().out


def test_a_report_on_a_run_that_has_not_started(tmp_path):
    write_run(tmp_path / "empty", [], arms=("A",), questions=QUESTIONS, repeats=1)
    rep = report.analyse(tmp_path / "empty", resamples=50)
    assert rep["arms"] == [] and "No cell has run yet" in report.render(rep)


def test_the_report_on_a_run_of_the_driver(tmp_path):
    """Two questions, one arm, through the fake ingest, gateway and judge: the report reads what the driver wrote."""
    repo = make_repo(tmp_path)
    models = judge_server()
    try:
        study = make_public_study(tmp_path, repo, models, PublicLauncher())
        assert run.run_study(study) == 0
    finally:
        models.stop()
    rep = report.analyse(study.run_dir, resamples=100)
    assert rep["arms"] == ["A"] and rep["accuracy"]["all"]["A"]["accuracy"] == 1.0 and rep["accuracy"]["all"]["A"]["ci"] == [1.0, 1.0]
    assert rep["accuracy"]["abstention"]["A"]["questions"] == 1 and rep["accuracy"]["single-session-user"]["A"]["questions"] == 1
    assert rep["accuracy"]["single-session-user"]["A"]["spread"] is None  # one repeat
    # q1's evidence was recalled, q2's (an abstention question) was not
    assert rep["retrieval"]["single-session-user"]["A"]["auto_hit"]["k"] == 1 and rep["retrieval"]["abstention"]["A"]["auto_hit"]["k"] == 0
    assert rep["ingest"]["all"]["ingests"] == 2 and rep["ingest"]["all"]["turns"] == 5 and rep["ingest"]["all"]["errors"] == 0
    assert rep["judge"]["A"] == {"rows": 2, "judged": 2, "not_judged": 0, "judge_retry": 0, "judge_unparsed": 0, "judge_error": 0}
    assert rep["paired"] == [] or all(p["a"] != p["b"] for p in rep["paired"])
    text = report.render(rep)
    assert "ProductName: test" in text and "Sampling: the server defaults" in text and "## Paired arm differences" not in text
    assert "Business" not in text


def test_a_wrong_answer_is_put_at_the_step_that_lost_the_evidence():
    base = {"arm": "A", "state": "finished", "correct": False, "abstention": False}
    assert report.miss_step({**base, "correct": True}) is None
    assert report.miss_step({**base, "state": "timed_out"}) == "turn failed"
    assert report.miss_step({**base, "abstention": True}) == "abstention: answered anyway"
    none = {"auto_hit": False, "search_hit": None}
    assert report.miss_step({**base, "retrieval": none, "evidence": {"best": 0.4}}) == "2 evidence cut at ingest, not retrieved"
    assert report.miss_step({**base, "retrieval": none, "evidence": {"best": 1.0}}) == "3 not retrieved"
    assert report.miss_step({**base, "retrieval": {"auto_hit": False, "search_hit": True}, "evidence": {"best": 0.9}}) == \
        "4 retrieved, evidence cut at ingest"
    assert report.miss_step({**base, "retrieval": {"auto_hit": True}, "evidence": {"best": None}}) == \
        "5 retrieved, evidence not checked"
    assert report.miss_step({**base, "retrieval": {"auto_hit": True, "evidence_seen": 1.0, "evidence_all": 0.3}, "evidence": {"best": 1.0}}) == \
        "5a session retrieved, evidence not all shown"
    assert report.miss_step({**base, "retrieval": {"search_hit": True, "evidence_all": 0.9}, "evidence": {"best": 1.0}}) == \
        "5b evidence shown, not used by the reader"
    table = report.miss_table([{**base, "retrieval": none, "evidence": {}}, {**base, "correct": True}], ["A"])
    assert table["A"]["3 not retrieved"] == 1 and sum(table["A"].values()) == 1
