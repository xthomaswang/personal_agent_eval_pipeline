"""The re-grading tool: shared verdicts for identical answers, resume, the pilot limit, the report's counts, skipped runs."""
import contextlib
import json
import os
import re
import signal
import threading
import time

import pytest

from memory.public import data, rejudge

TYPES = ("single-session-user", "knowledge-update", "temporal-reasoning")
QUESTION_IDS = ("q1", "q2", "q3_abs", "q4", "q5", "q6")
REPLY = re.compile(r"Model Response: (.*?)\n\n(?:Is|Does) the model")


def make_dataset(tmp_path):
    """A tiny LongMemEval-shaped file of six questions; q3_abs is an abstention question."""
    questions = [
        {
            "question_id": qid, "question_type": TYPES[i % 3], "question": f"Question {qid}?", "answer": f"gold {qid}",
            "question_date": "2023/05/30 (Tue) 23:40", "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [[{"role": "user", "content": "hi"}]], "answer_session_ids": ["s1"],
        }
        for i, qid in enumerate(QUESTION_IDS)
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    return path, {q["question_id"]: q["question_type"] for q in questions}


def write_run(root, name, cells, dataset, types, *, sha=None, questions=True, public=True, state="finished"):
    """A run directory with results.jsonl; ``cells`` is [(question id, arm, repeat, answer, old verdict)]."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    rows = []
    for qid, arm, repeat, answer, old in cells:
        rows.append({
            "question_id": qid, "question_type": types[qid], "abstention": qid.endswith("_abs"), "arm": arm, "repeat": repeat,
            "state": state, "status": "completed", "answer": answer, "correct": old,
            "judge": {"label": old, "model": "chat-model", "error": ""}})
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if public:
        sha = sha or data.file_sha256(dataset)
        (run_dir / "public.json").write_text(json.dumps({"config": {"data": {"name": dataset.name, "sha256": sha}}, "complete": True}))
    if questions:
        ids = sorted({c[0] for c in cells})
        (run_dir / "questions.json").write_text(json.dumps([{"question_id": q, "question_type": types[q]} for q in ids]))
    return run_dir


class FakeJudge:
    """The ``post`` of ``judge.judge`` with no network: replies "yes" to an answer that ends with "yes", "no" otherwise."""

    def __init__(self, fail=(), on_call=None):
        self.calls = []
        self.fail = set(fail)  # answers whose call raises OSError
        self.on_call = on_call
        self._lock = threading.Lock()

    def __call__(self, url, body, timeout):
        prompt = body["messages"][0]["content"]
        answer = REPLY.search(prompt).group(1)
        with self._lock:
            self.calls.append(answer)
            number = len(self.calls)
        if self.on_call:
            self.on_call(number)
        if answer in self.fail:
            raise OSError("judge is down")
        return 1.0, {"choices": [{"message": {"content": "yes" if answer.endswith("yes") else "no"}}]}


class Clock:
    """A clock the test sets by hand (seconds since the epoch)."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __call__(self):
        return self.seconds


def cli(tmp_path, post, *extra, out="out", runs_root="runs", data_path=None, now=None):
    """Run the tool on tmp_path's runs; returns (exit code, the lines it said)."""
    said = []
    argv = [
        "--runs-root", str(tmp_path / runs_root), "--data", str(data_path or tmp_path / "lme.json"), "--out", str(tmp_path / out),
        "--base-url", "http://judge.invalid/v1", "--judge-model", "new-judge", "--workers", "1", *extra]
    code = rejudge.main(argv, post=post, say=said.append, **({"now": now} if now else {}))
    return code, said


@pytest.fixture
def world(tmp_path):
    """Run A: two arms, six answers, five distinct (arm B repeats arm A's answer to q1); the new judge says yes to answers ending "yes"."""
    dataset, types = make_dataset(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    write_run(root, "runA", [
        ("q1", "A", 1, "alpha yes", True), ("q2", "A", 1, "beta no", True), ("q3_abs", "A", 1, "unknown no", False),
        ("q1", "B", 1, "alpha yes", True), ("q2", "B", 1, "gamma yes", False), ("q3_abs", "B", 1, "other no", True),
    ], dataset, types)
    return tmp_path, dataset, types


def counts(tally):
    """(answers, old correct, new correct, yes to no, no to yes) of one tally of the report."""
    return tally["n"], tally["old_correct"], tally["new_correct"], tally["yes_to_no"], tally["no_to_yes"]


def verdict_lines(tmp_path):
    return [json.loads(line) for line in (tmp_path / "out" / rejudge.VERDICTS_FILE).read_text().splitlines()]


def test_identical_answers_to_one_question_are_judged_once(world):
    tmp_path, dataset, _ = world
    collection = rejudge.collect(rejudge.find_runs(tmp_path / "runs"), rejudge.load_dataset(dataset))
    assert (collection.cells, len(collection.tasks), collection.saved) == (6, 5, 1)
    post = FakeJudge()
    code, said = cli(tmp_path, post)
    assert code == 0
    assert sorted(post.calls) == ["alpha yes", "beta no", "gamma yes", "other no", "unknown no"]  # "alpha yes" asked once
    assert len(verdict_lines(tmp_path)) == 5
    assert any("judge calls saved by sharing: 1" in line for line in said)
    # the same answer to another question is another key
    other = rejudge.answer_key("q2", "alpha yes", False)
    assert other not in collection.tasks and rejudge.answer_key("q1", "alpha yes", False) in collection.tasks


def test_resume_skips_settled_keys_and_asks_errored_ones_again(world):
    tmp_path, _, _ = world
    first = FakeJudge(fail={"gamma yes"})
    code, said = cli(tmp_path, first)
    assert code == 1 and len(first.calls) == 5
    assert any("1 answers got no reply" in line for line in said)
    assert not (tmp_path / "out" / rejudge.REPORT_JSON).exists()  # not every key is settled
    lines = verdict_lines(tmp_path)
    assert [bool(v["judge"]["error"]) for v in lines].count(True) == 1
    second = FakeJudge()
    code, _ = cli(tmp_path, second)
    assert code == 0 and second.calls == ["gamma yes"]  # only the errored key is asked again
    assert len(verdict_lines(tmp_path)) == 6
    assert (tmp_path / "out" / rejudge.REPORT_JSON).exists()
    third = FakeJudge()
    code, _ = cli(tmp_path, third)
    assert code == 0 and third.calls == []
    assert len(verdict_lines(tmp_path)) == 6


def test_a_torn_last_line_is_ignored_and_does_not_swallow_the_next_line(world):
    tmp_path, _, _ = world
    (tmp_path / "out").mkdir()
    key = rejudge.answer_key("q1", "alpha yes", False)
    judged = {"label": True, "error": "", "model": "new-judge"}
    settled = {"question_id": "q1", "answer_sha256": key[1], "abstention": False, "judge": judged}
    (tmp_path / "out" / rejudge.VERDICTS_FILE).write_text(json.dumps(settled) + '\n{"question_id": "q2", "answer_sh')
    post = FakeJudge()
    code, _ = cli(tmp_path, post)
    assert code == 0 and "alpha yes" not in post.calls and len(post.calls) == 4
    lines = (tmp_path / "out" / rejudge.VERDICTS_FILE).read_text().splitlines()
    whole = []
    for line in lines:
        with contextlib.suppress(ValueError):  # the torn line
            whole.append(json.loads(line))
    assert len(lines) == 6 and len(whole) == 5  # the settled line, the torn one, and four new lines, each on a line of its own


def test_limit_judges_only_the_first_keys_and_writes_no_report(world):
    tmp_path, _, _ = world
    post = FakeJudge()
    code, said = cli(tmp_path, post, "--limit", "2")
    assert code == 0 and post.calls == ["alpha yes", "beta no"]  # the first two keys, in the order the rows come
    assert not (tmp_path / "out" / rejudge.REPORT_JSON).exists()
    assert any("pilot done: 2 of 5" in line for line in said)
    again = FakeJudge()
    cli(tmp_path, again, "--limit", "2")
    assert again.calls == []  # the pilot's keys are settled
    code, _ = cli(tmp_path, FakeJudge(), "--report-only")  # a partial report is possible on request
    partial = json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())
    assert code == 0 and partial["cells_compared"] == 3 and partial["cells_without_new_verdict"] == 3  # alpha yes twice, beta no
    rest = FakeJudge()
    code, _ = cli(tmp_path, rest)
    assert code == 0 and sorted(rest.calls) == ["gamma yes", "other no", "unknown no"]


def test_dry_run_counts_and_calls_no_judge(world):
    tmp_path, _, _ = world

    def forbidden(url, body, timeout):
        raise AssertionError("a dry run must not call the judge")

    code, said = cli(tmp_path, forbidden, "--dry-run")
    assert code == 0 and not (tmp_path / "out").exists()
    text = "\n".join(said)
    assert "1 usable" in text and "answers to grade: 6; distinct (question, answer) keys: 5" in text and "to judge now: 5" in text


def test_report_counts_and_flips_of_a_run_with_two_arms(world):
    tmp_path, _, _ = world
    assert cli(tmp_path, FakeJudge())[0] == 0
    report = json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())
    arms = report["runs"]["runA"]["arms"]
    # arm A: old right on q1 and q2; the new judge says yes only to "alpha yes"
    assert counts(arms["A"]) == (3, 2, 1, 1, 0)
    # arm B: old right on q1 and q3_abs; new right on q1 and q2
    assert counts(arms["B"]) == (3, 2, 2, 1, 1)
    assert arms["A"]["old_judge_models"] == ["chat-model"]
    overall = report["overall"]
    assert (overall["n"], overall["agree"], overall["yes_to_no"], overall["no_to_yes"]) == (6, 3, 2, 1)
    assert overall["agreement"] == pytest.approx(0.5) and overall["old_correct"] == 4 and overall["new_correct"] == 3
    by_type = report["by_question_type"]
    assert (by_type["single-session-user"]["n"], by_type["single-session-user"]["yes_to_no"]) == (2, 0)  # q1 twice
    assert (by_type["knowledge-update"]["n"], by_type["knowledge-update"]["yes_to_no"], by_type["knowledge-update"]["no_to_yes"]) == (
        2, 1, 1)
    assert (by_type["temporal-reasoning"]["n"], by_type["temporal-reasoning"]["yes_to_no"]) == (2, 1)  # q3_abs
    assert report["abstention_questions"]["n"] == 2
    assert (report["cells"], report["distinct_keys"], report["calls_saved"]) == (6, 5, 1)
    assert report["new_judge"]["keys_judged"] == 5 and report["new_judge"]["retry"] == report["new_judge"]["unparsed"] == 0
    markdown = (tmp_path / "out" / rejudge.REPORT_MD).read_text()
    assert "### runA" in markdown and "## Flips by question type" in markdown and "`new-judge`" in markdown
    assert str(tmp_path) not in markdown and "/Users" not in markdown and str(tmp_path) not in json.dumps(report)
    assert "| A | 3 | 2 | 1 | 1 | 0 |" in markdown


def test_new_judge_retries_and_unparsed_replies_are_counted(world):
    tmp_path, _, _ = world

    def vague(url, body, timeout):
        answer = REPLY.search(body["messages"][0]["content"]).group(1)
        reply = {"alpha yes": "maybe", "beta no": "who knows"}.get(answer, "yes")
        if answer == "alpha yes" and "temperature" not in body:
            reply = "yes"  # the second ask is answered
        return 1.0, {"choices": [{"message": {"content": reply}}]}

    assert cli(tmp_path, vague)[0] == 0
    new = json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())["new_judge"]
    assert (new["keys_judged"], new["retry"], new["unparsed"]) == (5, 2, 1)


def test_runs_that_cannot_be_matched_are_skipped_and_named(world):
    tmp_path, dataset, types = world
    root = tmp_path / "runs"
    cells = [("q1", "A", 1, "never judged yes", True)]
    write_run(root, "noquestions", cells, dataset, types, questions=False)
    write_run(root, "otherfile", cells, dataset, types, sha="0" * 64)
    write_run(root, "nopublic", cells, dataset, types, public=False)
    write_run(root, "unknownquestion", [("zzz", "A", 1, "x", True)], dataset, {**types, "zzz": "multi-session"})
    write_run(root, "wrongtype", [("q1", "A", 1, "x", True)], dataset, {**types, "q1": "multi-session"})
    post = FakeJudge()
    code, said = cli(tmp_path, post)
    assert code == 0 and "never judged yes" not in post.calls and "x" not in post.calls
    report = json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())
    reasons = {s["run"]: s["reason"] for s in report["runs_skipped"]}
    assert set(reasons) == {"noquestions", "otherfile", "nopublic", "unknownquestion", "wrongtype"}
    assert reasons["noquestions"] == "no questions.json"
    assert "another question file" in reasons["otherfile"] and "no public.json" in reasons["nopublic"]
    assert "zzz" in reasons["unknownquestion"] and "another question type" in reasons["wrongtype"]
    assert (report["runs_found"], report["runs_used"]) == (6, 1)
    assert any("left out noquestions: no questions.json" in line for line in said)
    markdown = (tmp_path / "out" / rejudge.REPORT_MD).read_text()
    assert "## Runs left out" in markdown and "| noquestions | no questions.json |" in markdown


def test_runs_one_level_deeper_are_found_and_the_runs_option_picks_them(world):
    tmp_path, dataset, types = world
    root = tmp_path / "runs"
    write_run(root / "group", "part1", [("q4", "A", 1, "deep yes", True)], dataset, types)
    write_run(root / "runA", "work", [("q4", "A", 1, "internal yes", True)], dataset, types)  # a directory inside a run is not a run
    assert list(rejudge.find_runs(root)) == ["group/part1", "runA"]
    post = FakeJudge()
    code, _ = cli(tmp_path, post, "--runs", "group")
    assert code == 0 and post.calls == ["deep yes"]
    assert list(json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())["runs"]) == ["group/part1"]
    code, said = cli(tmp_path, post, "--runs", "nothing-here", out="out2")
    assert code == 2 and "no run named nothing-here" in said[0]


def test_rows_without_an_answer_or_not_finished_are_counted_and_left_out(world):
    tmp_path, dataset, types = world
    root = tmp_path / "runs"
    write_run(root, "failed", [("q1", "A", 1, "from a failed turn yes", False)], dataset, types, state="failed")
    write_run(root, "empty", [("q1", "A", 1, "  ", False), ("q2", "A", 1, "real yes", True)], dataset, types)
    post = FakeJudge()
    code, _ = cli(tmp_path, post, "--runs", "failed", "empty")
    report = json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())
    assert code == 0 and post.calls == ["real yes"]
    assert (report["rows_unfinished"], report["rows_no_answer"], report["cells"]) == (1, 1, 1)


def sign_change_world(tmp_path):
    """Run S with six arms of five answers each; ``(old correct, new correct)`` per arm sets every pair's difference."""
    dataset, types = make_dataset(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    counts = {"X": (5, 1), "Y": (3, 3), "Z": (3, 3), "P": (5, 4), "D": (5, 5), "E": (1, 4)}
    cells = []
    for arm, (old, new) in counts.items():
        for i, qid in enumerate(("q1", "q2", "q4", "q5", "q6")):
            cells.append((qid, arm, 1, f"{arm}{qid} {'yes' if i < new else 'no'}", i < old))
    write_run(root, "runS", cells, dataset, types)
    return counts


def test_pairs_whose_sign_changed_or_whose_difference_moved_are_listed_first(tmp_path):
    sign_change_world(tmp_path)
    assert cli(tmp_path, FakeJudge())[0] == 0
    run = json.loads((tmp_path / "out" / rejudge.REPORT_JSON).read_text())["runs"]["runS"]
    assert {a: (t["old_correct"], t["new_correct"]) for a, t in run["arms"].items()} == {
        "X": (5, 1), "Y": (3, 3), "Z": (3, 3), "P": (5, 4), "D": (5, 5), "E": (1, 4)}
    pairs = {(p["a"], p["b"]): p for p in run["pairs"]}
    assert len(pairs) == 15
    xy = pairs[("X", "Y")]  # +2 under the old judge, -2 under the new one: reversed
    assert (xy["old_difference"], xy["new_difference"], xy["move"], xy["sign_changed"], xy["flagged"]) == (2, -2, -4, True, True)
    xp = pairs[("P", "X")]  # P and X tied under the old judge (5 and 5), P ahead by 3 under the new one: a tie broken
    assert (xp["old_difference"], xp["new_difference"], xp["sign_changed"], xp["flagged"]) == (0, 3, True, True)
    de = pairs[("D", "E")]  # +4 and +1: the same sign, but moved by 3
    assert (de["old_difference"], de["new_difference"], de["sign_changed"], de["flagged"]) == (4, 1, False, True)
    yz = pairs[("Y", "Z")]  # tied both times
    assert (yz["old_difference"], yz["new_difference"], yz["sign_changed"], yz["flagged"]) == (0, 0, False, False)
    py = pairs[("P", "Y")]  # +2 and +1: same sign, moved by 1
    assert (py["sign_changed"], py["flagged"]) == (False, False)
    flags = [p["flagged"] for p in run["pairs"]]
    assert flags == sorted(flags, reverse=True)  # every flagged pair before every other pair
    assert (run["pairs"][0]["a"], run["pairs"][0]["b"], run["pairs"][0]["move"]) == ("E", "X", 7)  # +4 became -3 ... reversed, moved by 7
    assert [(p["a"], p["b"]) for p in run["pairs"][1:5]] == [("D", "X"), ("E", "P"), ("X", "Y"), ("X", "Z")]  # then the moves of 4, by name
    assert sum(p["flagged"] for p in run["pairs"]) == 10
    moves = [abs(p["move"]) for p in run["pairs"] if p["flagged"]]
    assert moves == sorted(moves, reverse=True)  # the biggest move first
    markdown = (tmp_path / "out" / rejudge.REPORT_MD).read_text()
    assert "| X vs Y | +2 | -2 | -4 | yes |" in markdown and "| D vs E | +4 | +1 | -3 | no |" in markdown
    assert "| D vs P | 0 | +1 | +1 | yes |" in markdown  # a tie broken is a sign change; zero is written plain
    assert "Y vs Z" not in markdown and "other pairs kept their sign" in markdown


def test_a_pair_of_arms_with_different_answer_counts_says_so(world):
    tmp_path, dataset, types = world
    write_run(tmp_path / "runs", "runU", [
        ("q1", "A", 1, "a yes", False), ("q2", "A", 1, "b yes", False), ("q1", "B", 1, "c no", True)], dataset, types)
    assert cli(tmp_path, FakeJudge(), "--runs", "runU")[0] == 0
    markdown = (tmp_path / "out" / rejudge.REPORT_MD).read_text()
    assert "| A vs B (answers 2 and 1) | -1 | +2 | +3 | yes |" in markdown


def terminate_on_call(monkeypatch, wanted):
    """An ``on_call`` for FakeJudge that sends SIGTERM to the process during call number ``wanted``.

    The signal is sent from a worker thread; the call goes on once the main thread has run the handler."""
    seen = {}
    real = rejudge.stop_on_signals

    def spy(stop):
        seen["stop"] = stop
        return real(stop)

    monkeypatch.setattr(rejudge, "stop_on_signals", spy)

    def on_call(number):
        if number == wanted:
            os.kill(os.getpid(), signal.SIGTERM)
            deadline = time.monotonic() + 5
            while not seen["stop"].is_set() and time.monotonic() < deadline:
                time.sleep(0.01)

    return on_call


def test_a_term_signal_finishes_the_call_in_flight_and_exits_75(world, monkeypatch):
    tmp_path, _, _ = world
    before = signal.getsignal(signal.SIGTERM)
    post = FakeJudge(on_call=terminate_on_call(monkeypatch, 2))
    code, said = cli(tmp_path, post)
    assert code == rejudge.EXIT_STOPPED == 75 and len(post.calls) == 2  # the second call finished, the rest were not started
    assert len(verdict_lines(tmp_path)) == 2
    assert any("stopped on request (signal): 3 of 5 answers still wait" in line for line in said)
    assert signal.getsignal(signal.SIGTERM) is before  # the handler is gone
    assert not (tmp_path / "out" / rejudge.REPORT_JSON).exists()
    rest = FakeJudge()
    code, _ = cli(tmp_path, rest)
    assert code == 0 and len(rest.calls) == 3 and (tmp_path / "out" / rejudge.REPORT_JSON).exists()


def test_a_signal_during_the_last_call_leaves_nothing_open_so_the_exit_is_zero(world, monkeypatch):
    tmp_path, _, _ = world
    post = FakeJudge(on_call=terminate_on_call(monkeypatch, 5))
    code, _ = cli(tmp_path, post)
    assert code == 0 and len(post.calls) == 5 and (tmp_path / "out" / rejudge.REPORT_JSON).exists()


def test_the_night_deadline_stops_new_calls_and_exits_75(world, monkeypatch):
    tmp_path, _, _ = world
    clock = Clock(900.0)
    monkeypatch.setenv("NIGHT_DEADLINE", "1000")
    post = FakeJudge(on_call=lambda number: setattr(clock, "seconds", 1001.0) if number == 2 else None)  # the deadline passes in call 2
    code, said = cli(tmp_path, post, now=clock)
    assert code == 75 and len(post.calls) == 2 and len(verdict_lines(tmp_path)) == 2
    assert any("stopped on request (deadline): 3 of 5 answers still wait" in line for line in said)
    assert not (tmp_path / "out" / rejudge.REPORT_JSON).exists()
    late = FakeJudge()  # a start after the deadline asks nothing
    code, _ = cli(tmp_path, late, now=Clock(2000.0))
    assert code == 75 and late.calls == []
    monkeypatch.delenv("NIGHT_DEADLINE")  # the next night: no deadline in reach, the rest is judged
    rest = FakeJudge()
    code, _ = cli(tmp_path, rest, now=Clock(2000.0))
    assert code == 0 and len(rest.calls) == 3 and (tmp_path / "out" / rejudge.REPORT_JSON).exists()
    monkeypatch.setenv("NIGHT_DEADLINE", "1")  # past, but nothing is left to ask: done, not stopped
    code, _ = cli(tmp_path, FakeJudge(), now=Clock(2000.0))
    assert code == 0


def test_a_deadline_that_is_not_a_number_is_refused(world, monkeypatch):
    tmp_path, _, _ = world
    monkeypatch.setenv("NIGHT_DEADLINE", "tomorrow")
    post = FakeJudge()
    code, said = cli(tmp_path, post)
    assert code == 2 and post.calls == [] and "NIGHT_DEADLINE" in said[0]
    monkeypatch.setenv("NIGHT_DEADLINE", "")  # empty means not set
    assert cli(tmp_path, post)[0] == 0


def test_the_stop_file_stops_new_calls_and_exits_75(world):
    tmp_path, _, _ = world
    stop_file = tmp_path / "stop-now"
    post = FakeJudge(on_call=lambda number: stop_file.write_text("") if number == 2 else None)  # the file appears during call 2
    code, said = cli(tmp_path, post, "--stop-file", str(stop_file))
    assert code == 75 and len(post.calls) == 2
    assert any("stopped on request (stop file): 3 of 5 answers still wait" in line for line in said)
    again = FakeJudge()  # the file is still there: nothing is asked
    code, _ = cli(tmp_path, again, "--stop-file", str(stop_file))
    assert code == 75 and again.calls == []
    assert stop_file.exists()  # the tool never removes it
    stop_file.unlink()
    rest = FakeJudge()
    code, _ = cli(tmp_path, rest, "--stop-file", str(stop_file))
    assert code == 0 and len(rest.calls) == 3 and (tmp_path / "out" / rejudge.REPORT_JSON).exists()


def test_a_judge_that_never_answers_ends_the_pass_early(world, monkeypatch):
    tmp_path, _, _ = world
    monkeypatch.setattr(rejudge, "MAX_DEAD_IN_A_ROW", 2)
    post = FakeJudge(fail={"alpha yes", "beta no", "unknown no", "gamma yes", "other no"})
    code, said = cli(tmp_path, post)
    assert code == 1 and len(post.calls) == 2  # not 75: nobody asked it to stop
    assert any("gave no reply 2 times in a row" in line for line in said)


def test_verdicts_of_another_judge_model_are_not_mixed_in(world):
    tmp_path, _, _ = world
    cli(tmp_path, FakeJudge(), "--limit", "1")
    post = FakeJudge()
    argv = [
        "--runs-root", str(tmp_path / "runs"), "--data", str(tmp_path / "lme.json"), "--out", str(tmp_path / "out"),
        "--base-url", "http://judge.invalid/v1", "--judge-model", "other-judge"]
    said = []
    assert rejudge.main(argv, post=post, say=said.append) == 2
    assert post.calls == [] and any("already holds verdicts of new-judge" in line for line in said)


def test_progress_is_printed_every_so_many_calls(world, monkeypatch):
    tmp_path, _, _ = world
    monkeypatch.setattr(rejudge, "PROGRESS_EVERY", 2)
    code, said = cli(tmp_path, FakeJudge())
    progress = [line for line in said if line.startswith("progress: ")]
    assert code == 0 and len(progress) == 2
    assert re.fullmatch(r"progress: 2/5 done, [\d.]+ per minute, about \d+ minutes left", progress[0])
    assert progress[1].startswith("progress: 4/5 done")


def test_the_arguments_are_checked(world, capsys):
    tmp_path, _, _ = world
    with pytest.raises(SystemExit):
        rejudge.main(["--runs-root", str(tmp_path / "runs"), "--data", str(tmp_path / "lme.json"), "--out", str(tmp_path / "out")])
    assert "--base-url and --judge-model are required" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        rejudge.main(["--runs-root", str(tmp_path / "runs"), "--data", str(tmp_path / "lme.json"), "--dry-run", "--limit", "0"])
    assert "at least 1" in capsys.readouterr().err
