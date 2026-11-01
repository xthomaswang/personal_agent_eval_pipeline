"""Arms of different runs side by side."""
import json

from memory.public import compare
from memory.tests.public_support import result_row, write_run

QUESTIONS = [{"question_id": q, "question_type": "single-session-user"} for q in ("q1", "q2", "q3")]


def two_runs(tmp_path):
    # run one: arm A gets q1 right twice, q2 once, q3 never; run two: arm L gets every repeat right
    first = [result_row(q, "single-session-user", "A", r, ok) for q, oks in (("q1", (1, 1)), ("q2", (1, 0)), ("q3", (0, 0)))
             for r, ok in enumerate(oks, 1)]
    second = [result_row(q, "single-session-user", "L", r, True) for q in ("q1", "q2", "q3") for r in (1, 2)]
    write_run(tmp_path / "tm22", first, arms=("A",), questions=QUESTIONS)
    write_run(tmp_path / "r1", second, arms=("L",), questions=QUESTIONS)
    return tmp_path / "tm22", tmp_path / "r1"


def test_columns_of_two_runs_are_paired_by_question(tmp_path):
    a, b = two_runs(tmp_path)
    columns, problems = compare.load_columns([f"{b}:L", f"{a}:A"])
    assert problems == [] and [c["name"] for c in columns] == ["r1/L", "tm22/A"]
    cmp = compare.compare(columns, resamples=100)
    assert cmp["accuracy"]["all"]["r1/L"]["accuracy"] == 1.0 and abs(cmp["accuracy"]["all"]["tm22/A"]["accuracy"] - 0.5) < 1e-9
    paired = next(p for p in cmp["paired"] if p["group"] == "all")
    assert (paired["a"], paired["b"], paired["questions"], paired["a_better"], paired["b_better"]) == ("r1/L", "tm22/A", 3, 2, 0)
    assert abs(paired["diff"] - 0.5) < 1e-9
    text = compare.render(cmp, note="the builds also differ by console changes")
    assert "| r1/L - tm22/A | all | 3 | +50.0" in text and "console changes" in text and "## Why answers were wrong" in text


def test_runs_of_other_questions_or_a_missing_arm_are_refused(tmp_path, capsys):
    a, b = two_runs(tmp_path)
    public = json.loads((b / "public.json").read_text())
    public["config"]["questions"] = ["q1", "q2"]
    (b / "public.json").write_text(json.dumps(public))
    _, problems = compare.load_columns([f"{a}:A", f"{b}:L", f"{a}:B"])
    assert any("questions differs" in p for p in problems) and any("no arm 'B'" in p for p in problems)
    assert compare.main([f"{a}:A", f"{b}:L"]) == 2
    assert "questions differs" in capsys.readouterr().err
