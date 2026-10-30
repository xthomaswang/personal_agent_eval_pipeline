"""The web study report (web/report.py) on synthetic records: counts, costs, zero-navigation answers, the closed-book
split, and D against S with its sign test."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from web import report

TIER = {1: "easy", 2: "easy", 3: "easy", 4: "easy", 5: "medium", 6: "medium", 7: "medium", 8: "medium"}
QUESTIONS = list(TIER)


class Study:
    """Records for a state file: cells of one harness and arm, filled from a table of verdicts."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.counter = itertools.count(1)

    def add(
        self, harness: str, arm: str, question: int, verdict: str, *, repeat: int = 1, attempt: int = 1, **kw: Any
    ) -> dict[str, Any]:
        index = next(self.counter)
        run_id = f"{index:03d}__q{question}__r{repeat}__{harness}__{arm}" + ("__retry" if attempt > 1 else "")
        record = {
            "state": "interrupted" if verdict == "interrupted" else "finished",
            "index": index, "run_id": run_id, "question": question, "tier": TIER[question], "repeat": repeat,
            "harness": harness, "arm": arm, "attempt": attempt, "verdict": verdict, "passed": verdict == "pass",
            "n_llm": 4, "n_tool": 3, "output_tokens": 100, "uncached_tokens": 1000, "n_images": 0, "wall_s": 60.0,
            "n_nav": None if arm == "C" else 2, "zero_nav": None if arm == "C" else False,
            "answer_key_suspect": None if arm == "C" else False, "search_tool_leak": False, "foreign_tools": [],
            "tap": None,
        }  # fmt: skip
        record.update(kw)
        self.records[run_id] = record
        return record

    def cell(
        self, harness: str, arm: str, verdicts: dict[int, str], extra: dict[int, dict[str, Any]] | None = None
    ) -> None:
        """One harness and arm; ``extra`` overrides fields of single questions' records."""
        for question, verdict in verdicts.items():
            self.add(harness, arm, question, verdict, **((extra or {}).get(question) or {}))

    def write(
        self,
        out: Path,
        *,
        harnesses: tuple[str, ...],
        arms: tuple[str, ...] = ("D", "S", "C"),
        repeats: int = 1,
        complete: bool = True,
    ) -> Path:
        config = {
            "protocol": "web-v1", "harnesses": list(harnesses), "arms": list(arms), "questions": QUESTIONS,
            "repeats": repeats, "model": "stub-model", "dataset_revision": "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
            "chrome_devtools_mcp": "1.11.0", "localmelo_sha": "a9fb42b6633f6da8a887b3a65bdb092f8f2c4605",
        }  # fmt: skip
        out.mkdir(parents=True, exist_ok=True)
        (out / "web.json").write_text(
            json.dumps(
                {"config": config, "chrome_version": "156.0.8078.80", "runs": self.records, "complete": complete}
            )
        )
        return out


def passes(*ids: int, others: str = "fail") -> dict[int, str]:
    return {q: "pass" if q in ids else others for q in QUESTIONS}


@pytest.fixture
def study(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """claude_code on D, S and C, with a blocked run and its retry, and an interrupted run; hermes on D and S."""
    s = Study()
    # C: right on 1, 2 and 7
    s.cell("claude_code", "C", passes(1, 2, 7))
    # D: question 3 was blocked once and passed on its retry; question 7 is a zero-navigation miss
    d = {1: "pass", 2: "pass", 4: "pass", 5: "pass", 6: "pass", 7: "fail", 8: "timeout"}
    s.cell(
        "claude_code", "D", d,
        {1: {"zero_nav": True, "n_nav": 0}, 7: {"zero_nav": True, "n_nav": 0}, 8: {"wall_s": 1800.0}},
    )  # fmt: skip
    s.add("claude_code", "D", 3, "env_blocked", captcha=True)
    s.add("claude_code", "D", 3, "pass", attempt=2, retry_of="x")
    # S: passes 1, 2 and 7's twin; question 8 was interrupted
    s.cell(
        "claude_code",
        "S",
        {1: "pass", 2: "pass", 3: "fail", 4: "fail", 5: "fail", 6: "fail", 7: "fail", 8: "interrupted"},
    )
    # hermes: D passes everything, S passes two
    s.cell("hermes", "D", passes(*QUESTIONS))
    s.cell("hermes", "S", passes(1, 2))
    out = s.write(tmp_path / "study", harnesses=("claude_code", "hermes"))
    return out, report.build_report(out)


# --- the sign test ---


@pytest.mark.parametrize(
    ("wins", "losses", "p"),
    [(0, 0, 1.0), (6, 0, 0.03125), (5, 0, 0.0625), (7, 1, 0.0703125), (5, 5, 1.0), (1, 6, 0.125), (9, 1, 0.021484375)],
)
def test_the_exact_two_sided_sign_test(wins: int, losses: int, p: float) -> None:
    assert report.sign_test(wins, losses) == pytest.approx(p)
    assert report.sign_test(losses, wins) == pytest.approx(p)


def test_the_minimum_detectable_difference() -> None:
    assert report.min_discordant() == 6  # a clean sweep needs six differing questions at 0.05
    assert report.min_margin(5) is None
    assert report.min_margin(6) == 6
    assert report.min_margin(8) == 8
    assert report.min_margin(10) == 8
    assert report.min_margin(24) == 12 and report.min_margin(0) is None
    assert report.mdd(24, 9) == {
        "n_questions": 24, "discordant": 9, "min_discordant": 6, "margin_at_discordant": 7, "margin_all_questions": 12,
    }  # fmt: skip


# --- verdicts by tier, with costs ---


def test_counts_accuracy_and_the_runs_left_out(study: tuple[Path, dict[str, Any]]) -> None:
    _, r = study
    d = r["cells"]["claude_code"]["D"]
    assert d["excluded"] == {"env_blocked": 1}
    assert d["all"]["runs"] == 8  # question 3 counts through its retry only
    assert d["all"]["verdicts"] == {
        "pass": 6,
        "fail": 1,
        "abstain": 0,
        "no_answer": 0,
        "timeout": 1,
        "harness_error": 0,
    }
    assert d["all"]["accuracy"] == 6 / 8
    assert d["tiers"]["easy"]["runs"] == 4 and d["tiers"]["easy"]["accuracy"] == 1.0
    assert d["tiers"]["medium"]["verdicts"]["timeout"] == 1 and d["tiers"]["medium"]["accuracy"] == 2 / 4
    s = r["cells"]["claude_code"]["S"]
    assert s["excluded"] == {"interrupted": 1} and s["all"]["runs"] == 7 and s["all"]["n_pass"] == 2
    assert r["excluded"] == {"env_blocked": 1, "interrupted": 1}
    assert r["runs_recorded"] == 8 + 7 + 8 + 8 + 8 and r["runs_planned"] == 8 * 1 * 2 * 3


def test_costs_are_means_over_the_counted_runs(study: tuple[Path, dict[str, Any]]) -> None:
    _, r = study
    d = r["cells"]["claude_code"]["D"]
    assert d["all"]["costs"]["wall_s"] == pytest.approx((7 * 60 + 1800) / 8)
    assert d["tiers"]["easy"]["costs"]["wall_s"] == 60.0  # the blocked run's wall time is not in it
    assert d["tiers"]["medium"]["costs"]["wall_s"] == pytest.approx((3 * 60 + 1800) / 4)
    assert d["all"]["costs"] == {
        "n_llm": 4, "n_tool": 3, "output_tokens": 100, "uncached_tokens": 1000, "n_images": 0,
        "wall_s": pytest.approx((7 * 60 + 1800) / 8),
    }  # fmt: skip


def test_a_missing_value_is_left_out_of_a_mean_and_an_empty_cell_has_none() -> None:
    rows = [
        {"verdict": "pass", "n_llm": 2, "n_tool": None}, {"verdict": "fail", "n_llm": 4, "n_tool": None},
        {"verdict": "fail", "n_llm": None, "n_tool": None},
    ]  # fmt: skip
    s = report.summarize(rows)
    assert s["costs"]["n_llm"] == 3.0 and s["costs"]["n_tool"] is None
    assert report.summarize([])["accuracy"] is None and report.summarize([])["costs"]["wall_s"] is None


# --- zero-navigation answers ---


def test_zero_navigation_answers_and_their_accuracy(study: tuple[Path, dict[str, Any]]) -> None:
    _, r = study
    z = r["zero_nav"]["claude_code"]["D"]
    assert z["answered"] == 7  # six passes and one fail; the timeout is not an answer
    assert (z["zero_nav"], z["zero_nav_pass"], z["share"], z["zero_nav_accuracy"]) == (2, 1, 2 / 7, 0.5)
    assert (z["other_answered"], z["other_pass"], z["other_accuracy"]) == (5, 5, 1.0)
    assert "C" not in r["zero_nav"]["claude_code"]  # C has no browser
    assert r["zero_nav"]["hermes"]["S"]["zero_nav"] == 0 and r["zero_nav"]["hermes"]["S"]["zero_nav_accuracy"] is None


# --- D and S by whether C was right ---


def test_d_and_s_accuracy_split_by_the_closed_book_answers(study: tuple[Path, dict[str, Any]]) -> None:
    _, r = study
    split = r["closed_book_split"]["claude_code"]["D"]
    assert split["c_right"] == {"questions": 3, "runs": 3, "n_pass": 2, "accuracy": 2 / 3}  # questions 1, 2, 7
    assert split["c_wrong"] == {"questions": 5, "runs": 5, "n_pass": 4, "accuracy": 4 / 5}
    assert r["closed_book_split"]["claude_code"]["S"]["c_right"]["n_pass"] == 2
    assert r["closed_book_split"]["hermes"]["D"] is None  # the study has no C runs for hermes


def test_c_right_in_any_repeat_counts() -> None:
    records = [
        {"harness": "h", "arm": "C", "question": 1, "verdict": "fail"},
        {"harness": "h", "arm": "C", "question": 1, "verdict": "pass"},
        {"harness": "h", "arm": "C", "question": 2, "verdict": "fail"},
        {"harness": "h", "arm": "C", "question": 3, "verdict": "env_blocked"},
        {"harness": "other", "arm": "C", "question": 2, "verdict": "pass"},
    ]
    assert report.c_solved(records, "h") == {1}
    assert report.c_solved(records, "nobody") is None


# --- D against S ---


def test_d_against_s_by_question_with_the_sign_test(study: tuple[Path, dict[str, Any]]) -> None:
    _, r = study
    p = r["paired"]["claude_code"]
    # question 8 has no counted S run, so 7 are paired: D wins on 3, 4, 5, 6; 1, 2 and 7 tie
    assert (p["questions"], p["d_better"], p["s_better"], p["ties"], p["net"]) == (7, 4, 0, 3, 4)
    assert p["p_value"] == pytest.approx(0.125) and p["significant"] is False
    assert p["mdd"] == {
        "n_questions": 7,
        "discordant": 4,
        "min_discordant": 6,
        "margin_at_discordant": None,
        "margin_all_questions": 7,
    }
    assert {x["question"]: x["result"] for x in p["pairs"]} == {
        1: "tie",
        2: "tie",
        3: "D",
        4: "D",
        5: "D",
        6: "D",
        7: "tie",
    }
    third = next(x for x in p["pairs"] if x["question"] == 3)
    assert (third["d"], third["s"], third["tier"]) == ("1/1", "0/1", "easy")  # D's retry is the D side

    h = r["paired"]["hermes"]
    assert (h["questions"], h["d_better"], h["s_better"], h["net"]) == (8, 6, 0, 6)
    assert h["p_value"] == pytest.approx(0.03125) and h["significant"] is True
    assert h["mdd"]["margin_at_discordant"] == 6 and h["mdd"]["margin_all_questions"] == 8


def test_repeats_compare_pass_rates_and_a_win_for_s_counts_against_d() -> None:
    records = [
        {"harness": "h", "arm": "D", "question": 1, "tier": "easy", "verdict": "pass"},
        {"harness": "h", "arm": "D", "question": 1, "tier": "easy", "verdict": "fail"},
        {"harness": "h", "arm": "S", "question": 1, "tier": "easy", "verdict": "pass"},
        {"harness": "h", "arm": "S", "question": 1, "tier": "easy", "verdict": "pass"},  # S 2/2 against D 1/2
        {"harness": "h", "arm": "D", "question": 2, "tier": "easy", "verdict": "fail"},
        {"harness": "h", "arm": "S", "question": 2, "tier": "easy", "verdict": "fail"},
    ]
    p = report.paired_d_vs_s(records, "h")
    assert p is not None
    assert (p["questions"], p["d_better"], p["s_better"], p["ties"]) == (2, 0, 1, 1)
    assert p["pairs"][0] == {"question": 1, "tier": "easy", "d": "1/2", "s": "2/2", "result": "S"}
    assert report.paired_d_vs_s([r for r in records if r["arm"] == "D"], "h") is None


def test_a_study_without_d_or_s_has_no_pairing(tmp_path: Path) -> None:
    s = Study()
    s.cell("hermes", "D", passes(1, 2))
    s.cell("hermes", "C", passes(1))
    r = report.build_report(s.write(tmp_path / "x", harnesses=("hermes",), arms=("D", "C")))
    assert r["paired"] == {} and r["zero_nav"]["hermes"].keys() == {"D"}
    assert "## D against S" not in report.render_markdown(r)


# --- marks, flags ---


def test_marks_show_one_symbol_per_repeat_and_a_retry_stands_in(tmp_path: Path) -> None:
    s = Study()
    s.add("hermes", "D", 1, "pass", repeat=1)
    s.add("hermes", "D", 1, "fail", repeat=2)
    s.add("hermes", "D", 2, "env_blocked", repeat=1)
    s.add("hermes", "D", 2, "pass", repeat=1, attempt=2)
    s.add("hermes", "D", 2, "env_blocked", repeat=2)
    s.add("hermes", "D", 3, "interrupted", repeat=1)
    r = report.build_report(s.write(tmp_path / "x", harnesses=("hermes",), arms=("D",), repeats=2))
    marks = {q: r["marks"][q]["hermes"]["D"] for q in (1, 2, 3, 4)}
    assert marks == {1: "P -", 2: "P x", 3: "x .", 4: ". ."}


def test_flags_list_the_runs_and_tools(tmp_path: Path) -> None:
    s = Study()
    s.add("hermes", "D", 1, "env_blocked", captcha=True)
    s.add("hermes", "D", 2, "pass", answer_key_suspect=True)
    s.add("hermes", "S", 3, "fail", search_tool_leak=True, foreign_tools=["web_search", "memory"])
    s.add("hermes", "S", 4, "fail", foreign_tools=["memory"], tap={"child_failed": True})
    flags = report.build_report(s.write(tmp_path / "x", harnesses=("hermes",), arms=("D", "S")))["flags"]
    assert flags["captcha"] == ["001__q1__r1__hermes__D"]
    assert flags["answer_key_suspect"] == ["002__q2__r1__hermes__D"]
    assert flags["search_tool_leak"] == ["003__q3__r1__hermes__S"]
    assert flags["foreign_tools"] == ["memory", "web_search"]
    assert flags["tap_child_failed"] == ["004__q4__r1__hermes__S"]


# --- Markdown and the command line ---


def test_the_markdown_has_every_section(study: tuple[Path, dict[str, Any]]) -> None:
    _, r = study
    md = report.render_markdown(r)
    for heading in (
        "# Web search suite: claude_code, hermes on arms D, S, C",
        "## Verdicts by tier", "## Zero-navigation answers", "## D and S by closed-book knowledge",
        "## D against S, paired by question", "## Passes per question", "## Flags",
    ):  # fmt: skip
        assert heading in md, heading
    assert "- Model `stub-model`; 8 questions x 1 repeats x 2 harnesses x 3 arms; dataset revision `0dc97e0d28d8`" in md
    assert "- Chrome 156.0.8078.80; chrome-devtools-mcp 1.11.0; localmelo `a9fb42b`" in md
    assert "- Counted runs: 39 of 48 planned\n" in md and "not complete" not in md
    assert "- Left out of the counts: 1 env_blocked, 1 interrupted" in md
    assert "| claude_code | D | all | 8 | 6 | 1 | 0 | 0 | 1 | 0 | 75.0% | 4.0 | 3.0 | 100 | 1,000 | 0.0 | 278 |" in md
    assert "| claude_code | D | medium | 4 | 2 | 1 | 0 | 0 | 1 | 0 | 50.0% |" in md
    assert "| hermes | C | all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |" in md
    assert "| claude_code | D | 7 | 2 | 28.6% | 1/2 (50.0%) | 5/5 (100.0%) |" in md
    assert (
        "| claude_code | D | yes | 3 | 3 | 2 | 66.7% |" in md and "| claude_code | D | no | 5 | 5 | 4 | 80.0% |" in md
    )
    assert "| hermes | D | no C runs |" in md
    assert "| hermes | 8 | 6 | 0 | 2 | +6 | 0.031 * | net 6 of 6 differing questions; if all 8 differed, net 8 |" in md
    assert (
        "| claude_code | 7 | 4 | 0 | 3 | +4 | 0.125 | none: 4 differing questions are fewer than 6; if all 7 differed, net 7 |"
        in md
    )
    assert (
        "| 8 | medium | - | x | - | P | - | . |" in md
    )  # D timed out, S was interrupted, C missed; hermes has no C runs
    assert "- Blocked by a CAPTCHA (left out of the counts): `" in md and "__claude_code__D`" in md
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_the_markdown_marks_a_partial_study(tmp_path: Path) -> None:
    s = Study()
    s.cell("hermes", "D", {1: "pass"})
    md = report.render_markdown(
        report.build_report(s.write(tmp_path / "x", harnesses=("hermes",), arms=("D",), complete=False))
    )
    assert "(the study is not complete)" in md
    assert "| hermes | D | all | 1 | 1 |" in md and "| hermes | D | medium | 0 | 0 |" in md and "n/a" in md


def test_the_command_line_writes_both_files(
    study: tuple[Path, dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    out, _ = study
    assert report.main([str(out)]) == 0
    assert capsys.readouterr().out.strip() == f"{out / 'report.md'}: 39/48 runs counted"
    loaded = json.loads((out / "report.json").read_text())
    assert loaded["paired"]["hermes"]["net"] == 6 and loaded["schema_version"] == 1
    assert (out / "report.md").read_text() == report.render_markdown(report.build_report(out))


def test_a_directory_without_a_study_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert report.main([str(tmp_path)]) == 2
    assert "no web.json" in capsys.readouterr().err


def test_s_cua_is_reported_like_the_other_browsing_arms(tmp_path: Path) -> None:
    s = Study()
    s.cell("localmelo", "C", passes(1))
    s.cell(
        "localmelo",
        "S-cua",
        {1: "pass", 2: "fail", 3: "env_blocked", 4: "env_blocked"},
        {1: {"zero_nav": True, "n_nav": 0}, 3: {"blocked_by": "space"}, 4: {"blocked_by": "captcha", "captcha": True}},
    )
    s.add("localmelo", "S-cua", 3, "pass", attempt=2)  # the retry of the run whose Space left the screen
    r = report.build_report(s.write(tmp_path / "x", harnesses=("localmelo",), arms=("S-cua", "C")))
    cell = r["cells"]["localmelo"]["S-cua"]
    assert cell["all"]["runs"] == 3 and cell["excluded"] == {"env_blocked": 2}  # 1, 2 and the retry of 3 count
    assert r["zero_nav"]["localmelo"]["S-cua"]["zero_nav"] == 1
    assert r["closed_book_split"]["localmelo"]["S-cua"]["c_right"]["n_pass"] == 1  # C was right on question 1
    assert r["paired"] == {}  # the pair is D and S
    flags = r["flags"]
    assert len(flags["space_lost"]) == 1 and flags["space_lost"][0].endswith("q3__r1__localmelo__S-cua")
    assert len(flags["captcha"]) == 1 and flags["captcha"][0].endswith("q4__r1__localmelo__S-cua")
    md = report.render_markdown(r)
    assert "| localmelo | S-cua | all | 3 | 2 | 1 |" in md and "localmelo S-cua | localmelo C" in md
    assert "- S-cua window off the Space on screen (left out of the counts): `" in md
