"""stats.py: sign test, per-task scores, the decision rules of DESIGN §6.6, MDD, paired times."""

from __future__ import annotations

import math

import pytest

from modeleval import stats

H = ("claude_code", "hermes", "localmelo")


def rec(h: str, task: str, solved: bool, set_name: str = "gate", **kw):
    return {"harness": h, "task": task, "set": set_name, "verdict": "pass" if solved else "fail",
            "check": {"passed": solved}, "wall_s": kw.pop("wall_s", 100.0), "model_s": kw.pop("model_s", 80.0),
            "steps": kw.pop("steps", 5), **kw}


def grid(solved: dict[str, tuple[bool, bool, bool]], set_name: str = "gate", **kw):
    return [rec(h, t, s, set_name, **kw) for t, row in solved.items() for h, s in zip(H, row)]


def test_sign_test_exact_values():
    assert stats.sign_test(0, 0) == 1.0
    assert stats.sign_test(6, 0) == pytest.approx(2 / 64)
    assert stats.sign_test(0, 6) == pytest.approx(2 / 64)
    assert stats.sign_test(5, 0) == pytest.approx(2 / 32)
    assert stats.sign_test(3, 3) == 1.0
    n = 22
    assert stats.sign_test(17, 5) == pytest.approx(2 * sum(math.comb(n, k) for k in range(6)) / 2**n)


def test_min_margin_and_discordant():
    assert stats.min_discordant() == 6
    assert stats.min_margin(5) is None
    assert stats.min_margin(6) == 6
    assert stats.min_margin(22) == 12  # 17:5 is p 0.017, 16:6 is p 0.052
    assert stats.min_margin(0) is None


def test_wilson_display_interval():
    lo, hi = stats.wilson(11, 12)
    assert 0.6 < lo < 0.7 and 0.98 < hi <= 1.0
    assert stats.wilson(0, 0) == (None, None)


def test_solved_counts_a_checker_pass_after_a_wall_timeout():
    r = {"harness": "hermes", "task": "t", "set": "gate", "verdict": "timeout", "check": {"passed": True},
         "wall_s": 3600.0}
    assert stats.rec_solved(r) is True
    assert stats.rec_solved_fast(r) is False  # not within the 1200 s column
    assert stats.rec_solved_fast(rec("hermes", "t", True, wall_s=1199.0)) is True


def test_scores_drop_invalid_and_keep_the_last_rerun():
    rs = [rec("hermes", "a", False), rec("hermes", "a", True),  # re-run: the last record wins
          {**rec("localmelo", "a", True), "verdict": "invalid"},  # pipeline mismatch: excluded
          rec("claude_code", "a", True), rec("hermes", "b", False), rec("hermes", "c", True, set_name="ranking")]
    assert stats.per_task_scores(rs, "gate") == {"a": 2, "b": 0}
    assert stats.per_harness(rs, "gate") == {"claude_code": {"a": True}, "hermes": {"a": True, "b": False}}
    assert stats.per_task_scores(rs, "ranking") == {"c": 1}


def test_gate_rule_needs_margin_and_two_consistent_harnesses():
    tasks = [f"t{i:02d}" for i in range(22)]
    ref = grid({t: (True, True, True) for t in tasks})
    # new model loses 5 tasks in claude_code and hermes: net -5 at task level, two harnesses behind
    new = grid({t: ((False, False, True) if i < 5 else (True, True, True)) for i, t in enumerate(tasks)})
    c = stats.compare(new, ref, set_name="gate")
    assert (c["wins"], c["losses"], c["ties"], c["net"]) == (0, 5, 17, -5)
    assert c["pairs"] == 66 and c["excluded"] == [] and not c["provisional"]
    assert c["per_harness"]["claude_code"]["net"] == -5 and c["per_harness"]["localmelo"]["net"] == 0
    assert c["decision"] == "worse"
    assert c["mdd"]["gate_margin"] == 5 and c["tasks"] == 22


def test_gate_rule_single_harness_loss_is_not_worse():
    tasks = [f"t{i:02d}" for i in range(22)]
    ref = grid({t: (True, True, True) for t in tasks})
    new = grid({t: ((False, True, True) if i < 6 else (True, True, True)) for i, t in enumerate(tasks)})
    c = stats.compare(new, ref, set_name="gate")
    assert c["losses"] - c["wins"] == 6
    assert c["decision"] == "not worse"  # only claude_code is behind
    four = stats.compare(grid({f"t{i}": (False, False, True) for i in range(4)}),
                         grid({f"t{i}": (True, True, True) for i in range(4)}), set_name="gate")
    assert four["decision"] == "not worse"  # a 4-task margin is below the rule


def _scores(score: dict[str, int], set_name: str = "ranking") -> list[dict]:
    """Records with `score[t]` of the three harnesses solving task t."""
    return grid({t: tuple(i < n for i in range(3)) for t, n in score.items()}, set_name)


def test_ranking_rule_uses_the_sign_test():
    ref_s = {f"t{i}": 1 for i in range(22)}
    ref = _scores(ref_s)
    better = {t: (3 if i < 6 else 1) for i, t in enumerate(ref_s)}
    c = stats.compare(_scores(better), ref, set_name="ranking")
    assert c["p"] == pytest.approx(2 / 64) and c["decision"] == "better"
    worse = {t: (0 if i < 6 else 1) for i, t in enumerate(ref_s)}
    assert stats.compare(_scores(worse), ref, set_name="ranking")["decision"] == "worse"
    mixed = {t: (3 if i < 6 else 0 if i < 9 else 1) for i, t in enumerate(ref_s)}
    m = stats.compare(_scores(mixed), ref, set_name="ranking")
    assert m["decision"] == "no detectable difference" and m["p"] > 0.05
    assert m["mdd"]["discordant"] == 9 and m["mdd"]["margin_at_discordant"] == stats.min_margin(9)
    assert m["mdd"]["margin_all_tasks"] == 12


def test_compare_reports_tasks_missing_on_one_side():
    c = stats.compare(_scores({"a": 1, "b": 2}), _scores({"a": 1, "c": 3}), set_name="ranking")
    assert c["tasks"] == 1 and c["only_new"] == ["b"] and c["only_ref"] == ["c"]
    assert c["provisional"] and len(c["excluded"]) == 6
    assert "claude_code/b: no valid record for reference" in c["excluded"]


def test_an_excluded_pair_is_never_counted_as_a_loss():
    """A pair invalid twice on one side (excluded) caps that side's task at 0-2. Scored per task over
    the harnesses that have a valid record on both sides, it is not a loss."""
    tasks = [f"t{i:02d}" for i in range(22)]
    ref = grid({t: (True, True, True) for t in tasks})
    new = grid({t: (True, True, True) for t in tasks})
    for t in tasks[:6]:   # claude_code invalid twice on six tasks of the new model
        i = next(i for i, r in enumerate(new) if r["task"] == t and r["harness"] == "claude_code")
        new[i] = {**new[i], "verdict": "invalid"}
        new.append({**new[i]})
    c = stats.compare(new, ref, set_name="gate")
    assert (c["wins"], c["losses"], c["ties"]) == (0, 0, 22) and c["decision"] == "not worse"
    assert c["pairs"] == 60 and c["provisional"] and len(c["excluded"]) == 6
    assert c["excluded"][0] == "claude_code/t00: no valid record for new"
    assert c["score_new"] == c["score_ref"] == 60
    # per_task_scores alone would have put those six tasks at 2 against 3
    assert sum(1 for t in tasks if stats.per_task_scores(new, "gate")[t] < 3) == 6


def test_time_to_solution_is_paired():
    new = [rec("hermes", "a", True, wall_s=50, model_s=40, steps=4), rec("hermes", "b", True, wall_s=500),
           rec("localmelo", "a", True, wall_s=30, model_s=20, steps=2)]
    ref = [rec("hermes", "a", True, wall_s=100, model_s=90, steps=6), rec("hermes", "b", False, wall_s=10),
           rec("localmelo", "a", True, wall_s=20, model_s=10, steps=4)]
    t = stats.time_to_solution(new, ref, "gate")
    assert t["pairs"] == 2  # hermes/b is solved by the new model only: unpaired, left out
    assert (t["wall_new"], t["wall_ref"]) == (80, 120)
    assert t["wall_ratio"] == pytest.approx(80 / 120)
    assert t["model_ratio"] == pytest.approx(60 / 100)
    assert (t["steps_new"], t["steps_ref"]) == (3, 5)


def test_harness_counts_and_anomalies():
    rs = [rec("hermes", "a", True, anomalies=["5xx"]), rec("hermes", "b", True, wall_s=1500), rec("hermes", "c", False)]
    hc = stats.harness_counts(rs, "gate")["hermes"]
    assert (hc["solved"], hc["fast"], hc["runs"], hc["anomalies"]) == (2, 1, 3, 1)
    assert hc["wilson"][0] is not None
