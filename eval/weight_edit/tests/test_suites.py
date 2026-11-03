import os
from pathlib import Path

import pytest

from wsp import suites, util

needs_data = pytest.mark.skipif(not (os.environ.get("WSP_LME") and Path(os.environ["WSP_LME"]).exists() and util.stage_dir().exists()), reason="datasets not available")


def test_rounds_of_session_pairs_user_with_following_assistant():
    turns = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "q1", "has_answer": True},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a1b"},
        {"role": "user", "content": "  "},
        {"role": "user", "content": "q2"},
    ]
    r = suites.rounds_of_session(turns)
    assert [t for t, _ in r] == ["Assistant: orphan", "User: q1\nAssistant: a1\nAssistant: a1b", "User: q2"]
    assert [h for _, h in r] == [False, True, False]


def test_parse_evidence_handles_odd_spellings():
    assert suites.parse_evidence("D1:3") == ["D1:3"]
    assert suites.parse_evidence("D9:1 D4:4 D4:6") == ["D9:1", "D4:4", "D4:6"]
    assert suites.parse_evidence("D8:6; D9:17") == ["D8:6", "D9:17"]
    assert suites.parse_evidence("D:11:26") == ["D11:26"]
    assert suites.parse_evidence("D30:05") == ["D30:5"]
    assert suites.parse_evidence("D") == []


def test_format_instruct_matches_the_model_card():
    assert suites.format_instruct("do x", "hello") == "Instruct: do x\nQuery:hello"
    assert suites.format_instruct(None, "hello") == "hello"


@needs_data
def test_dev_and_test_lme_ids_are_disjoint_and_clean():
    suites.assert_split_disjoint()
    import json

    questions = json.loads(util.lme_path().read_text())
    dev = suites.lme_dev_ids(questions, 100)
    test = suites.lme_test_ids()
    assert len(dev) == 100 and len(set(dev)) == 100
    assert len(test) == 102 and not set(dev) & set(test)
    assert not any(suites.is_abstention(i) for i in dev + test)
    assert not set(dev) & (set(suites.stage_ids()) | set(suites.pilot_ids()))
    assert not suites.mentioned_ids(dev, util.stage_dir())
    assert suites.lme_dev_ids(questions, 10) == dev[:10]  # a smaller dev set is a prefix


@needs_data
def test_limited_task_keeps_gold_and_is_deterministic():
    task = suites.load_lme(suites.lme_test_ids()[:6], "t")
    small = suites.limited(task, 3)
    assert small.n_queries() == 3 and small.limited
    for g in small.groups:
        assert len(g.dids) < 40
        assert all(g.qrels[i] for i in range(len(g.qids)))  # gold survives
    again = suites.limited(task, 3)
    assert [g.qids for g in small.groups] == [g.qids for g in again.groups]
    assert small.text_hash() == again.text_hash() != task.text_hash()
