"""TM57 turn-weighted cuts on made-up histories; no gateway, model or dataset reads."""
from dataclasses import replace

import pytest

from memory.public import data, run


def document(counts, dates=None):
    dates = dates or [f"2023-05-{i + 1:02d}T10:00:00" for i in range(len(counts))]
    return {"sessions": [
        {"id": f"s{i}", "date": date,
         "turns": [{"role": "user" if j % 2 == 0 else "assistant", "text": "Synthetic turn."}
                   for j in range(count)]}
        for i, (count, date) in enumerate(zip(counts, dates, strict=True))
    ]}


def ids(part):
    return [s["id"] for s in part["sessions"]]


def question_for(doc, answers=()):
    return {
        "question_id": "made-up",
        "haystack_session_ids": [s["id"] for s in doc["sessions"]],
        "haystack_dates": [s["date"] for s in doc["sessions"]],
        "haystack_sessions": [[{"role": t["role"], "content": t["text"]} for t in s["turns"]]
                              for s in doc["sessions"]],
        "answer_session_ids": list(answers),
    }


def study(tmp_path, **options):
    return run.Study(
        run_dir=tmp_path / "run", questions=[], ids=(), arms={}, arm_names=(), repeats=1,
        localmelo=tmp_path, expect_sha="", base_url="", model="", embedding_model="", judge_model="",
        ingest_x=("ingest=compress", "memory.example=1"), **options,
    )


def test_turn_cut_is_the_smallest_oldest_prefix_not_a_session_fraction():
    doc = document([1, 8, 1], ["2023-05-03", "2023-05-01", "2023-05-02"])
    encoded, plain = run.split_sessions_by_turns(doc, 0.9)
    assert ids(encoded) == ["s1", "s2"] and ids(plain) == ["s0"]
    assert sum(len(s["turns"]) for s in encoded["sessions"]) == 9
    assert len(encoded["sessions"][0]["turns"]) / 10 < 0.9
    assert len(run.split_sessions(doc, 0.9)[0]["sessions"]) == 3
    # Half the sessions can already hold 90% of turns: do not require a turns source in that smoke.
    assert ids(run.split_sessions(doc, 0.5)[0]) == ids(encoded)
    assert ids(doc) == ["s0", "s1", "s2"]  # no in-place sorting or mutation


def test_exact_boundary_and_date_ties_keep_file_order():
    doc = document([9, 1], ["2023-05-01", "2023-05-01"])
    encoded, plain = run.split_sessions_by_turns(doc, 0.9)
    assert ids(encoded) == ["s0"] and ids(plain) == ["s1"]


def test_whole_sessions_can_overshoot_the_target():
    encoded, plain = run.split_sessions_by_turns(document([4, 4, 4]), 0.9)
    assert ids(encoded) == ["s0", "s1", "s2"] and not plain["sessions"]


def test_recent_keeps_middle_answer_plain_and_encodes_other_sessions_oldest_first():
    encoded, plain = run.split_sessions_by_turns(document([8, 1, 1]), 0.9, ["s1"])
    assert ids(encoded) == ["s0", "s2"] and ids(plain) == ["s1"]
    assert sum(len(s["turns"]) for s in encoded["sessions"]) == 9


def test_recent_keeps_the_row_when_answer_turns_make_target_unreachable():
    encoded, plain = run.split_sessions_by_turns(document([2, 1, 7]), 0.9, ["s0"])
    assert ids(encoded) == ["s1", "s2"] and ids(plain) == ["s0"]
    # Denominator includes the protected answer; do not silently renormalize the eligible history.
    assert sum(len(s["turns"]) for s in encoded["sessions"]) / 10 == 0.8


def test_empty_or_all_protected_history_stays_plain():
    assert run.split_sessions_by_turns({"sessions": []}, 0.9) == ({"sessions": []}, {"sessions": []})
    doc = document([3, 2])
    assert run.split_sessions_by_turns(doc, 0.9, ["s0", "s1"]) == ({"sessions": []}, doc)
    zero = document([0, 0])
    assert run.split_sessions_by_turns(zero, 0.9) == ({"sessions": []}, zero)


@pytest.mark.parametrize("share", [0, 1, -0.1, float("nan")])
def test_turn_cut_rejects_invalid_shares(share):
    with pytest.raises(ValueError, match="between 0 and 1"):
        run.split_sessions_by_turns(document([1]), share)


def test_answer_metadata_protects_repeated_copies_and_deduplicates_exact_repeats():
    turns = [{"role": "user", "content": "Synthetic user."}, {"role": "assistant", "content": "Synthetic reply."}]
    q = {
        "haystack_session_ids": ["answer", "other", "answer", "answer"],
        "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/02 (Tue) 10:00",
                           "2023/05/03 (Wed) 10:00", "2023/05/01 (Mon) 10:00"],
        "haystack_sessions": [turns, turns, turns, turns], "answer_session_ids": ["answer"],
    }
    doc = data.to_ingest(q)
    protected = run.answer_file_ids(q, doc)
    assert protected == ["answer", f"answer{data.REPEAT_MARK}2"]
    encoded, plain = run.split_sessions_by_turns(doc, 0.9, protected)
    assert ids(encoded) == ["other"] and ids(plain) == protected
    assert len(doc["sessions"]) == 3


def test_answer_cut_needs_usable_metadata_not_text_matching():
    doc = document([2, 2])
    q = question_for(doc)
    assert run.answer_file_ids(q, doc) == []  # explicit empty evidence metadata is allowed
    del q["answer_session_ids"]
    with pytest.raises(run.IngestError, match="needs answer_session_ids"):
        run.answer_file_ids(q, doc)
    q["answer_session_ids"] = ["absent"]
    with pytest.raises(run.IngestError, match="absent from"):
        run.answer_file_ids(q, doc)
    q["answer_session_ids"] = ["s0"]
    with pytest.raises(run.IngestError, match="omits an answer"):
        run.answer_file_ids(q, {"sessions": [doc["sessions"][1]]})


def test_flags_off_preserve_old_checkpoint_identity_and_session_share(tmp_path, monkeypatch):
    monkeypatch.setattr(run.server, "git_sha", lambda path: "fixed-sha")
    original = study(tmp_path)
    expected = {"notes": "rules", "x": ["ingest=compress", "memory.example=1"]}
    assert original.identity()["ingest"] == expected
    legacy = replace(original, encode_share=0.5)
    assert legacy.identity()["ingest"] == {**expected, "encode_share": 0.5}
    auto = replace(original, encode_turn_share=0.9)
    assert auto.identity()["ingest"] == {**expected, "encode_turn_share": 0.9}
    recent = replace(auto, keep_answer_sessions_plain=True)
    assert recent.identity()["ingest"] == {**expected, "encode_turn_share": 0.9, "keep_answer_sessions_plain": True}


@pytest.mark.parametrize("options,expected", [
    ({}, [["s0", "s1", "s2"]]),
    ({"encode_share": 0.5}, [["s0", "s1"], ["s2"]]),
    ({"encode_turn_share": 0.9}, [["s0", "s1"], ["s2"]]),
    ({"encode_turn_share": 0.9, "keep_answer_sessions_plain": True}, [["s0", "s2"], ["s1"]]),
])
def test_ingest_uses_the_selected_cut_and_keeps_plain_x_pairs(tmp_path, monkeypatch, options, expected):
    seen = []
    doc = document([8, 1, 1])
    q = question_for(doc, ["s1"])

    def fake_ingest(study, question, home, source, part, xs):
        seen.append((ids(part), tuple(xs)))
        return {"sessions": len(part["sessions"]), "turns": sum(len(s["turns"]) for s in part["sessions"]),
                "ids": {s["id"]: "stored-" + s["id"] for s in part["sessions"]}}

    monkeypatch.setattr(run, "run_ingest", fake_ingest)
    monkeypatch.setattr(run, "template_home", lambda *args: tmp_path / "home")
    monkeypatch.setattr(run, "write_ingest_config", lambda *args: None)
    report = run.ingest(study(tmp_path, **options), q, tmp_path, doc)
    assert [part for part, xs in seen] == expected
    assert seen[0][1] == ("ingest=compress", "memory.example=1")
    if len(seen) == 2:
        assert seen[1][1] == ("memory.example=1",)
        assert report["encoded_sessions"] == 2 and report["plain_sessions"] == 1
    assert report["turns"] == 10 and len(report["ids"]) == 3


def test_unreachable_recent_cut_encodes_all_eligible_sessions_without_a_second_pass(tmp_path, monkeypatch):
    doc = document([10])
    q = question_for(doc, ["s0"])
    seen = []

    def fake_ingest(study, question, home, source, part, xs):
        seen.append((ids(part), tuple(xs)))
        return {"sessions": 1, "turns": 10, "ids": {"s0": "stored-s0"}}

    monkeypatch.setattr(run, "run_ingest", fake_ingest)
    monkeypatch.setattr(run, "template_home", lambda *args: tmp_path / "home")
    monkeypatch.setattr(run, "write_ingest_config", lambda *args: None)
    report = run.ingest(study(tmp_path, encode_turn_share=0.9, keep_answer_sessions_plain=True), q, tmp_path, doc)
    assert seen == [(["s0"], ("memory.example=1",))]
    assert (report["encoded_sessions"], report["plain_sessions"], report["turns"]) == (0, 1, 10)


def test_cli_keeps_old_defaults_and_makes_cut_modes_mutually_exclusive():
    argv = ["--run-dir", "/synthetic/run", "--localmelo", "/synthetic/app", "--localmelo-sha", "synthetic",
            "--base-url", "synthetic", "--arms-file", "/synthetic/arms.json"]
    args = run.parse_args(argv)
    assert args.encode_share is None and args.encode_turn_share is None and not args.keep_answer_sessions_plain
    args = run.parse_args(argv + ["--encode-turn-share", "0.9", "--keep-answer-sessions-plain"])
    assert args.encode_turn_share == 0.9 and args.keep_answer_sessions_plain
    with pytest.raises(SystemExit):
        run.parse_args(argv + ["--encode-share", "0.5", "--encode-turn-share", "0.9"])
    assert run.encode_turn_share(None, ()) is None
    assert run.encode_turn_share(0.9, ["ingest=compress"], True) == 0.9
    for share, pairs, protected in [(None, (), True), (0.9, (), False), (0, ("ingest=compress",), False)]:
        with pytest.raises(SystemExit):
            run.encode_turn_share(share, pairs, protected)
