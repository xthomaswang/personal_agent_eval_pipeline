"""The public benchmark's data: the stage set, the conversion to localmelo's ingest file, the dates."""
import json

import pytest
from memory.public import data
from memory.tests.public_support import mini_questions, synthetic_questions, write_dataset


def test_stage_set_is_deterministic_and_stratified():
    questions = synthetic_questions()
    stage = data.stage_set(questions, per_type=3, abstention=4, seed=0)
    assert stage == data.stage_set(list(reversed(questions)), per_type=3, abstention=4, seed=0)  # the order of the file does not matter
    assert stage == sorted(stage) and len(stage) == len(set(stage)) == 6 * 3 + 4
    by_id = {q["question_id"]: q for q in questions}
    plain = [by_id[i] for i in stage if not i.endswith("_abs")]
    assert {t: sum(1 for q in plain if q["question_type"] == t) for t in data.TYPES} == dict.fromkeys(data.TYPES, 3)
    absent = [by_id[i] for i in stage if i.endswith("_abs")]
    # four abstention questions go round three types in alphabetical order: one each, then one more for the first
    assert sorted(q["question_type"] for q in absent) == ["knowledge-update", "knowledge-update", "multi-session", "temporal-reasoning"]


def test_stage_set_seed_counts_and_short_types():
    questions = synthetic_questions()
    assert data.stage_set(questions, 3, 4, seed=1) != data.stage_set(questions, 3, 4, seed=0)
    assert len(data.stage_set(questions, 2, 0)) == 12 and len(data.stage_set(questions, 0, 5)) == 5
    everything = data.stage_set(questions, per_type=99, abstention=99)
    assert len(everything) == len(questions)  # fewer questions than asked for: all of them, no error
    assert len(data.stage_set(synthetic_questions(per_type=17), 17, 18)) == 6 * 17 + 9  # only 9 abstention questions exist here


def test_defaults_are_the_120_question_stage_set():
    assert (data.PER_TYPE, data.ABSTENTION_COUNT, data.SEED) == (17, 18, 0)
    questions = synthetic_questions(per_type=20, abs_each=6)
    stage = data.stage_set(questions)
    assert len(stage) == 120 and sum(1 for i in stage if i.endswith("_abs")) == 18


def test_select_by_count_by_ids_and_default():
    questions = synthetic_questions()
    stage = data.stage_set(questions, 3, 4)
    assert data.select(questions, "", 3, 4) == stage
    small = data.select(questions, "5", 3, 4)
    assert small == data.select(questions, "5", 3, 4) and len(small) == 5 and set(small) <= set(stage)
    by_id = {q["question_id"]: q for q in questions}
    assert len({data.group(by_id[i]) for i in data.select(questions, "7", 3, 4)}) == 7  # spread over the groups, abstention included
    assert data.select(questions, "999", 3, 4) == stage
    ids = [questions[3]["question_id"], questions[0]["question_id"]]
    assert data.select(questions, ",".join(ids), 3, 4) == sorted(ids)  # ids need not be in the stage set
    with pytest.raises(data.DataError):
        data.select(questions, "nope", 3, 4)


def test_to_ingest_matches_localmelos_format():
    q = mini_questions()[0]
    doc = data.to_ingest(q)
    assert list(doc) == ["sessions"] and [s["id"] for s in doc["sessions"]] == ["sess_a", "sess_b", "sess_c"]
    first = doc["sessions"][0]
    assert first == {"id": "sess_a", "date": "2023-05-20T02:21:00", "turns": [
        {"role": "user", "text": "I hold a degree in Business Administration."},
        {"role": "assistant", "text": "Congratulations on the degree!"}]}
    assert all(set(t) == {"role", "text"} for s in doc["sessions"] for t in s["turns"])  # has_answer is not passed on
    assert all("title" not in s for s in doc["sessions"])  # a title would show the model a chat label
    assert doc["sessions"][2]["turns"][-1] == {"role": "user", "text": "  "}  # blank turns are localmelo's to skip
    assert json.loads(json.dumps(doc)) == doc


def test_a_repeated_haystack_session_is_kept_under_a_marked_id():
    q = mini_questions()[0]
    q["haystack_session_ids"] = ["sess_a", "sess_b", "sess_a"]  # the same session twice, as 13 of the 500 questions have
    q["haystack_sessions"][2] = q["haystack_sessions"][0]
    doc = data.to_ingest(q)
    assert [s["id"] for s in doc["sessions"]] == ["sess_a", "sess_b", "sess_a#2"]
    assert doc["sessions"][2]["turns"] == doc["sessions"][0]["turns"] and doc["sessions"][2]["date"] == "2023-05-22T09:30:00"
    assert data.haystack_ids(q) == {"sess_a": "sess_a", "sess_b": "sess_b", "sess_a#2": "sess_a"}
    assert data.file_session_ids(mini_questions()[0]) == ["sess_a", "sess_b", "sess_c"]
    q["haystack_dates"][2] = q["haystack_dates"][0]  # word for word, the date too: written once
    assert [s["id"] for s in data.to_ingest(q)["sessions"]] == ["sess_a", "sess_b"]
    assert data.haystack_ids(q) == {"sess_a": "sess_a", "sess_b": "sess_b"}


def test_dates_are_iso_local_time():
    assert data.iso_local("2023/05/30 (Tue) 23:40") == "2023-05-30T23:40:00"
    assert data.iso_local("2024/01/02 (Tue) 00:05") == "2024-01-02T00:05:00"
    assert data.iso_local("2024/01/02 00:05") == "2024-01-02T00:05:00"
    assert data.question_now(mini_questions()[1]) == "2023-06-02T08:05:00"
    for bad in ("", "2023-05-30", "30/05/2023 (Tue) 23:40"):
        with pytest.raises(data.DataError):
            data.iso_local(bad)


def test_load_checks_the_file(tmp_path):
    path = write_dataset(tmp_path / "d.json")
    assert [q["question_id"] for q in data.load(path)] == ["q1aa", "q2bb_abs"]
    assert len(data.file_sha256(path)) == 64
    broken = mini_questions()
    del broken[0]["answer"]
    with pytest.raises(data.DataError, match="missing answer"):
        data.load(write_dataset(tmp_path / "e.json", broken))
    short = mini_questions()
    short[0]["haystack_dates"].pop()
    with pytest.raises(data.DataError, match="differ in length"):
        data.load(write_dataset(tmp_path / "f.json", short))
    odd = mini_questions()
    odd[0]["question_type"] = "other"
    with pytest.raises(data.DataError, match="unknown question type"):
        data.load(write_dataset(tmp_path / "g.json", odd))
    with pytest.raises(data.DataError):
        data.load(tmp_path / "missing.json")


def test_what_a_run_records_about_a_question_has_none_of_its_text():
    q1, q2 = mini_questions()
    assert data.summary(q1) == {"question_id": "q1aa", "question_type": "single-session-user", "abstention": False,
                                "answer_sessions": ["sess_a"], "sessions": 3, "turns": 7}
    assert data.group(q2) == data.ABSTENTION and data.group(q1) == "single-session-user"
    assert data.answer_user_texts(q1) == {"sess_a": ["I hold a degree in Business Administration."]}


def test_the_frozen_stage_set_is_read_as_given_and_refuses_an_unknown_id(tmp_path):
    questions = [{"question_id": "q1"}, {"question_id": "q2_abs"}, {"question_id": "q3"}]
    frozen = tmp_path / "stage.txt"
    frozen.write_text("# ids only\nq3\nq1\n\n", encoding="utf-8")
    assert data.frozen_stage_set(questions, frozen) == ["q1", "q3"]
    assert data.frozen_stage_set(questions, tmp_path / "missing.txt") is None
    frozen.write_text("q1\nq9\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="q9"):
        data.frozen_stage_set(questions, frozen)
    frozen.write_text("x1\nx2\n", encoding="utf-8")  # another data file's ids: not this one's frozen set
    assert data.frozen_stage_set(questions, frozen) is None


def test_the_committed_stage_set_has_120_ids_with_18_abstentions():
    ids = [line for line in data.STAGE_SET_FILE.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert len(ids) == len(set(ids)) == 120
    assert sum(i.endswith("_abs") for i in ids) == 18


def test_evidence_kept_measures_how_much_of_each_evidence_turn_the_index_keeps():
    q = {"haystack_sessions": [[
        {"role": "user", "content": "u" * 100, "has_answer": True},
        {"role": "assistant", "content": "a" * 3000, "has_answer": True},
    ], [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]]}
    kept = data.evidence_kept(q)
    assert kept["turns"] == 2 and kept["full"] == 1 and kept["best"] == 1.0
    q["haystack_sessions"][0][0]["has_answer"] = False
    only_reply = data.evidence_kept(q)
    assert only_reply["full"] == 0 and only_reply["best"] == round((2000 - 6 - 100 - 7) / 3000, 3)
    assert data.evidence_kept({"haystack_sessions": [[{"role": "user", "content": "hi"}]]})["best"] is None
