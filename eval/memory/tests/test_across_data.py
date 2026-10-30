"""The across-chats task file: schema, counts, ids, teaching order, and the validator's refusals."""
from pathlib import Path

import pytest
import yaml
from memory.across import data, score
from memory.tests.across_fake import mini_raw


@pytest.fixture(scope="module")
def tasks():
    return data.load()


def raw():
    return yaml.safe_load(data.DATA_FILE.read_text(encoding="utf-8"))


def refuses(mutate, fragment, expect=data.V1):
    doc = raw()
    mutate(doc)
    with pytest.raises(data.DataError) as err:
        data.parse(doc, expect)
    assert fragment in str(err.value)


def test_counts_per_kind_and_role(tasks):
    kinds = {k: sum(q.kind == k for q in tasks.questions) for k in data.KINDS}
    assert kinds == {"exact": 20, "paraphrase": 15, "crosslang": 10, "superseded": 5, "lookalike": 5, "nothing": 5}
    roles = {r: sum(f.role == r for f in tasks.facts) for r in data.ROLES}
    assert roles == {"plain": 20, "old": 5, "new": 5, "target": 5, "lookalike": 5, "filler": 3}
    assert len(tasks.facts) == 43 and len(tasks.questions) == 60
    assert len({f.block for f in tasks.facts}) == 10
    assert {f.lang for f in tasks.facts} == {"en", "zh"} and {q.lang for q in tasks.questions} == {"en", "zh"}


def test_ids_are_unique_and_gold_facts_exist(tasks):
    assert len({f.id for f in tasks.facts}) == 43 and len({q.id for q in tasks.questions}) == 60
    for q in tasks.questions:
        for fact_id in (*q.facts, *q.wrong_facts):
            assert fact_id in tasks.by_fact, (q.id, fact_id)
        assert (not q.facts) == (q.kind == "nothing")
        assert bool(q.answer) == (q.kind != "nothing")


def test_teaching_order_and_superseded_values(tasks):
    assert [f.order for f in tasks.facts] == list(range(1, 44))
    for q in (q for q in tasks.questions if q.kind == "superseded"):
        new, old = tasks.fact(q.facts[0]), tasks.fact(q.wrong_facts[0])
        assert old.order < new.order and new.supersedes == old.id
        assert q.reject and any(score.contains(old.text, r) for r in q.reject)
    assert [f.role for f in tasks.facts[-3:]] == ["filler"] * 3  # the pinned-quote window is full of fillers
    assert sum(len(f.text) + 5 for f in tasks.facts[-3:]) > data.PIN_WINDOW


def test_languages_fit_the_kinds(tasks):
    for q in tasks.questions:
        if q.facts:
            fact = tasks.fact(q.facts[0])
            assert (fact.lang != q.lang) == (q.kind == "crosslang"), q.id
    cross = [q for q in tasks.questions if q.kind == "crosslang"]
    assert {q.lang for q in cross} == {"en", "zh"} and sum(q.lang == "zh" for q in cross) == 5


def test_every_gold_answer_is_in_its_fact_and_scores_itself(tasks):
    for q in tasks.questions:
        if q.kind == "nothing":
            assert score.grade(q, "I don't know.").correct and score.grade(q, "我不知道。").correct
            continue
        assert score.contains(tasks.fact(q.facts[0]).text, q.answer), q.id
        for target in q.targets:
            assert score.grade(q, f"It is {target}.").correct, (q.id, target)
        assert not score.grade(q, "Something else entirely.").correct


def test_ask_order_is_a_fixed_shuffle(tasks):
    order = tasks.ask_order()
    assert sorted(q.id for q in order) == sorted(q.id for q in tasks.questions)
    assert order == tasks.ask_order() and [q.id for q in order] != [q.id for q in tasks.questions]
    assert [q.id for q in order][:3] == ["Q07", "Q58", "Q25"]  # changes only when ask_seed or an id does
    assert len(tasks.sha256()) == 64


def test_public_text_has_no_home_paths():
    text = data.DATA_FILE.read_text(encoding="utf-8")
    assert "/Users/" not in text and "~/" not in text


def test_mini_set_loads_without_the_v1_counts():
    assert len(data.parse(mini_raw(), None).questions) == 5
    with pytest.raises(data.DataError):
        data.parse(mini_raw())  # v1 counts do not hold for it


def test_refuses_wrong_counts():
    refuses(lambda d: d["questions"].pop(), "questions per kind")
    refuses(lambda d: d["facts"][3].update(block="B09"), "block")


def test_refuses_unknown_ids_and_bad_fields():
    refuses(lambda d: d["questions"][0].update(facts=["F99"]), "unknown fact ids")
    refuses(lambda d: d["questions"][0].update(kind="fuzzy"), "kind 'fuzzy'")
    refuses(lambda d: d["facts"][0].update(lang="fr"), "lang 'fr'")
    refuses(lambda d: d["facts"][0].update(color="red"), "unknown keys")
    refuses(lambda d: d["questions"][1].update(id="Q01"), "duplicate ids")
    refuses(lambda d: d["facts"][1].update(id="F01"), "duplicate ids")


def test_refuses_superseded_taught_in_the_wrong_order():
    def swap(d):
        d["facts"][2]["order"], d["facts"][32]["order"] = 33, 3
    refuses(swap, "order must be 1..N")

    def reorder(d):
        facts = d["facts"]
        facts[2], facts[32] = facts[32], facts[2]
        for i, f in enumerate(facts, 1):
            f["order"] = i
    refuses(reorder, "taught before the fact it supersedes")


def test_refuses_fillers_that_do_not_fill_the_window():
    refuses(lambda d: d["facts"][-1].update(text="I prefer short answers."), "a filler needs")
    refuses(lambda d: d["facts"].insert(10, d["facts"].pop()), "order must be")


def test_refuses_questions_that_do_not_fit_their_kind():
    refuses(lambda d: d["questions"][0].update(lang="zh"), "language it was taught in")
    refuses(lambda d: d["questions"][35].update(lang="en"), "other language")
    refuses(lambda d: d["questions"][0].update(answer="not in the text"), "is not in the text of")
    refuses(lambda d: d["questions"][45].update(reject=[]), "needs reject strings")
    refuses(lambda d: d["questions"][45].update(reject=["Utrecht"]), "is in the gold fact's text")
    refuses(lambda d: d["questions"][55].update(answer="x"), "nothing question")


def test_refuses_a_plain_fact_nobody_asks():
    def drop_asks(d):
        d["questions"] = [q for q in d["questions"] if "F01" not in q["facts"]]
    refuses(drop_asks, "no question asks", expect=None)


def test_refuses_broken_twins():
    refuses(lambda d: d["facts"][7].update(twin="F01"), "twin")
    refuses(lambda d: d["facts"][32].update(supersedes="F01"), "not an old fact")


def test_load_reads_a_file(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump(mini_raw(), allow_unicode=True), encoding="utf-8")
    assert len(data.load(Path(path), None).facts) == 4
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(data.DataError):
        data.load(path)
