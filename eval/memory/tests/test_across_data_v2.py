"""The v2 task file (tasks_v2.yaml): kinds, counts, language mix, gold strings against the filler, and the validator's
refusals. v1's file and fingerprint stay what they were."""
import copy
import hashlib

import pytest
import yaml
from memory.across import data, data_v2, score
from memory.tests.across_fake import mini_raw_v2

V1_FILE_SHA256 = "b51e41424cd80b8925135195d596a62c9e16ade2f90556e7b6764934f01baf18"
V1_TASKS_SHA256 = "8cd7464b028ab3d76887d2cbc3ae70bff8e2a44cf8048dc66c4c4f61d557a77b"


@pytest.fixture(scope="module")
def tasks():
    return data_v2.load()


def raw():
    return yaml.safe_load(data_v2.DATA_FILE.read_text(encoding="utf-8"))


def refuses(mutate, fragment, expect=data_v2.V2):
    doc = raw()
    mutate(doc)
    with pytest.raises(data.DataError) as err:
        data_v2.parse(doc, expect)
    assert fragment in str(err.value)


def item_index(doc, item_id):
    return next(n for n, i in enumerate(doc["items"]) if i["id"] == item_id)


def test_v1_file_and_fingerprint_are_untouched():
    assert hashlib.sha256(data.DATA_FILE.read_bytes()).hexdigest() == V1_FILE_SHA256
    v1 = data.load()
    assert v1.sha256() == V1_TASKS_SHA256 and v1.protocol == "across-v1"
    assert len(v1.facts) == 43 and len(v1.questions) == 60


def test_load_any_chooses_the_version_by_the_file(tasks):
    assert isinstance(data_v2.load_any(data.DATA_FILE), data.TaskSet)
    assert isinstance(data_v2.load_any(data_v2.DATA_FILE), data_v2.TaskSet)
    assert tasks.version == 2 and tasks.protocol == "across-v2" and tasks.sha256() != V1_TASKS_SHA256 and len(tasks.sha256()) == 64


def test_counts_per_kind_and_language_mix(tasks):
    for kind in ("longrange", "followup"):
        mine = [i for i in tasks.items if i.kind == kind]
        assert len(mine) == 20 and len({i.id for i in mine}) == 20
        assert 6 <= sum(i.lang == "zh" for i in mine) <= 8  # about a third
    assert {i.lang for i in tasks.items} == {"en", "zh"} and len(tasks.items) == 40
    assert set(data_v2.KINDS) == {"longrange", "followup", "recall"}  # the kinds built so far; the table grows with the file


def test_at_least_70_percent_of_long_range_facts_are_details_the_note_rules_should_not_save(tasks):
    long = [i for i in tasks.items if i.kind == "longrange"]
    assert all(isinstance(i.note_expected, bool) for i in long)
    assert sum(i.note_expected is False for i in long) >= 14 and sum(i.note_expected for i in long) >= 3  # and a control of note-worthy facts


def test_the_arm_table_is_data_per_kind():
    assert data_v2.KINDS["longrange"].arms == {"N0": "out", "M2": "out", "FC": "in"}
    assert data_v2.KINDS["followup"].arms == {"M2": "", "B0": "", "N0": ""}
    assert [k for k, spec in data_v2.KINDS.items() if "FC" in spec.arms] == ["longrange"]  # FC runs only in-chat kinds


def test_gold_strings_are_in_the_fact_and_never_in_the_filler_or_fixtures(tasks):
    hidden = [t.text for t in tasks.filler.values()] + [f.path for f in tasks.fixtures] + [f.text for f in tasks.fixtures]
    for item in tasks.items:
        for n, ask in enumerate(item.asks):
            holder = item.fact if item.kind == "longrange" else item.facts[n]
            assert score.contains(holder, ask.answer), ask.id
            if item.kind == "longrange":
                for target in ask.targets:
                    assert not any(score.contains(text, target) for text in hidden), (ask.id, target)
                assert not any(score.contains(x, t) for x in (item.opener, item.bridge, ask.text) if x for t in ask.targets), ask.id
            assert score.grade(ask, f"It is {ask.answer}.").correct and not score.grade(ask, "Something else.").correct


def test_reject_strings_are_in_the_wrong_text_and_not_in_the_gold(tasks):
    with_reject = [i for i in tasks.items if any(a.reject for a in i.asks)]
    assert len(with_reject) >= 10
    for item in with_reject:
        for n, ask in enumerate(item.asks):
            if not ask.reject:
                continue
            gold = item.fact if item.kind == "longrange" else item.facts[n]
            wrong = [item.opener, item.bridge] if item.kind == "longrange" else [f for w in item.wrong for f in tasks.item(w).facts]
            assert any(score.contains(t, r) for r in ask.reject for t in wrong), ask.id
            assert not any(score.contains(gold, r) for r in ask.reject), ask.id
            assert not score.grade(ask, f"{ask.answer} and {ask.reject[0]}").correct  # the stale string next to the gold is a miss


def test_fixtures_are_small_and_the_filler_uses_only_read_tier_tools(tasks):
    assert tasks.fixtures and all(len(f.text) <= 300 for f in tasks.fixtures)
    assert data_v2.READ_TOOLS == ("file_read", "list_files")  # git_status is not a builtin of a gateway chat
    assert {t.tool for t in tasks.filler.values()} - {""} == set(data_v2.READ_TOOLS)
    assert sum(bool(t.tool) for t in tasks.filler.values() if t.lang == "en") == 22
    assert all(any(s in t.text.lower() for s in data_v2.SHORT_ANSWER) for t in tasks.filler.values())
    paths = [f.path for f in tasks.fixtures]
    assert all(any(p in t.text for p in paths) for t in tasks.filler.values() if t.tool == "file_read")
    for name, ids in tasks.sets.items():
        turns = [tasks.filler[i] for i in ids]
        assert len(turns) >= 35 and 2 * sum(bool(t.tool) for t in turns) >= len(turns), name
        assert 2 * len(turns) >= 64 + 6, name  # the floor: even when no tool is called (2 messages a turn) the window cap + 6 follow the fact
        assert sum(4 if t.tool else 2 for t in turns) >= 110, name  # the aim, with the tool turns at 4 messages
        assert {t.lang for t in turns} == {name[:2]}


def test_a_long_chat_is_opener_fact_filler_question(tasks):
    for item in (i for i in tasks.items if i.kind == "longrange"):
        steps = tasks.long_steps(item)
        roles = [s.role for s in steps]
        assert roles[0] == "opener" and roles[-1] == "question" and roles.index("fact") + 1 == item.fact_turn in (2, 3)
        assert roles.count("filler") == 40 and ("bridge" in roles) == (item.fact_turn == 3)
        assert [s.phase for s in steps] == ["long"] * (len(steps) - 1) + ["ask"]
        assert steps[0].label == f"{item.id}.t01" and steps[-1].label == f"{item.id}.q1" and len({s.label for s in steps}) == len(steps)
        assert steps[item.fact_turn - 1].text == item.fact
        assert len(steps) >= 43  # the opener, the fact, 40 filler turns and the question (the filler test counts the messages)


def test_a_followup_item_teaches_two_facts_and_asks_a_pronoun_question(tasks):
    for item in (i for i in tasks.items if i.kind == "followup"):
        assert [s.label for s in tasks.teach_steps(item)] == [f"{item.id}.f1", f"{item.id}.f2"]
        assert [s.label for s in tasks.ask_steps(item)] == [f"{item.id}.q1", f"{item.id}.q2"]
        first, second = item.asks
        assert not any(score.contains(second.text, t) for t in first.targets)  # Q2 says "her", not the name
        assert not any(score.contains(item.facts[0], t) for t in second.targets)  # the answer to Q2 is only in the second fact


def test_order_is_a_fixed_shuffle(tasks):
    order = tasks.order()
    assert sorted(i.id for i in order) == sorted(i.id for i in tasks.items) and order == tasks.order()
    assert [i.id for i in order][:3] == ["L12", "L13", "P13"]  # changes only when ask_seed or an id does


def test_public_text_has_no_home_paths():
    text = data_v2.DATA_FILE.read_text(encoding="utf-8")
    assert "/Users/" not in text and "~/" not in text


def test_the_mini_set_loads_without_the_counts_of_the_shipped_file():
    mini = data_v2.parse(mini_raw_v2(), None)
    assert [i.kind for i in mini.items] == ["longrange", "longrange", "followup", "followup"]
    with pytest.raises(data.DataError):
        data_v2.parse(mini_raw_v2())  # the shipped counts do not hold for it


def test_refuses_a_gold_string_in_the_filler_a_fixture_or_a_fixture_path():
    def in_filler(d):
        d["filler"][0]["text"] = "Please read todo.md and tell me about river. Answer in one sentence."
    refuses(in_filler, "L01.q1: 'river' occurs in a filler message")

    def in_fixture(d):
        d["fixtures"][0]["text"] = "the 4B room\n"
    refuses(in_fixture, "L02.q1: '4B' occurs in a filler message or a fixture file")

    def in_path(d):
        d["fixtures"][1]["path"] = "Dana.txt"
    refuses(in_path, "L09.q1: 'Dana' occurs in a filler message or a fixture file")

    def in_opener(d):
        d["items"][item_index(d, "L04")]["opener"] = "Good morning, code 7421 again."
    refuses(in_opener, "item L04: the answer is in the opener")


def test_refuses_a_fixture_that_is_too_big_or_leaves_the_workspace():
    refuses(lambda d: d["fixtures"][0].update(text="x" * 301), "over 300")
    refuses(lambda d: d["fixtures"][0].update(path="../escape.txt"), "stay inside the workspace")


def test_refuses_filler_that_is_not_read_tier_or_not_short():
    refuses(lambda d: d["filler"][0].update(tool="file_write"), "not a read-tier tool")
    refuses(lambda d: d["filler"][0].update(tool="git_status"), "not a read-tier tool")
    refuses(lambda d: d["filler"][0].update(text="Please read todo.md and describe it at length."), "ask for a short answer")
    refuses(lambda d: d["filler"][0].update(text="Please summarise it. Answer in one sentence."), "must name one of the fixture files")


def test_refuses_a_filler_set_that_is_short_or_mostly_plain():
    refuses(lambda d: d["sets"]["en_a"].__delitem__(slice(0, 6)), "needs at least 35 turns")

    def plain_only(d):
        d["sets"]["en_b"] = [i["id"] for i in d["filler"] if i["lang"] == "en" and "tool" not in i] * 2
    refuses(plain_only, "at least half of the turns must be tool turns")


def test_refuses_a_filler_set_under_the_floor_of_two_messages_a_turn_and_accepts_it_at_the_floor():
    """34 turns of mostly tools are 112 messages with tools, over the aim, but 68 when the model calls none: under 70."""
    def tools_first(d, plain):
        ids = d["sets"]["en_a"]
        d["sets"]["en_a"] = [i for i in ids if i.startswith("ft")] + [i for i in ids if i.startswith("fp")][:plain]

    refuses(lambda d: tools_first(d, 12), "34 turns are 68 messages at 2 per turn, the floor is 70 (the 64-message window plus 6): needs at least 35 turns")
    doc = raw()
    tools_first(doc, 13)  # 35 turns: 70 messages at 2 per turn, 114 with the tools
    assert len(data_v2.parse(doc).sets["en_a"]) == 35
    assert data_v2.FILLER_FLOOR == 70 and data_v2.MIN_FILLER_TURNS == 35


def test_refuses_a_chat_too_short_to_push_the_fact_out_of_the_window():
    def shorten(d):
        d["sets"]["en_a"] = [i for i in d["sets"]["en_a"] if i.startswith("fp")][:14] + [i for i in d["sets"]["en_a"] if i.startswith("ft")][:14]
    refuses(shorten, "messages between the fact and the question")


def test_refuses_reject_strings_in_the_wrong_place():
    refuses(lambda d: d["items"][0]["asks"][0].update(reject=["river"]), "reject 'river' is in the gold text")
    refuses(lambda d: d["items"][0]["asks"][0].update(reject=["nowhere"]), "no reject string is in the wrong text")
    refuses(lambda d: d["items"][item_index(d, "P01")]["asks"][1].update(reject=["9C"]) or d["items"][item_index(d, "P01")].update(wrong=[]),
            "reject strings need wrong items that hold them")
    refuses(lambda d: d["items"][item_index(d, "P01")].update(wrong=["P01"]), "is not another followup item")


def test_refuses_answers_that_are_not_in_their_text():
    refuses(lambda d: d["items"][0]["asks"][0].update(answer="nowhere"), "is not in")
    refuses(lambda d: d["items"][item_index(d, "P01")]["asks"][1].update(answer="999"), "P01.q2: answer '999' is not in")


def test_refuses_a_followup_that_is_not_a_pronoun_followup():
    refuses(lambda d: d["items"][item_index(d, "P01")]["asks"][1].update(text="And where is Ortiz's office?"), "not a pronoun follow-up")
    refuses(lambda d: d["items"][item_index(d, "P01")].update(facts=["My advisor is Dr. Lena Ortiz in room 512.", "Dr. Ortiz's office is room 512."]), "already in the first fact")


def test_refuses_bad_shapes_and_counts():
    refuses(lambda d: d["items"].pop(), "items per kind")
    refuses(lambda d: d["items"][0].pop("note_expected"), "missing keys ['note_expected']")
    refuses(lambda d: d["items"][0].update(color="red"), "unknown keys ['color']")
    refuses(lambda d: d["items"][0].update(kind="temporal"), "kind 'temporal'")
    refuses(lambda d: d["items"][0].update(filler="nope"), "filler set 'nope' does not exist")
    refuses(lambda d: d["items"][1].update(id="L01"), "duplicate ids")
    refuses(lambda d: d["items"][0].update(lang="fr"), "lang 'fr'")
    refuses(lambda d: d["items"][0]["asks"].append(copy.deepcopy(d["items"][0]["asks"][0])), "has 1 ask(s)")
    refuses(lambda d: [i.update(note_expected=True) for i in d["items"] if i["kind"] == "longrange"], "note_expected: false")
    refuses(lambda d: [i.update(lang="en") for i in d["items"] if i["kind"] == "followup"], "Chinese")
    refuses(lambda d: d.update(version=1), "version must be 2")
    refuses(lambda d: d.pop("sets"), "top-level keys")
