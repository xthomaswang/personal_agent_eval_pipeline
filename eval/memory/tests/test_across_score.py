"""The across-chats scorer: reply grading and recall facts."""
import pytest
from memory.across import data, score


def question(kind="exact", answer="21 June", aliases=(), reject=()):
    return data.Question("Q", kind, "en", "text", () if kind == "nothing" else ("F",), answer, tuple(aliases), tuple(reject))


def test_normalize_folds_case_accents_and_punctuation():
    assert score.normalize("  Café  Marigold! ") == "cafe marigold"
    assert score.normalize("I’m 10:30") == "i m 10 30"
    assert score.normalize("３月９日，ＡＢＣ") == "3月9日 abc"


def test_ascii_targets_match_whole_words_only():
    assert score.contains("Your locker is number 214.", "214")
    assert not score.contains("Your locker is number 214.", "14")
    assert not score.contains("a ThinkPadX", "ThinkPad")
    assert score.contains("It is Cafe Marigold.", "café marigold")
    assert score.contains("On 21 June at 10:30", "10:30")
    assert score.contains("你的信箱是三楼的18号", "18")


def test_cjk_targets_match_as_substrings_without_spaces():
    assert score.contains("你的室友是 小满。", "小满")
    assert score.contains("她的生日是 3月9日", "3月9")
    assert not score.contains("你的室友是小明", "小满")
    assert not score.contains("", "x") and not score.contains("x", "")


def test_strip_thinking():
    assert score.strip_thinking("<think>maybe 5</think>It is 6.") == "It is 6."
    assert score.strip_thinking("hidden reasoning</think>The answer.") == "The answer."
    assert score.strip_thinking("<think>never closed, 21 June") == ""
    assert score.strip_thinking(None) == ""


def test_grade_correct_wrong_empty_and_alias():
    q = question(aliases=("June 21",), reject=())
    assert score.grade(q, "Your wedding anniversary is on 21 June.").outcome == score.CORRECT
    assert score.grade(q, "That's June 21st.").outcome == score.WRONG  # "21st" is another word: list it as an alias
    assert score.grade(q, "It is on June 21.").matched == "June 21"
    assert score.grade(q, "It is in March.").outcome == score.WRONG
    assert score.grade(q, "").outcome == score.EMPTY
    assert score.grade(q, "<think>21 June</think>").outcome == score.EMPTY
    assert score.grade(q, "<think>21 June?</think>It is in March.").outcome == score.WRONG  # thinking is not an answer


def test_grade_abstain_is_not_correct_for_an_answerable_question():
    g = score.grade(question(), "I don't have that in my notes.")
    assert (g.outcome, g.correct, g.abstained) == (score.ABSTAIN, False, True)
    assert score.grade(question(), "我不知道你的纪念日。").outcome == score.ABSTAIN
    # stating the answer wins over a hedge
    assert score.grade(question(), "I don't recall exactly, but it's 21 June.").correct


def test_grade_superseded_needs_the_new_value_alone():
    q = question("superseded", "Utrecht", reject=("Lisbon",))
    assert score.grade(q, "You live in Utrecht.").correct
    assert score.grade(q, "You live in Lisbon.").outcome == score.WRONG
    mixed = score.grade(q, "You moved from Lisbon to Utrecht.")
    assert (mixed.outcome, mixed.correct, mixed.rejected) == (score.MIXED, False, ("Lisbon",))


def test_grade_nothing_to_recall():
    q = question("nothing", reject=("genmaicha",))
    assert score.grade(q, "I don't know your favourite board game.").correct
    assert score.grade(q, "You haven't told me that yet.").correct
    assert score.grade(q, "你还没告诉我这件事。").correct
    assert score.grade(q, "Your favourite is Catan.").outcome == score.WRONG
    assert not score.grade(q, "I don't know, but you like genmaicha.").correct  # a reject string was used
    assert score.grade(q, "").outcome == score.EMPTY


@pytest.mark.parametrize("phrase", ["I couldn't find anything", "no record of that", "I'm not sure", "没有记录", "不记得了", "没有告诉过我"])
def test_abstention_phrases(phrase):
    assert score.abstained(phrase)


def used(*sessions):
    return {"used": [{"kind": "record", "id": i, "text": "t", "source": "s", "session_id": s, "turn": 1, "at": 0} for i, s in enumerate(sessions)]}


def test_recall_hit_at_k_and_rank():
    r = score.recall_score(used("a", "g", "b"), ["g"], ["w"], ["a", "g", "b", "w"])
    assert (r["hit_at_1"], r["hit_at_3"], r["hit_any"], r["gold_rank"]) == (False, True, True, 2)
    assert r["n_used"] == 3 and r["other_taught"] == 2 and not r["wrong_recall"]
    assert score.recall_score(used("g"), ["g"])["hit_at_1"]
    miss = score.recall_score(used("a", "b", "c", "g"), ["g"], taught_sessions=["a", "b", "c", "g"])
    assert (miss["hit_at_3"], miss["hit_at_5"], miss["gold_rank"]) == (False, True, 4)


def test_recall_with_no_event_or_no_gold_chat_is_a_miss():
    r = score.recall_score(None, ["g"])
    assert (r["n_used"], r["hit_at_3"], r["hit_any"], r["gold_rank"]) == (0, False, False, None)
    assert not score.recall_score(used("g"), [])["hit_any"]


def test_recall_wrong_and_other_chats():
    r = score.recall_score(used("w", "q9"), ["g"], ["w"], ["g", "w"])
    assert r["wrong_recall"] and r["n_taught"] == 1 and r["n_other"] == 1  # q9 is not a teaching chat


def test_recall_false_recall_on_nothing():
    assert score.recall_score(None, [], nothing=True)["false_recall"] is False
    r = score.recall_score(used("q9"), [], taught_sessions=["a"], nothing=True)
    assert (r["false_recall"], r["false_recall_taught"]) == (True, False)
    assert score.recall_score(used("a"), [], taught_sessions=["a"], nothing=True)["false_recall_taught"]
    assert "hit_at_3" not in r
