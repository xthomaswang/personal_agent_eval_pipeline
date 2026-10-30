"""Answer extraction from synthetic agent replies: the marker, markup, the last line, abstain and no_answer."""

from __future__ import annotations

import pytest
from web import answer
from web.answer import ABSTAIN, ANSWER, NO_ANSWER, extract


def test_plain_final_line() -> None:
    result = extract("I checked three pages.\n\nFINAL ANSWER: Delhi")
    assert result == answer.Extraction(ANSWER, "Delhi")


@pytest.mark.parametrize("marker", ["FINAL ANSWER:", "Final Answer:", "final answer:", "FiNaL   AnSwEr :"])
def test_marker_is_case_insensitive(marker: str) -> None:
    assert extract(f"{marker} 1889") == answer.Extraction(ANSWER, "1889")


def test_last_marker_line_wins() -> None:
    reply = "FINAL ANSWER: 1961\nOn a second look the page says 1962.\nFINAL ANSWER: 1962\n"
    assert extract(reply).answer == "1962"


def test_a_quoted_format_earlier_in_the_reply_does_not_hide_the_real_line() -> None:
    reply = "You asked me to end with FINAL ANSWER: <answer>. Here it is.\nFINAL ANSWER: Wutip"
    assert extract(reply).answer == "Wutip"


def test_text_after_the_answer_line_is_ignored() -> None:
    assert extract("FINAL ANSWER: Wutip\n\nThanks for asking!").answer == "Wutip"


@pytest.mark.parametrize(
    "reply",
    [
        "**FINAL ANSWER:** Delhi",
        "**FINAL ANSWER**: Delhi",
        "**FINAL ANSWER: Delhi**",
        "FINAL ANSWER: **Delhi**",
        "__FINAL ANSWER:__ Delhi",
        "_FINAL ANSWER: Delhi_",
        "`FINAL ANSWER: Delhi`",
        "FINAL ANSWER: `Delhi`",
        "> FINAL ANSWER: Delhi",
        "- FINAL ANSWER: Delhi",
        "* FINAL ANSWER: Delhi",
        "1. FINAL ANSWER: Delhi",
        "### FINAL ANSWER: Delhi",
        "<b>FINAL ANSWER:</b> Delhi",
        "<strong>Final answer:</strong> <em>Delhi</em>",
        "FINAL ANSWER: [Delhi](https://en.wikipedia.org/wiki/Delhi)",
        "```\nFINAL ANSWER: Delhi\n```",
        "FINAL ANSWER: ~~Mumbai~~ Delhi",
        "  FINAL ANSWER: Delhi  \r\n",
        'FINAL ANSWER: "Delhi"',
        "FINAL ANSWER: “Delhi”",
        "FINAL ANSWER: 'Delhi'",
    ],
)
def test_markup_is_stripped(reply: str) -> None:
    result = extract(reply)
    assert result.status == ANSWER
    assert result.answer.replace("Mumbai ", "") == "Delhi"


def test_quotes_inside_an_answer_stay() -> None:
    assert extract('FINAL ANSWER: "Delhi" and "Mumbai"').answer == '"Delhi" and "Mumbai"'
    assert extract("FINAL ANSWER: 'Allo 'Allo").answer == "'Allo 'Allo"


def test_underscores_inside_a_word_stay() -> None:
    assert extract("FINAL ANSWER: snake_case_name").answer == "snake_case_name"


def test_colons_and_punctuation_in_the_answer_stay() -> None:
    assert extract("FINAL ANSWER: 12:30, Paris.").answer == "12:30, Paris."
    assert extract("FINAL ANSWER: $1,234").answer == "$1,234"


def test_template_placeholder_is_an_answer_not_an_abstention() -> None:
    result = extract("FINAL ANSWER: <answer>")
    assert result.status == ANSWER
    assert result.answer == "<answer>"


@pytest.mark.parametrize(
    "reply", ["", None, "No line here.", "The final answer is Delhi.", "FINAL ANSWER Delhi"]
)
def test_missing_marker_line_is_no_answer(reply: str | None) -> None:
    assert extract(reply) == answer.Extraction(NO_ANSWER, "")


@pytest.mark.parametrize(
    "reply",
    [
        "FINAL ANSWER: unknown",
        "FINAL ANSWER: Unknown",
        "FINAL ANSWER: UNKNOWN.",
        "FINAL ANSWER: **unknown**",
        "FINAL ANSWER: `unknown`",
        'FINAL ANSWER: "unknown"',
        "FINAL ANSWER:",
        "FINAL ANSWER:    ",
        "**FINAL ANSWER:**",
        "FINAL ANSWER: .",
    ],
)
def test_unknown_or_empty_is_abstain(reply: str) -> None:
    assert extract(reply).status == ABSTAIN


def test_abstain_only_when_the_whole_answer_is_unknown() -> None:
    assert extract("FINAL ANSWER: unknown unknowns").status == ANSWER
    assert extract("FINAL ANSWER: Unknown Soldier").status == ANSWER


def test_a_later_answer_replaces_an_earlier_abstention_and_the_reverse() -> None:
    assert extract("FINAL ANSWER: unknown\nFINAL ANSWER: 12").answer == "12"
    assert extract("FINAL ANSWER: 12\nFINAL ANSWER: unknown").status == ABSTAIN


def test_the_answer_is_one_line_only() -> None:
    assert extract("FINAL ANSWER:\nDelhi").status == ABSTAIN
