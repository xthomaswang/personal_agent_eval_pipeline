"""Scoring (DESIGN.md, section 7): every step, each kind, the tolerance edges, hedges, and an oracle over the pilot rows."""

from __future__ import annotations

import pytest
from web import data, score
from web.score import grade, matches, normalize_str, prenormalize
from web.score import score as score_answer


def gold_check(answer: str, gold: str, kind: str, *, unordered: bool = False) -> bool:
    return matches(answer, gold, kind, unordered=unordered)


# -- pre-normalization ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rafael Benítez", "Rafael Benitez"),  # NFKD fold: accents dropped
        ("Beńitez", "Benitez"),  # combining accent already separate
        ("ﬁnal", "final"),  # compatibility ligature
        ("  Delhi \n", "Delhi"),  # trim, including a no-break space
        (" Delhi ", "Delhi"),
        ("Delhi.", "Delhi"),  # one trailing period
        ("Delhi . ", "Delhi"),
        ("Delhi..", "Delhi."),  # only one
        ("The Mixers", "Mixers"),  # leading "the "
        ("the Mixers", "Mixers"),
        ("THE   Mixers", "Mixers"),
        ("The Mixers.", "Mixers"),  # period first, then "the"
        ("  The Hague. ", "Hague"),
        ("Theodore", "Theodore"),  # not a word of its own
        ("There", "There"),
        ("Hague, The", "Hague, The"),  # only a leading one
        ("the", "the"),  # nothing follows, so it stays
        ("", ""),
        ("12.5.", "12.5"),
    ],
)
def test_prenormalize(raw: str, expected: str) -> None:
    assert prenormalize(raw) == expected


def test_prenormalization_applies_to_gold_and_answer_alike() -> None:
    assert matches("Mixers", "The Mixers", "string")
    assert matches("the mixers.", "The Mixers", "string")
    assert matches("Rafael Benitez", "Rafael Benítez", "string")
    assert matches("Rafael Benítez", "Rafael Benitez", "string")


# -- string -----------------------------------------------------------------------------------------------------


def test_normalize_str_is_gaias() -> None:
    assert normalize_str("Sea Gull!") == "seagull"
    assert normalize_str(" St. Louis,  MO ") == "stlouismo"
    assert normalize_str("A\tB\nC") == "abc"
    assert normalize_str("St. Louis", remove_punct=False) == "st.louis"
    assert normalize_str("O’Brien") == "obrien"  # non-ASCII punctuation goes too


@pytest.mark.parametrize(
    "answer",
    ["Bettino Ricasoli", "bettino ricasoli", "BETTINO  RICASOLI.", "Bettino-Ricasoli", "“Bettino Ricasoli”"],
)
def test_string_accepts_formatting_differences(answer: str) -> None:
    assert gold_check(answer, "Bettino Ricasoli", "string")


@pytest.mark.parametrize(
    "answer",
    [
        "Camillo Cavour",
        "Ricasoli",
        "Bettino Ricasoli or Camillo Cavour",
        "Bettino Ricasoli (Iron Baron)",
        "Baron Bettino Ricasoli",
        "Bettino",
        "",
        "   ",
        ".",
    ],
)
def test_string_is_whole_answer_equality(answer: str) -> None:
    assert not gold_check(answer, "Bettino Ricasoli", "string")


def test_string_digits_and_symbols() -> None:
    assert gold_check("7,034", "7034", "string")  # a comma is punctuation
    assert not gold_check("7035", "7034", "string")


# -- number -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    ["$1,234", "1234", "1,234", "1234.0", " 1,234 ", "$ 1234", "1234.", "+1234", "1234 dollars", "1,234 USD"],
)
def test_number_strips_dollar_percent_and_commas(answer: str) -> None:
    assert gold_check(answer, "1234", "number")


def test_percent_sign_is_dropped() -> None:
    assert gold_check("5%", "5", "number")
    assert gold_check("5 %", "5", "number")
    assert gold_check("5", "5%", "number")
    assert not gold_check("6%", "5", "number")


def test_number_with_a_unit() -> None:
    assert gold_check("41 years", "41", "number")
    assert gold_check("41years", "41", "number")
    assert gold_check("41 years old", "41", "number")
    assert gold_check("10 yrs.", "10", "number")
    assert gold_check("15th", "15", "number")
    assert not gold_check("41 years old, give or take", "41", "number")  # more than three unit words
    assert not gold_check("42 years", "41", "number")


def test_number_negative_and_decimal() -> None:
    assert gold_check("-5", "-5", "number")
    assert gold_check("−5", "-5", "number")
    assert not gold_check("5", "-5", "number")
    assert gold_check("0.5", "0.5", "number")
    assert gold_check(".5", "0.5", "number")
    assert not gold_check("0.6", "0.5", "number")


@pytest.mark.parametrize(
    ("gold", "answer", "ok"),
    [
        # Up to 50: exact.
        ("50", "50", True),
        ("50", "50.0", True),
        ("50", "51", False),
        ("50", "49", False),
        ("50", "50.1", False),
        ("2", "3", False),
        ("2", "1", False),
        ("41", "40", False),
        ("41", "42", False),
        # Above 50, up to 10,000: within 1% of the gold.
        ("51", "51", True),
        ("51", "51.51", True),  # 51 * 1.01, the edge
        ("51", "51.52", False),
        ("51", "50.49", True),
        ("51", "50.48", False),
        ("51", "52", False),
        ("100", "101", True),
        ("100", "99", True),
        ("100", "102", False),
        ("100", "98", False),
        ("10000", "10000", True),
        ("10000", "10,100", True),  # the edge
        ("10000", "10,101", False),
        ("10000", "9,900", True),
        ("10000", "9,899", False),
        ("7034", "7035", True),
        ("7034", "7104", True),
        ("7034", "7105", False),
        # Above 10,000: within 5%.
        ("10001", "10501", True),  # 500 <= 500.05
        ("10001", "10502", False),
        ("20000", "21,000", True),  # the edge
        ("20000", "21,001", False),
        ("20000", "19,000", True),
        ("20000", "18,999", False),
        ("20000", "$21,000", True),
        ("1000000", "1,050,000", True),
        ("1000000", "1,050,001", False),
        # The tolerance follows the gold, not the answer.
        ("101", "100", True),
        ("100", "101.01", False),
    ],
)
def test_number_tolerance_edges(gold: str, answer: str, ok: bool) -> None:
    assert gold_check(answer, gold, "number") is ok


def test_tolerance_function() -> None:
    from fractions import Fraction

    assert score.tolerance(Fraction(50)) == 0
    assert score.tolerance(Fraction(51)) == Fraction(51, 100)
    assert score.tolerance(Fraction(10_000)) == 100
    assert score.tolerance(Fraction(10_001)) == Fraction(10_001, 20)
    assert score.tolerance(Fraction(-20_000)) == 1000


@pytest.mark.parametrize(
    "answer",
    [
        "12 or 13",
        "12 or thirteen",
        "12 to 13",
        "12-13",
        "12–13",
        "12/13",
        "12, 13",
        "12 and 13",
        "between 12 and 13",
        "about 12",
        "approximately 12",
        "~12",
        "12 or so",
        "12 or more",
        "12 (maybe 13)",
        "at least 12",
        "twelve",
        "12 13",
        "1e1+2",
        "12.5.3",
        "",
        "none",
        "NaN",
        "inf",
    ],
)
def test_number_hedges_and_non_numbers_fail(answer: str) -> None:
    assert not gold_check(answer, "12", "number")
    assert not gold_check(answer, "13", "number")


# -- year -------------------------------------------------------------------------------------------------------


def test_year_is_exact() -> None:
    assert gold_check("1987", "1987", "year")
    assert gold_check("1987.", "1987", "year")
    assert gold_check(" 1987 ", "1987", "year")
    for neighbour in ("1986", "1988", "1968", "1978", "1887", "1789", "19870"):
        assert not gold_check(neighbour, "1987", "year")


@pytest.mark.parametrize(
    "answer",
    [
        "1987 or 1988",
        "1987/88",
        "1987-88",
        "in 1987",
        "c. 1987",
        "late 1987",
        "March 1987",
        "the year 1987",
        "",
    ],
)
def test_year_must_be_the_bare_year(answer: str) -> None:
    assert not gold_check(answer, "1987", "year")


# -- date -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "3 March 1962",
        "March 3, 1962",
        "March 3 1962",
        "3rd March 1962",
        "3rd of March, 1962",
        "the 3rd of March 1962",
        "03 Mar 1962",
        "Mar. 3, 1962",
        "1962 March 3",
        "1962, 3 March",
        "1962-03-03",
        "1962-3-3",
        "3 MARCH 1962.",
        "3 march 1962",
    ],
)
def test_date_in_any_order_or_iso(answer: str) -> None:
    assert gold_check(answer, "3 March 1962", "date")
    assert gold_check(answer, "1962-03-03", "date")


@pytest.mark.parametrize(
    "answer",
    [
        "4 March 1962",
        "3 April 1962",
        "3 March 1963",
        "3 March 1926",
        "1962-03-04",
        "1962-04-03",
        "1963-03-03",
        "March 1962",  # no day
        "1962",  # no month or day
        "3 March",  # no year
        "3 March 1962 or 1963",
        "March 3 or 4, 1962",
        "3-4 March 1962",
        "3 March 1962 and 4 March 1962",
        "3 3 1962",  # two days, no month
        "3 March March 1962",
        "3 Marchy 1962",
        "early March 1962",
        "about 3 March 1962",
        "1962-13-03",
        "1962-03-32",
        "3/3/1962",  # numeric forms other than ISO are ambiguous
        "",
    ],
)
def test_date_wrong_or_hedged_fails(answer: str) -> None:
    assert not gold_check(answer, "3 March 1962", "date")


def test_date_checks_only_the_components_the_gold_has() -> None:
    assert gold_check("March 1962", "March 1962", "date")
    assert gold_check("3 March 1962", "March 1962", "date")  # more detail than asked is fine
    assert gold_check("1962-03", "March 1962", "date")
    assert gold_check("1962-03-03", "March 1962", "date")
    assert not gold_check("April 1962", "March 1962", "date")
    assert not gold_check("March 1963", "March 1962", "date")
    assert not gold_check("1962", "March 1962", "date")
    assert gold_check("1962", "1962", "date")
    assert gold_check("March 1962", "1962", "date")
    assert not gold_check("1963", "1962", "date")
    assert gold_check("5 September 1990", "Sept 5, 1990", "date")
    assert gold_check("May 5, 1990", "5 May 1990", "date")


def test_date_gold_without_a_date_matches_nothing() -> None:
    assert not gold_check("March 1962", "sometime", "date")
    assert not gold_check("sometime", "sometime", "date")


# -- list -------------------------------------------------------------------------------------------------------


def test_list_in_order() -> None:
    gold = "red, green, blue"
    for answer in (
        "red, green, blue",
        "red,green,blue",
        "Red; Green; Blue",
        "red, green, blue.",
        " RED , green ;blue ",
    ):
        assert gold_check(answer, gold, "list")
    for answer in (
        "blue, green, red",
        "red, green",
        "red, green, blue, yellow",
        "red, green, yellow",
        "red green blue",
        "",
    ):
        assert not gold_check(answer, gold, "list")


def test_list_unordered_is_a_multiset() -> None:
    gold = "red, green, blue"
    assert gold_check("blue, red, green", gold, "list", unordered=True)
    assert gold_check("green; blue; red.", gold, "list", unordered=True)
    assert not gold_check("blue, red", gold, "list", unordered=True)
    assert not gold_check("blue, red, red", gold, "list", unordered=True)
    assert not gold_check("red, green, blue, blue", gold, "list", unordered=True)
    assert gold_check("a, b, a", "a, a, b", "list", unordered=True)
    assert not gold_check("a, b, b", "a, a, b", "list", unordered=True)


def test_list_numbers_are_compared_as_numbers() -> None:
    assert gold_check("1, 2", "1, 2", "list")
    assert gold_check("$1, 2.0", "1, 2", "list")
    assert gold_check("1.0; 2%", "1, 2", "list")
    assert not gold_check("1, 3", "1, 2", "list")
    assert not gold_check("one, two", "1, 2", "list")
    assert gold_check("2, 1", "1, 2", "list", unordered=True)


def test_list_elements_keep_their_punctuation_as_in_gaia() -> None:
    assert gold_check("St. Louis, Rome", "st. louis, rome", "list")
    assert not gold_check("St Louis, Rome", "St. Louis, Rome", "list")


def test_list_drops_a_leading_and() -> None:
    assert gold_check("red, green, and blue", "red, green, blue", "list")
    assert gold_check("red, green, blue", "red, green, and blue", "list")
    assert gold_check("blue, red, and green", "red, green, and blue", "list", unordered=True)
    assert not gold_check("red, green and blue", "red, green, blue", "list")  # that is two elements
    assert not gold_check("red, green, android", "red, green, droid", "list")  # "and" must be a word


def test_the_hegel_row_forms() -> None:
    gold = "Painting, music, and poetry."
    for answer in (
        "Painting, music, and poetry.",
        "Painting, music, poetry",
        "painting; music; poetry",
        "Poetry, music, painting",
        "Music, painting, and poetry",
        "Painting, poetry, and music",
    ):
        assert gold_check(answer, gold, "list", unordered=True), answer
    for answer in (
        "Painting, music",
        "Painting, music, poetry, sculpture",
        "Architecture, sculpture, painting",
        "Painting, music, dance",
        "Painting, music, or poetry",
        "Painting, music and poetry",  # two elements: only an alias covers this form
    ):
        assert not gold_check(answer, gold, "list", unordered=True), answer
    assert not gold_check("Poetry, music, painting", gold, "list")  # ordered would refuse it


# -- aliases, kinds and grades ----------------------------------------------------------------------------------


def test_score_returns_the_matching_target() -> None:
    kwargs = {"gold": "15th place", "kind": "string", "aliases": ("15th", "15")}
    assert score_answer("15th place", **kwargs) == "15th place"
    assert score_answer("15th", **kwargs) == "15th"
    assert score_answer("15", **kwargs) == "15"
    assert score_answer("14th", **kwargs) is None
    assert score_answer("16th place", **kwargs) is None


def test_unknown_kind_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        matches("1", "1", "integer")
    with pytest.raises(ValueError):
        score_answer("1", gold="1", kind="")


def test_kinds_are_the_five_of_the_design() -> None:
    assert score.KINDS == ("number", "year", "date", "string", "list")


def test_grade_from_a_whole_reply() -> None:
    kwargs = {"gold": "1889", "kind": "year"}
    right = grade("I looked it up.\n\n**FINAL ANSWER:** 1889", **kwargs)
    assert (right.verdict, right.answer, right.matched) == ("pass", "1889", "1889")
    wrong = grade("FINAL ANSWER: 1890", **kwargs)
    assert (wrong.verdict, wrong.answer, wrong.matched) == ("fail", "1890", None)
    assert grade("FINAL ANSWER: unknown", **kwargs).verdict == "abstain"
    assert grade("FINAL ANSWER:", **kwargs).verdict == "abstain"
    assert grade("It was 1889.", **kwargs).verdict == "no_answer"
    assert grade("", **kwargs).verdict == "no_answer"
    assert grade(None, **kwargs).verdict == "no_answer"


def test_grade_hedge_in_the_final_line_fails() -> None:
    assert grade("FINAL ANSWER: 1889 or 1890", gold="1889", kind="year").verdict == "fail"
    assert grade("FINAL ANSWER: 12 or 13", gold="12", kind="number").verdict == "fail"
    assert grade("1889 or 1890\nFINAL ANSWER: 1889", gold="1889", kind="year").verdict == "pass"


# -- the oracle over the 24 pilot rows --------------------------------------------------------------------------

# Plausible wrong answers per row: a neighbouring number or year, a wrong name of the right type, a hedge, a
# partial name where no alias allows it.
DECOYS: dict[int, list[str]] = {
    3503: [
        "P. W. Botha",
        "Gerrit Viljoen",
        "Marais Viljoen or P. W. Botha",
        "Viljoen",
        "Marais",
        "unknown person",
    ],
    1337: [
        "Jacob Temminck",
        "Johann Friedrich Gmelin",
        "Louis Pierre Vieillot",
        "Coenraad Temminck or Gmelin",
    ],
    3531: ["Camillo Cavour", "Urbano Rattazzi", "Giovanni Giolitti", "Ricasoli or Cavour"],
    3010: [
        "Cornell University",
        "Amherst College",
        "University of Syracuse",
        "Syracuse University or Cornell",
    ],
    3561: ["Mumbai", "Haryana", "Delhi or Mumbai", "Chandigarh", "Puducherry"],
    302: ["Camden", "Hackney", "Islington or Camden", "Southwark"],
    3181: ["1", "3", "4", "12", "2 or 3", "two or three", "20", "none"],
    3635: ["7033", "7035", "7043", "7000", "8034", "7034 or 7035", "Executive Order 7035", "EO 8802"],
    2923: ["1961", "1963", "1926", "1972", "1962 or 1963", "in 1962", "1962-63"],
    3372: ["1888", "1890", "1899", "1989", "1889 or 1890"],
    1442: ["Pabuk", "Sepat", "Mun", "Danas", "Typhoon Pabuk", "Wutip or Sepat"],
    3538: [
        "Architecture, sculpture, painting",
        "Painting, music, dance",
        "Painting, music",
        "Painting, music, poetry, sculpture",
        "Sculpture, painting, music",
        "Painting, music or poetry",
    ],
    1934: ["Peter Roth", "Sandy Grushow", "Chris Albrecht", "Matoian", "John Matoian or Chris Albrecht"],
    378: [
        "Kojo Botsio",
        "Krobo Edusei",
        "Kwame Nkrumah",
        "Tawia Adamafio or Kojo Botsio",
        "Adamafio Tawia Kojo",
    ],
    2146: [
        "Charles Jacob",
        "Jacobs",
        "Charles Jacobson",
        "Charles Dickens",
        "Charles Jacobs or Charles Jacob",
    ],
    1963: [
        "Johannesburg",
        "Cape Town",
        "Kimberley, British Columbia",
        "Durban",
        "Kimberly",
        "Kimberley or Cape Town",
    ],
    2849: [
        "Port Perry",
        "Prince Edward",
        "Albert",
        "Prince Albert, Saskatchewan",
        "Prince George",
        "Princeton",
    ],
    1608: ["1", "3", "20", "0", "2 or 3", "two or three", "22"],
    2783: ["40", "42", "14", "41 or 42", "4", "410", "30"],
    323: ["9", "11", "1", "100", "10 or 11", "20", "tenth"],
    3157: ["1976", "1978", "1987", "1797", "1977 or 1978", "1977-78"],
    1059: [
        "The Mixer",
        "The Dead Mixers",
        "The Pretty Flowers",
        "The Mixers or The Beatles",
        "Mixers Anonymous",
    ],
    69: [
        "14th place",
        "16th place",
        "17th",
        "14",
        "16",
        "15th or 16th",
        "fifteen",
        "150th",
        "5th place",
        "15th place or 16th",
    ],
    3352: ["2014", "2124", "6061", "7075", "Al 2014", "Al 6061", "2024 or 6061", "2025", "20240", "2,014"],
}


@pytest.fixture(scope="module")
def pilot() -> data.Pilot:
    return data.load()


def test_oracle_has_decoys_for_every_row(pilot: data.Pilot) -> None:
    assert sorted(DECOYS) == sorted(r.id for r in pilot.rows)
    assert all(len(DECOYS[r.id]) >= 3 for r in pilot.rows)


def test_oracle_every_gold_and_alias_passes(pilot: data.Pilot) -> None:
    for row in pilot.rows:
        for target in row.targets:
            assert score_answer(target, **row.scoring()) is not None, (row.id, target)


def test_oracle_every_gold_and_alias_passes_through_a_reply(pilot: data.Pilot) -> None:
    for row in pilot.rows:
        for target in row.targets:
            for reply in (
                f"FINAL ANSWER: {target}",
                f"I searched and found it on the second page.\n\n**FINAL ANSWER:** {target}\n",
                f"final answer: {target.upper()}" if row.kind == "string" else f"Final answer: {target}",
            ):
                result = grade(reply, **row.scoring())
                assert result.verdict == "pass", (row.id, reply)


def test_oracle_decoys_fail(pilot: data.Pilot) -> None:
    for row in pilot.rows:
        for decoy in DECOYS[row.id]:
            assert score_answer(decoy, **row.scoring()) is None, (row.id, decoy)
            assert grade(f"FINAL ANSWER: {decoy}", **row.scoring()).verdict == "fail", (row.id, decoy)


def test_oracle_abstentions_and_missing_lines_never_pass(pilot: data.Pilot) -> None:
    for row in pilot.rows:
        assert grade("FINAL ANSWER: unknown", **row.scoring()).verdict == "abstain"
        assert grade(f"The answer is {row.answer}", **row.scoring()).verdict == "no_answer"


def test_oracle_decoys_are_not_aliases(pilot: data.Pilot) -> None:
    for row in pilot.rows:
        normalized = {normalize_str(prenormalize(t)) for t in row.targets}
        assert not normalized & {normalize_str(prenormalize(d)) for d in DECOYS[row.id]}, row.id


def test_oracle_specific_rows(pilot: data.Pilot) -> None:
    def ok(original_index: int, answer: str) -> bool:
        row = pilot.by_id(original_index)
        return score_answer(answer, **row.scoring()) is not None

    assert ok(69, "15th") and ok(69, "15") and ok(69, "Fifteenth place") and not ok(69, "16th")
    assert (
        ok(3352, "2024") and ok(3352, "Al-2024") and ok(3352, "aluminium alloy 2024") and not ok(3352, "2014")
    )
    assert (
        ok(2849, "Prince Albert")
        and ok(2849, "Prince Albert, Ontario")
        and not ok(2849, "Prince Albert, Saskatchewan")
    )
    assert ok(3538, "Poetry, painting, and music") and not ok(3538, "Painting, music")
    assert ok(1059, "Mixers") and ok(1059, "The Mixers") and not ok(1059, "The Mixer")
    assert ok(3635, "7034") and ok(3635, "EO 7034") and not ok(3635, "7035")
    assert ok(2783, "41 years") and not ok(2783, "42")
    assert ok(2923, "1962") and not ok(2923, "1963")
