"""The long-document case: the same register for the same seed, one incident per section that no search pattern
finds, and a score that counts a section only when its line names that section's incident and not a list of them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from marks import needs_data

from smoke import checks, longdoc, tasks


def _digest(facts: list[longdoc.Fact]) -> str:
    return "".join(f"Section {f.section}: {f.incident}\n" for f in facts)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def test_the_register_is_the_same_for_the_same_seed() -> None:
    text, facts = longdoc.document(sections=12)
    assert (text, facts) == longdoc.document(sections=12)
    assert text != longdoc.document(seed=longdoc.SEED + 1, sections=12)[0]
    assert text.count("\n## Section ") == 12 and [f.section for f in facts] == list(range(1, 13))
    with pytest.raises(ValueError, match="between 1 and"):
        longdoc.document(sections=longdoc.MAX_SECTIONS + 1)


def test_every_incident_is_in_its_section_once_and_its_words_name_nothing_else() -> None:
    text, facts = longdoc.document(sections=longdoc.MAX_SECTIONS)
    assert len({f.incident for f in facts}) == len(facts) == longdoc.MAX_SECTIONS >= 96
    sections = text.split("\n## Section ")[1:]
    for fact, body in zip(facts, sections, strict=True):
        assert body.startswith(f"{fact.section}: {fact.place}\n")
        assert body.count(fact.incident) == 1 and text.count(fact.incident) == 1
        assert fact.keys[0] in _words(fact.incident)  # the first word is the register's own
    # in the whole register, an incident's words are found in its own sentence and in no other: a digest line that
    # repeats routine text, or another section's incident, cannot be taken for it
    sentences = [s.strip() for s in re.split(r"(?<=[.])\s+|\n+", text) if s.strip()]
    for fact in facts:
        assert [s for s in sentences if longdoc.names_it(_words(s), fact.keys)] == [fact.incident], fact.keys


def test_the_incident_is_not_at_a_fixed_place_or_in_a_fixed_form() -> None:
    text, facts = longdoc.document(sections=60)
    sections = text.split("\n## Section ")[1:]
    places = {round(body.index(fact.incident) / len(body), 1) for fact, body in zip(facts, sections, strict=True)}
    assert len(places) >= 8
    starts = {" ".join(sentence.split()[:2]) for sentence, _keys in longdoc.INCIDENTS}
    assert len(starts) >= 60  # hand-written sentences: no opening words to search for


def test_score_counts_right_wrong_and_missing() -> None:
    _text, facts = longdoc.document(sections=10)
    assert longdoc.score(_digest(facts), facts) == {
        "sections": 10, "correct": 10, "wrong": 0, "missing": 0, "coverage": 1.0, "furthest": 10,
        "wrong_sections": [], "missing_sections": [], "lines_of_no_section": 0,
    }  # fmt: skip
    lines = [
        f"Section 1: {facts[0].incident}",
        f"- **Section 2**: the {facts[1].keys[0]} affair",  # a list mark, bold and one word of the incident
        f"section 3 - {facts[2].incident.upper()}",
        f"Section 4: {' / '.join(f.incident for f in facts[3:7])}",  # a list of incidents: not an answer
        "Section 5: 2086 coasters berthed and dues came to 496 marks",  # routine text
        f"Section 6: {facts[6].incident}",  # another section's incident
        f"Section 7: {facts[5].incident}",
        f"Section 7: {facts[6].incident}",  # corrected below the first try: the last line counts
        f"Section 44: {facts[0].incident}",
        "Sections 8 to 10 were not read.",
    ]
    score = longdoc.score("\n".join(lines), facts)
    assert (score["correct"], score["wrong"], score["missing"]) == (4, 3, 3)
    assert score["wrong_sections"] == [4, 5, 6] and score["missing_sections"] == [8, 9, 10]
    assert score["coverage"] == 0.4 and score["furthest"] == 7 and score["lines_of_no_section"] == 1
    assert longdoc.score("", facts)["missing"] == 10


def test_a_word_counts_whole_in_its_plural_or_from_six_letters_by_its_start() -> None:
    assert longdoc.names_it({"cats"}, ("cat",)) and longdoc.names_it({"a", "cat"}, ("cat",))
    assert not longdoc.names_it({"cattle", "catch"}, ("cat",))  # a short word is not a start of longer ones
    assert longdoc.names_it({"photographs"}, ("photograph",)) and longdoc.names_it({"smuggler"}, ("whale", "smuggl"))
    assert not longdoc.names_it({"anchorage"}, ("whale", "bell"))
    fact = longdoc.Fact(1, "a place", "A whale stranded on the bar.", ("whale",))
    other = longdoc.Fact(2, "a place", "The bell cracked.", ("bell",))
    assert longdoc.score("Section 1: a whale, as the bell rang", [fact, other])["correct"] == 1  # one other word is let pass
    assert longdoc.score("Section 1: a whaler came in", [fact, other])["wrong"] == 1


@pytest.mark.parametrize("name", ["register", "register-x", "ledger-24", "24"])
def test_a_task_name_without_a_size_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="unknown long-document task"):
        longdoc.sections_of(name)


@needs_data
def test_the_tasks_are_selectable_and_not_in_the_default_suite() -> None:
    chosen = tasks.select(["longdoc/*"])
    assert [t.id for t in chosen] == [f"longdoc/{n}" for n in tasks.LONGDOC_TASKS]
    assert all(t.kind == "longdoc" and t.scored for t in chosen)
    assert not any(t.kind == "longdoc" for t in tasks.default_suite(extended=True))


def test_the_workspace_holds_the_register_and_no_answers(tmp_path: Path) -> None:
    task = tasks.longdoc_task("register-24")
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)
    assert [p.name for p in ws.iterdir()] == [longdoc.FILE_NAME]
    text, _facts = longdoc.document(sections=24)
    assert (ws / longdoc.FILE_NAME).read_text(encoding="utf-8") == text
    prompt = tasks.prompt(task, ws)
    assert "24 numbered sections" in prompt and longdoc.DIGEST_NAME in prompt and prompt.endswith(tasks.CLOSING.format(ws=ws))
    assert not any(longdoc.names_it(_words(prompt), keys) for _sentence, keys in longdoc.INCIDENTS)  # the example is no incident


def test_the_check_reports_coverage(tmp_path: Path) -> None:
    task = tasks.longdoc_task("register-24")
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)
    _text, facts = longdoc.document(sections=24)

    missing = checks.check(task, ws, tmp_path / "run")
    assert not missing["passed"] and missing["detail"] == "digest.md missing" and not missing["digest_written"]

    (ws / longdoc.DIGEST_NAME).write_text(_digest(facts[:18]), encoding="utf-8")
    part = checks.check(task, ws, tmp_path / "run")
    assert not part["passed"] and part["kind"] == "longdoc" and part["register_untouched"]
    assert (part["correct"], part["missing"], part["coverage"], part["furthest"]) == (18, 6, 0.75, 18)
    assert part["detail"] == "18/24 sections right, 0 wrong, 6 missing"

    (ws / longdoc.DIGEST_NAME).write_text(_digest(facts), encoding="utf-8")
    (ws / longdoc.FILE_NAME).write_text("shortened", encoding="utf-8")
    full = checks.check(task, ws, tmp_path / "run")
    assert full["passed"] and full["coverage"] == 1.0 and not full["register_untouched"]
