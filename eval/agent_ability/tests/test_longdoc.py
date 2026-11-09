"""The long-document case: the same register for the same seed, one answer per section that only that section
gives, and a score that counts a section only when its line names the right person and no other of that section."""

from __future__ import annotations

from pathlib import Path

import pytest
from marks import needs_data

from smoke import checks, longdoc, tasks


def _digest(facts: list[longdoc.Fact]) -> str:
    return "".join(f"Section {f.section}: {f.closer}\n" for f in facts)


def test_the_register_is_the_same_for_the_same_seed() -> None:
    text, facts = longdoc.document(sections=12)
    assert (text, facts) == longdoc.document(sections=12)
    assert text != longdoc.document(seed=longdoc.SEED + 1, sections=12)[0]
    assert text.count("\n## Section ") == 12 and [f.section for f in facts] == list(range(1, 13))


def test_every_name_is_in_one_section_only_and_no_surname_twice() -> None:
    text, facts = longdoc.document(sections=longdoc.MAX_SECTIONS)
    people = [name for f in facts for name in (f.closer, *f.others)]
    surnames = [name.split()[1] for name in people]
    assert len(set(surnames)) == len(surnames) == 3 * longdoc.MAX_SECTIONS
    sections = text.split("\n## Section ")[1:]
    assert len(sections) == longdoc.MAX_SECTIONS
    for fact, body in zip(facts, sections, strict=True):
        assert body.startswith(f"{fact.section}: {fact.place}\n")
        for name in (fact.closer, *fact.others):
            assert body.count(name) == 1 and text.count(name.split()[1]) == 1
    with pytest.raises(ValueError, match="between 1 and"):
        longdoc.document(sections=longdoc.MAX_SECTIONS + 1)


def test_the_answer_is_not_at_a_fixed_place_or_in_fixed_words() -> None:
    text, facts = longdoc.document(sections=60)
    sections = text.split("\n## Section ")[1:]
    wordings, places = set(), set()
    for fact, body in zip(facts, sections, strict=True):
        sentence = next(s for s in body.replace("\n", " ").split(". ") if fact.closer in s)
        wordings.add(sentence.replace(fact.closer, "{name}").strip())
        places.add(round(body.index(fact.closer) / len(body), 1))
    assert len(wordings) >= 6 and len(places) >= 6
    # the words of the role sentences are also words of other sentences: a search for one of them is not the answer list
    assert text.count("closed") > sum("closed" in w for w in wordings) and text.count("finished") > 10


def test_score_counts_right_wrong_and_missing() -> None:
    _text, facts = longdoc.document(sections=10)
    assert longdoc.score(_digest(facts), facts) | {"wrong_sections": [], "missing_sections": []} == {
        "sections": 10, "correct": 10, "wrong": 0, "missing": 0, "coverage": 1.0, "furthest": 10,
        "wrong_sections": [], "missing_sections": [], "lines_of_no_section": 0,
    }  # fmt: skip
    lines = [
        f"Section 1: {facts[0].closer}",
        f"- **Section 2**: {facts[1].closer.split()[1]} closed it",  # a list mark, bold and the surname alone
        f"section 3 - {facts[2].closer.upper()}",
        f"Section 4: {facts[3].closer}, {facts[3].others[0]}",  # every name of the section: not an answer
        f"Section 5: {facts[4].others[1]}",  # the witness
        f"Section 6: {facts[6].closer}",  # another section's person
        f"Section 7: {facts[6].others[0]}",
        f"Section 7: {facts[6].closer}",  # corrected below the first try: the last line counts
        f"Section 44: {facts[0].closer}",
        "Sections 8 to 10 were not read.",
    ]
    score = longdoc.score("\n".join(lines), facts)
    assert (score["correct"], score["wrong"], score["missing"]) == (4, 3, 3)
    assert score["wrong_sections"] == [4, 5, 6] and score["missing_sections"] == [8, 9, 10]
    assert score["coverage"] == 0.4 and score["furthest"] == 7 and score["lines_of_no_section"] == 1
    assert longdoc.score("", facts)["missing"] == 10


def test_a_surname_inside_another_word_is_not_a_match() -> None:
    fact = longdoc.Fact(1, "a place", "Maren Oskic", ("Tobias Belrow", "Ilse Zorke"))
    assert longdoc.score("Section 1: Maren Oskicson", [fact])["wrong"] == 1
    assert longdoc.score("Section 1: Maren Oskic.", [fact])["correct"] == 1


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
