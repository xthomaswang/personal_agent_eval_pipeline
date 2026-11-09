"""A long document no model can know, with one checkable fact planted in every section.

A public document (an RFC, a statute) is a poor long-reading case: a model that knows it can write every section's
summary after reading half of it, and the section headings of its output prove nothing. This register is generated from
a seed. Each section names three people with three roles in varied wording, and the task asks for one of them (who
closed the entry). The answer cannot be written without that section's text, and a line that lists every name of a
section does not count. The checker makes the same document again from the seed, so no answer file is ever in the
workspace.

What it does not rule out: a text search finds many of the sentences without a read of the whole register (the role
sentences share no single word, and other sentences use the same words of berths, sheds and accounts, so a search is
noisy, not useless). The score says which facts reached the output; the run's tool events say how they were found.

    document(seed, sections) -> (text, [Fact])
    score(digest_text, facts) -> {"sections", "correct", "wrong", "missing", "coverage", ...}

Only the standard library is used. The text is made with `random.Random(seed)`, whose sequence for a seed is fixed by
Python's documentation for the methods used here (`randrange`, `choice`, `sample`, `shuffle`).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

FILE_NAME = "register.md"
DIGEST_NAME = "digest.md"
SEED = 20261109
SECTIONS = 80

FIRST = (
    "Maren", "Tobias", "Ilse", "Corwin", "Anneke", "Lucan", "Sunniva", "Edrik", "Halla", "Joren", "Petra", "Vasco",
    "Linnea", "Osric", "Tamsin", "Garrick", "Yrsa", "Benno", "Odile", "Fenwick", "Signe", "Lothar", "Mirela", "Caspian",
    "Runa", "Dunstan", "Elske", "Hartwin", "Solveig", "Ansel",
)  # fmt: skip
# A surname is one beginning and one ending: 288 of them, each used at most once in a register, so that a surname alone
# names one person. No beginning is the start of another, so no two pairs spell the same surname.
SURNAME_STARTS = (
    "Osk", "Thar", "Bel", "Quil", "Vander", "Hral", "Pember", "Zor", "Alden", "Fys", "Corr", "Lind", "Massen", "Ubr",
    "Wynd", "Skarr", "Della", "Rons",
)  # fmt: skip
SURNAME_ENDS = ("vik", "row", "marsh", "lon", "see", "dt", "wick", "ic", "moor", "ke", "ane", "qvar", "grave", "echt", "low", "ild")
MAX_SECTIONS = len(SURNAME_STARTS) * len(SURNAME_ENDS) // 3
PLACES = (
    "Saltgate", "Low Fennick", "Marrow Quay", "Thistlewharf", "Coldharrow", "Bexmouth", "Ember Stairs", "Grayling Reach",
    "Tollard Basin", "Hobbwick", "Sundering Lock", "Petrel Yard", "Wren Shoal", "Oxley Cut", "Lantern Row", "Drummore Slip",
    "Kestrel Bar", "Vane Narrows", "Mossgiel Dock", "Arlen Steps",
)  # fmt: skip
CARGO = ("barley", "slate", "salt cod", "lamp oil", "rope", "copper wire", "tallow", "dried kelp", "glass sand", "timber", "wool bales", "ice")
VESSELS = ("lighters", "ketches", "barges", "coasters", "tenders", "dredgers", "cutters", "pilot boats")
WEATHER = ("a north-east gale", "three days of fog", "an early frost", "a spring tide", "steady rain", "a long calm", "a late thaw")
WORKS = (
    "the fender piles on the east face were replaced", "the crane rails were re-laid", "the tide gauge was reset against the stone mark",
    "two bollards were recast", "the lamp standards were moved clear of the rail", "the slip was scraped and tarred",
    "the weighbridge was tested with the standard weights", "the sluice paddles were renewed", "the customs shed roof was relined",
    "the mooring chains were proved to load",
)  # fmt: skip

# The three roles of a section. A section's sentences use one wording of each; only "closer" is asked for.
ROLE_SENTENCES: dict[str, tuple[str, ...]] = {
    "closer": (
        "The entry was closed by {name}.",
        "{name} closed the entry.",
        "Closing of the entry fell to {name}, who initialled the last page.",
        "{name} signed the entry off as finished.",
        "The entry was finished and shut by {name}.",
        "It was {name} who brought the entry to its end.",
        "The last hand in the entry, the one that ended it, is that of {name}.",
        "{name} completed the entry and put the book away.",
    ),
    "opener": (
        "The entry was opened by {name}.",
        "{name} opened the entry at first light.",
        "Opening of the entry was the work of {name}.",
        "The first lines of the entry are in the hand of {name}.",
        "{name} began the entry and set down the state of the berths.",
        "The entry was started by {name}.",
    ),
    "witness": (
        "The count was witnessed by {name}.",
        "{name} stood witness to the count.",
        "As witness to the tallies the board had {name}.",
        "{name} witnessed the weighing and signed beneath.",
        "The witness named for the count is {name}.",
        "{name} was present at the count and attested it.",
    ),
}
FILLER = (
    "Over the quarter {n1} {vessel} berthed at {place}, of which {n2} discharged {cargo} and the rest took on {cargo2}.",
    "Dues collected came to {n3} marks, against {n4} in the same quarter of the year before.",
    "After {weather}, {work}; the cost, {n5} marks, was carried to the works account.",
    "The harbour master reported {n6} soundings along the fairway, the least of them {n7} spans at low water.",
    "A complaint from the {cargo} factors about berth delays was heard and set aside, the delay being put down to {weather}.",
    "The night watch was kept at {n8} men through the quarter, and no loss from the sheds was reported.",
    "Stocks held in bond at the quarter's end were {n9} measures of {cargo} and {n10} of {cargo2}.",
    "The board allowed {n11} marks for the upkeep of the lamps, which burned {n12} nights without a failure.",
    "Pilotage was taken by {n13} vessels inbound; {n14} were brought in without a pilot under the standing exemption.",
    "By order of the board, {work}, the yard foreman answering for the work.",
    "The ferry at {place} ran {n15} crossings, fewer than usual because of {weather}.",
    "A survey of the {vessel} laid up over winter found {n16} fit for the season and {n17} to be broken up.",
    "The east berth at {place} was closed to {vessel} for {n1} days after {weather}.",
    "The bonded shed was opened {n2} times under the collector's key and closed again each night.",
    "The quarter's accounts were finished {n3} days late and signed off by the board in session.",
    "Unloading of the {cargo} was witnessed by the factors' own clerks, who started and ended each shift with a count.",
)
LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?section\s+(\d+)(?:\*\*)?\s*[:.\-–]\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Fact:
    section: int
    place: str
    closer: str  # the answer: the full name of who closed the entry
    others: tuple[str, ...]  # the section's other named people (opener, witness): on the answer's line they void it


def _names(rng: random.Random, count: int) -> list[str]:
    """``count`` full names, no surname used twice."""
    surnames = [start + end for start in SURNAME_STARTS for end in SURNAME_ENDS]
    rng.shuffle(surnames)
    return [f"{rng.choice(FIRST)} {surname}" for surname in surnames[:count]]


def _section(rng: random.Random, number: int, place: str, people: dict[str, str]) -> str:
    def filler() -> str:
        values = {f"n{i}": rng.randrange(3, 4800) for i in range(1, 18)}
        cargo, cargo2 = rng.sample(CARGO, 2)
        return rng.choice(FILLER).format(place=rng.choice(PLACES), vessel=rng.choice(VESSELS), cargo=cargo, cargo2=cargo2,
                                         weather=rng.choice(WEATHER), work=rng.choice(WORKS), **values)  # fmt: skip

    paragraphs = []
    roles = [rng.choice(ROLE_SENTENCES[role]).format(name=name) for role, name in people.items()]
    slots = sorted(rng.sample(range(6), 3))  # which of the six paragraphs carry a role sentence: anywhere in the section
    rng.shuffle(roles)
    for index in range(6):
        sentences = [filler() for _ in range(rng.randrange(4, 7))]
        if index in slots:
            sentences.insert(rng.randrange(len(sentences) + 1), roles[slots.index(index)])
        paragraphs.append(" ".join(sentences))
    return f"## Section {number}: {place}\n\n" + "\n\n".join(paragraphs) + "\n"


def document(seed: int = SEED, sections: int = SECTIONS) -> tuple[str, list[Fact]]:
    """The register's text and its facts, the same for the same arguments."""
    if not 1 <= sections <= MAX_SECTIONS:
        raise ValueError(f"sections must be between 1 and {MAX_SECTIONS}")
    rng = random.Random(seed)
    names = _names(rng, 3 * sections)
    parts = [
        "# Harbour Board of Velmora: quarterly register\n",
        (
            f"This register has {sections} numbered sections, one for each berth district and quarter. Each section is an "
            "entry that one officer opened, one witnessed and one closed.\n"
        ),
    ]
    facts = []
    for number in range(1, sections + 1):
        opener, witness, closer = names[3 * (number - 1) : 3 * number]
        place = f"{rng.choice(PLACES)}, quarter {rng.randrange(1, 5)} of year {rng.randrange(31, 58)}"
        parts.append(_section(rng, number, place, {"opener": opener, "witness": witness, "closer": closer}))
        facts.append(Fact(number, place, closer, (opener, witness)))
    return "\n".join(parts), facts


def prompt_text(sections: int = SECTIONS) -> str:
    return (
        f"The file {FILE_NAME} is a long register with {sections} numbered sections. Every section is an entry that one officer "
        "opened, one witnessed and one closed.\n\n"
        f"Read the whole register. For every section, find the full name of the officer who closed (finished) the entry. Write the answers to "
        f"{DIGEST_NAME}, one line per section, in the order of the sections, each line in exactly this form:\n\n"
        "Section 12: Firstname Lastname\n\n"
        "Give only the officer who closed the entry, not the other two. Do not guess: leave out the line of a section you did not read.\n"
    )


def _surname(name: str) -> str:
    return name.split()[1].lower()


def sections_of(task_name: str) -> int:
    """`register-24` -> 24: the task's name carries the size of its register."""
    kind, _, number = task_name.partition("-")
    if kind != "register" or not number.isdigit():
        raise ValueError(f"unknown long-document task {task_name!r}")
    return int(number)


def score(digest: str, facts: list[Fact]) -> dict[str, Any]:
    """How much of the register the digest proves was read: per section right, wrong or missing.

    A section is right when its line names the officer who closed it (the surname as a whole word is enough: no two
    people share one) and names neither of the section's other two people. A section with more than one line is judged on the last one.
    ``coverage`` is right / sections; ``furthest`` is the highest section answered right.
    """
    lines: dict[int, set[str]] = {}
    for raw in digest.splitlines():
        match = LINE_RE.match(raw)
        if match:
            lines[int(match.group(1))] = set(re.findall(r"[a-z]+", match.group(2).lower()))
    correct, wrong, missing = [], [], []
    for fact in facts:
        value = lines.get(fact.section)
        if value is None:
            missing.append(fact.section)
        elif _surname(fact.closer) in value and not any(_surname(other) in value for other in fact.others):
            correct.append(fact.section)
        else:
            wrong.append(fact.section)
    return {
        "sections": len(facts),
        "correct": len(correct),
        "wrong": len(wrong),
        "missing": len(missing),
        "coverage": round(len(correct) / len(facts), 3) if facts else 0.0,
        "furthest": max(correct, default=0),
        "wrong_sections": wrong[:20],
        "missing_sections": missing[:20],
        "lines_of_no_section": sum(1 for n in lines if not 1 <= n <= len(facts)),
    }
