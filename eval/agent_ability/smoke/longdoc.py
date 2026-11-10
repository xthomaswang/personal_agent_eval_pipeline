"""A long document no model can know, with one checkable fact planted in every section.

A public document (an RFC, a statute) is a poor long-reading case: a model that knows it can write every section's
summary after reading half of it, and the section headings of its output prove nothing. This register is generated from
a seed. Every section is routine harbour accounts (counts, dues, works, who opened and closed the entry) made from a
few sentence patterns, and among them one out-of-the-ordinary incident: a sentence written by hand, one of INCIDENTS,
at a place of its own. The task asks for the incident of every section. The checker makes the same register again from
the seed, so no answer file is ever in the workspace.

What the pilots showed (2026-11-09, one local model through a gateway chat, 24 sections):
- Asked who closed each entry (a name in one of eight sentence forms), the model wrote a search pattern per form and
  answered all 24 in 46 s without reading the register.
- Asked for the incident (sentences with no common form or word), it wrote code that set aside every sentence of a
  repeated form and kept the rest: 24 of 24 in 76 s, again without a full read.
- Told in the task to read the register itself and not to pick sentences out with a script or a search, it read the
  file in three parts and answered 24 of 24 in 45 s.
So the task text carries that rule, and the incidents stay: a search has no word to look for. A run that breaks the
rule can still score; the score says which facts reached the output, the run's tool events say how they were found.

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
SURNAME_STARTS = (
    "Osk", "Thar", "Bel", "Quil", "Vander", "Hral", "Pember", "Zor", "Alden", "Fys", "Corr", "Lind", "Massen", "Ubr",
    "Wynd", "Skarr", "Della", "Rons",
)  # fmt: skip
SURNAME_ENDS = ("vik", "row", "marsh", "lon", "see", "dt", "wick", "ic", "moor", "ke", "ane", "qvar", "grave", "echt", "low", "ild")
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
# One incident per section: (the sentence, the words that show a digest line names it). The first word of each tuple is
# in the sentence; the others are words a faithful line may use instead. No word of one incident names another, and
# none is a word of the routine sentences (tests/test_longdoc.py checks both against the whole register).
INCIDENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A whale stranded on the bar at low water and was towed off by two tenders on the next flood.", ("whale",)),
    ("The great bell of the harbour office cracked while ringing in the fleet and has been sent to the founder.", ("bell",)),
    ("A comet was seen over the roads for nine nights, and the pilots refused to sail until it had gone.", ("comet",)),
    ("A travelling circus landed an elephant on the quay, which broke through the planking of the old stage.", ("elephant", "circus")),
    ("A chest of counterfeit coin was found walled up in the collector's cellar.", ("counterfeit", "coin", "forged")),
    ("The harbour office cat was carried off to sea in a grain ship and returned two months later by the same master.", ("cat",)),
    ("Lightning struck the signal mast and melted the copper cap at its head.", ("lightning",)),
    ("A bride was rowed across to her wedding in the pilot gig, the ferry being laid up.", ("bride", "wedding")),
    ("Smugglers were taken at night with forty casks of brandy sunk on a line off the mole.", ("smugglers", "brandy", "smuggling")),
    ("A swarm of bees settled in the crane house and stopped work for a day and a half.", ("bees", "swarm")),
    ("The body of an unknown sailor was washed up and buried at the board's cost.", ("body", "corpse", "sailor")),
    ("A fire in the sailmaker's loft was put out with the harbour's own pump before it reached the tar store.", ("fire", "sailmaker")),
    ("An earthquake at dawn threw down the chimney of the smithy and cracked the dry dock sill.", ("earthquake", "tremor")),
    ("A message in a bottle, three years in the water, was brought in by a boy and read out to the board.", ("bottle",)),
    ("The clock on the customs tower stopped for eleven days and vessels were timed by the sun-dial.", ("clock", "dial", "sundial")),
    ("A seal took to sleeping on the lower steps and bit a porter who tried to move it.", ("seal", "porter")),
    ("A balloon came down in the outer harbour and its two aeronauts were picked up by a cutter.", ("balloon", "aeronauts")),
    ("The harbour master's wooden leg was lost overboard and found again in a dredger's bucket.", ("leg", "wooden")),
    ("A cargo of oranges, refused by its buyer, was given away on the quay to all comers.", ("oranges",)),
    ("A strike of the coal heavers closed the east wharf for six days until the board met their price.", ("strike", "heavers")),
    ("A lighthouse keeper's wife gave birth in the tender on the way ashore, and the child was named for the boat.", ("birth", "child", "baby", "born")),
    ("A waterspout crossed the anchorage and lifted two skiffs clean out of the water.", ("waterspout", "skiffs")),
    ("A sunken wreck of unknown age was uncovered by the dredging, with brass cannon still aboard.", ("wreck", "shipwreck")),
    ("A duel was fought on the breakwater between two ship's officers; neither was hurt and both were fined.", ("duel",)),
    ("A barque from the south came in with fever aboard and lay in quarantine forty days.", ("quarantine", "fever")),
    ("A thief made off with the brass name-board from the harbour office door.", ("thief", "theft", "stole", "stolen")),
    ("A blind fiddler was licensed to play on the ferry stage, the first such licence the board has given.", ("fiddler", "fiddle", "blind")),
    ("A shoal of herring so thick came into the basin that boys took them with baskets from the steps.", ("herring", "baskets")),
    ("An eclipse of the sun at noon stopped all work on the quays for an hour.", ("eclipse",)),
    ("A runaway horse dragged a dray off the quay edge; the horse was saved and the dray was not.", ("horse", "dray")),
    ("A stowaway of about ten years was found in a collier's bunker and put to school at the board's charge.", ("stowaway",)),
    ("Gold dust to the weight of a man's hand was found in the sweepings of the bonded floor.", ("gold",)),
    ("A petition signed by ninety fishwives against the new fish stage was laid before the board.", ("petition", "fishwives")),
    ("The portrait of the first harbour master fell from the boardroom wall during a sitting.", ("portrait", "painting")),
    ("A school of porpoises followed the mail packet up to the inner lock gates.", ("porpoises", "porpoise")),
    ("A foreign prince came ashore unannounced and was shown the works by the foreman in his shirtsleeves.", ("prince",)),
    ("A meteor fell hissing into the outer basin in full view of the night shift.", ("meteor",)),
    ("The old tide mill was sold for breaking and its wheel given to the school as a curiosity.", ("mill", "wheel")),
    ("A parrot escaped from a brig and lived a month in the rigging loft, where it learned the foreman's whistle.", ("parrot",)),
    ("A new chart of the approaches was engraved and the old copper plates melted down.", ("chart", "engraved")),
    ("A chimney sweep fell into the dry dock and walked away unhurt.", ("sweep",)),
    ("A cask of gunpowder was found among the wool in the customs shed and carried out by hand.", ("gunpowder",)),
    ("The first steam tug in these waters arrived and was hired for a trial of one month.", ("steam", "tug")),
    ("A diver went down to the lock sill in the new dress and stayed under for half an hour.", ("diver", "diving")),
    ("A funeral barge carried the old pilot master out to be buried at sea, by his own wish.", ("funeral",)),
    ("A wager was laid that a man could swim the harbour mouth at slack water, and he did it.", ("swim", "swam", "swimmer", "wager")),
    ("Rats got into the grain store in such numbers that a terrier man was hired from inland.", ("rats", "terrier")),
    ("A new flag for the board was sewn by the pilots' wives and hoisted on the first of the quarter.", ("flag",)),
    ("A small iceberg drifted into the roads and grounded off the mole for a week.", ("iceberg",)),
    ("The harbour band was formed, with instruments bought from the fines box.", ("band", "instruments")),
    ("A mutiny aboard a timber ship at anchor was put down by the harbour guard without a shot.", ("mutiny",)),
    ("A lottery was held to pay for the new lifeboat, and the first prize went to a shrimper's widow.", ("lottery",)),
    ("A sea eagle nested on the outer beacon and the lamp was left unlit until the young had flown.", ("eagle", "nest", "nested")),
    ("A lunatic climbed the crane jib and preached from it until evening.", ("lunatic", "madman", "preached")),
    ("A stone idol of great age was dug out of the foreshore and set up before the office.", ("idol", "statue")),
    ("The harbour well turned brackish and water was carted from the hills for the rest of the quarter.", ("brackish", "well")),
    ("A chess match between the board and the pilots was played by signal lamps across the basin.", ("chess",)),
    ("A two-headed calf was landed from a cattle boat and shown on the quay for a penny.", ("calf",)),
    ("A foreign frigate saluted the port with twenty-one guns and broke the windows of the pilot house.", ("frigate", "salute", "saluted")),
    ("Eels in their thousands choked the lock culverts for three nights.", ("eels",)),
    ("A woman dressed as a man was found to have served two years as a lumper, and was kept on at the same wage.", ("woman", "disguised")),
    ("The harbour ghost, a grey lady on the old stairs, was reported by three watchmen on the same night.", ("ghost",)),
    ("A snowfall of three feet stopped the cranes, and the lumpers were paid to clear the quays.", ("snowfall", "snow")),
    ("A tightrope walker crossed the inner basin on a hawser for a collection.", ("tightrope", "acrobat")),
    ("A pearl of fair size was found in an oyster at the board's own dinner.", ("pearl", "oyster")),
    ("A cannon ball from the old siege was dredged up and found to be still charged.", ("ball", "siege", "cannonball")),
    ("A mermaid was sworn to by two fishermen before the magistrate, who entered it without comment.", ("mermaid",)),
    ("A carrier pigeon brought the first news of the fleet's return two days before the fleet.", ("pigeon",)),
    ("A tooth of some great beast, long as a forearm, came up in the dredge and was sent to the college.", ("tooth", "fossil", "tusk")),
    ("A library of sea books was left to the board by a retired master and housed in the boardroom.", ("library", "books")),
    ("A lamplighter was found to have been dead drunk on the night the leading lights failed, and was let go.", ("lamplighter", "drunk")),
    ("A sinkhole opened behind the east quay wall and swallowed a loaded cart.", ("sinkhole", "cart")),
    ("A knighthood was conferred on the chairman, who received it on the quay from the royal yacht.", ("knighthood", "knighted", "yacht")),
    ("A rain of small frogs fell on the fish stage after a thunderstorm.", ("frogs",)),
    ("The pilots' regatta was revived after twenty years and won by the oldest crew.", ("regatta",)),
    ("A pair of swans was presented to the harbour by the town and given the upper basin.", ("swans",)),
    ("An organ grinder's monkey got aboard the customs launch and threw the day's dockets overboard.", ("monkey",)),
    ("A boy fell from the pier head and was pulled out by the harbour dog, which was given a collar by the board.", ("dog", "collar")),
    ("A blight of shipworm was found in the piles of the fish stage, the first seen in these waters.", ("shipworm", "worm", "blight")),
    ("The first photograph of the harbour was taken from the customs tower, the sitters keeping still for a minute.", ("photograph", "photo")),
    ("A treaty between two fishing villages over the mussel beds was signed in the boardroom.", ("treaty", "mussel")),
    ("A bolt of silk was found sewn inside a bale of rags, and the consignee could not be traced.", ("silk",)),
    ("The harbour's first gas lamp was lit on the pier head before a large crowd.", ("gas",)),
    ("A cooper was crushed between two casks and the board paid his widow a pension.", ("cooper", "crushed", "pension")),
    ("A bear kept by a ship's cook broke its chain and was shot on the ropewalk.", ("bear",)),
    ("A captain was fined for flying his washing from the signal halyards on a Sunday.", ("washing", "laundry")),
    ("A hermit was found living in the hulk at the head of the creek, and was left in peace by order.", ("hermit", "hulk")),
    ("A landslip carried forty yards of the cliff road onto the coal wharf.", ("landslip", "landslide", "cliff")),
    ("A printing press was set up in the harbour office to print the tide tables.", ("press", "printing")),
    ("A beached squid of twelve arms' length was measured by the surveyor and drawn for the minutes.", ("squid",)),
    ("A fortune teller was put off the ferry for frightening the passengers with drownings.", ("fortune", "teller")),
    ("A cask of wine a hundred years old was found in the vault and emptied by the board at its next sitting.", ("wine", "vault")),
    ("A sword was fished up in a trawl and claimed by three families.", ("sword",)),
    ("A troupe of players acted a tragedy in the transit shed, and the takings went to the pilots' fund.", ("players", "tragedy", "actors", "troupe")),
    ("An albatross, never before seen so far north, was found exhausted on the breakwater and fed until it flew.", ("albatross",)),
    ("A census of the harbour's people was taken for the first time, and came to two thousand and eleven.", ("census",)),
    ("A chandler's shop sign in the shape of a gilt dolphin was blown into the harbour and never found.", ("dolphin", "gilt")),
    ("A leper was landed by a foreign ship in the night and housed in the old powder magazine until a passage was found.", ("leper", "magazine")),
    ("A walled-up doorway in the old fort was opened and a skeleton in irons found behind it.", ("skeleton", "irons")),
    ("A hailstorm with stones the size of hens' eggs broke every pane in the lighthouse.", ("hailstorm", "hail", "hailstones")),
)
MAX_SECTIONS = len(INCIDENTS)
LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?section\s+(\d+)(?:\*\*)?\s*[:.\-–]\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Fact:
    section: int
    place: str
    incident: str  # the sentence in the register
    keys: tuple[str, ...]  # a digest line names the incident when it has one of these words


def _names(rng: random.Random, count: int) -> list[str]:
    surnames = [start + end for start in SURNAME_STARTS for end in SURNAME_ENDS]
    return [f"{rng.choice(FIRST)} {rng.choice(surnames)}" for _ in range(count)]


def _section(rng: random.Random, number: int, place: str, people: dict[str, str], incident: str) -> str:
    def filler() -> str:
        values = {f"n{i}": rng.randrange(3, 4800) for i in range(1, 18)}
        cargo, cargo2 = rng.sample(CARGO, 2)
        return rng.choice(FILLER).format(place=rng.choice(PLACES), vessel=rng.choice(VESSELS), cargo=cargo, cargo2=cargo2,
                                         weather=rng.choice(WEATHER), work=rng.choice(WORKS), **values)  # fmt: skip

    # The sentences with a place of their own: who opened, witnessed and closed the entry, and the incident. Each goes
    # into one of the six paragraphs at any position, so nothing is at a fixed place in a section.
    own = [rng.choice(ROLE_SENTENCES[role]).format(name=name) for role, name in people.items()] + [incident]
    rng.shuffle(own)
    slots = sorted(rng.sample(range(6), len(own)))
    paragraphs = []
    for index in range(6):
        sentences = [filler() for _ in range(rng.randrange(4, 7))]
        if index in slots:
            sentences.insert(rng.randrange(len(sentences) + 1), own[slots.index(index)])
        paragraphs.append(" ".join(sentences))
    return f"## Section {number}: {place}\n\n" + "\n\n".join(paragraphs) + "\n"


def document(seed: int = SEED, sections: int = SECTIONS) -> tuple[str, list[Fact]]:
    """The register's text and its facts, the same for the same arguments."""
    if not 1 <= sections <= MAX_SECTIONS:
        raise ValueError(f"sections must be between 1 and {MAX_SECTIONS}")
    rng = random.Random(seed)
    names = _names(rng, 3 * sections)
    incidents = rng.sample(INCIDENTS, sections)
    parts = [
        "# Harbour Board of Velmora: quarterly register\n",
        (
            f"This register has {sections} numbered sections, one for each berth district and quarter. Each section is an "
            "entry of routine counts and accounts, and each records one incident out of the ordinary.\n"
        ),
    ]
    facts = []
    for number in range(1, sections + 1):
        opener, witness, closer = names[3 * (number - 1) : 3 * number]
        place = f"{rng.choice(PLACES)}, quarter {rng.randrange(1, 5)} of year {rng.randrange(31, 58)}"
        sentence, keys = incidents[number - 1]
        parts.append(_section(rng, number, place, {"opener": opener, "witness": witness, "closer": closer}, sentence))
        facts.append(Fact(number, place, sentence, keys))
    return "\n".join(parts), facts


def prompt_text(sections: int = SECTIONS) -> str:
    return (
        f"The file {FILE_NAME} is a long register with {sections} numbered sections. Every section is routine harbour "
        "counts and accounts, and among them exactly one incident out of the ordinary (an accident, a find, a visitor, "
        "a strange event).\n\n"
        "Read the whole register yourself, part by part, with your file reading tool. Do not use a script, a search or "
        "a pattern to pick sentences out: the task is the reading.\n\n"
        f"For every section, find its incident. Write the answers to {DIGEST_NAME}, one line per section, in the order of "
        "the sections, each line in exactly this form, with the incident in a few of the register's own words:\n\n"
        "Section 12: a lamp post was blown down in the night\n\n"
        "Do not guess: leave out the line of a section you did not read.\n"
    )


def sections_of(task_name: str) -> int:
    """`register-24` -> 24: the task's name carries the size of its register."""
    kind, _, number = task_name.partition("-")
    if kind != "register" or not number.isdigit():
        raise ValueError(f"unknown long-document task {task_name!r}")
    return int(number)


def names_it(words: set[str], keys: tuple[str, ...]) -> bool:
    """Whether a line's words have one of an incident's words: the word itself, its plural, or (from six letters) its start."""
    return any(w in (k, k + "s", k + "es") or (len(k) >= 6 and w.startswith(k)) for k in keys for w in words)


def score(digest: str, facts: list[Fact]) -> dict[str, Any]:
    """How much of the register the digest proves was read: per section right, wrong or missing.

    A section is right when its line names its incident (one of the incident's words) and names at most one incident
    of another section, so a line that lists many incidents is no answer. A section with more than one line is judged
    on the last one. ``coverage`` is right / sections; ``furthest`` is the highest section answered right.
    """
    lines: dict[int, set[str]] = {}
    for raw in digest.splitlines():
        match = LINE_RE.match(raw)
        if match:
            lines[int(match.group(1))] = set(re.findall(r"[a-z]+", match.group(2).lower()))
    correct, wrong, missing = [], [], []
    for fact in facts:
        words = lines.get(fact.section)
        if words is None:
            missing.append(fact.section)
        elif names_it(words, fact.keys) and sum(1 for other in facts if other is not fact and names_it(words, other.keys)) <= 1:
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
