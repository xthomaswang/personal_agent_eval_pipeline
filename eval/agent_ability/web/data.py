"""Load and validate the pilot question set (DESIGN.md, section 6).

``load()`` reads data/simpleqa_verified_pilot.yaml and checks:

- the schema: the source block, and for every row its fields, types and a known ``kind`` and ``tier``;
- the shape of the set: 12 rows per tier, unique ``original_index`` values;
- that the gold answer and every alias are valid under the row's ``kind`` (each must score as correct against
  itself), so a typo in an alias or a gold that does not parse fails here, not in the middle of a study.

``check_csv()`` compares the rows with the pinned CSV, which fetch.py puts in the Hugging Face cache. It never
downloads anything: it returns None when the CSV is not cached.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from . import fetch, score

PILOT_YAML = Path(__file__).resolve().parent / "data" / "simpleqa_verified_pilot.yaml"
TIERS = ("easy", "medium")
ROWS_PER_TIER = 12
DATASET_FIELDS = ("problem", "answer", "topic", "answer_type", "multi_step", "requires_reasoning", "urls")

_SOURCE_KEYS = {"dataset", "file", "revision", "csv_sha256", "paper", "license", "license_notice", "citation"}
_ROW_KEYS = {"original_index", "tier", "kind", "aliases", "unordered", *DATASET_FIELDS}
_ROW_REQUIRED = _ROW_KEYS - {"unordered"}
_URL_SPLIT = re.compile(r",(?=['\"]?https?://)")


class DataError(ValueError):
    """The question file is malformed or does not match the pinned dataset."""


@dataclass(frozen=True)
class Row:
    original_index: int
    tier: str
    kind: str
    problem: str
    answer: str
    aliases: tuple[str, ...]
    unordered: bool
    topic: str
    answer_type: str
    multi_step: bool
    requires_reasoning: bool
    urls: tuple[str, ...]

    @property
    def id(self) -> int:
        return self.original_index

    @property
    def targets(self) -> tuple[str, ...]:
        """The gold answer, then the aliases."""
        return (self.answer, *self.aliases)

    def scoring(self) -> dict[str, Any]:
        """Keyword arguments for ``score.score`` and ``score.grade``."""
        return {"gold": self.answer, "kind": self.kind, "aliases": self.aliases, "unordered": self.unordered}

    def gold_hosts(self) -> frozenset[str]:
        """Host names of the gold URLs, lower case and without a leading ``www.``."""
        return frozenset(h for h in map(url_host, self.urls) if h)


@dataclass(frozen=True)
class Pilot:
    source: Mapping[str, Any]
    rows: tuple[Row, ...]

    def tier(self, name: str) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if r.tier == name)

    def by_id(self, original_index: int) -> Row:
        for row in self.rows:
            if row.original_index == original_index:
                return row
        raise KeyError(original_index)


def url_host(url: str) -> str:
    """The host of a dataset URL; the dataset has stray quotes around a few of them."""
    host = urlsplit(url.strip(" '\"")).hostname or ""
    return host.lower().removeprefix("www.")


def split_urls(field: str) -> list[str]:
    """Split the CSV's ``urls`` field into URLs; joining them with ``,`` gives the field back."""
    return _URL_SPLIT.split(field)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _validate_source(source: Any) -> None:
    _require(isinstance(source, dict), "source must be a mapping")
    _require(set(source) == _SOURCE_KEYS, f"source keys must be {sorted(_SOURCE_KEYS)}, got {sorted(source)}")
    _require(source["dataset"] == fetch.REPO_ID, f"source.dataset must be {fetch.REPO_ID}")
    _require(source["file"] == fetch.FILENAME, f"source.file must be {fetch.FILENAME}")
    _require(source["revision"] == fetch.REVISION, f"source.revision must be the pinned {fetch.REVISION}")
    _require(
        isinstance(source["csv_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", source["csv_sha256"]) is not None,
        "source.csv_sha256 must be 64 lower-case hex digits",
    )
    _require(source["license"] == "MIT", "source.license must be MIT")
    for key in ("paper", "license_notice", "citation"):
        _require(
            isinstance(source[key], str) and source[key].strip() != "", f"source.{key} must be non-empty text"
        )
    _require("2509.07968" in source["citation"], "source.citation must cite arXiv 2509.07968")


def _validate_row(raw: Any, position: int) -> Row:
    _require(isinstance(raw, dict), f"rows[{position}] must be a mapping")
    missing, extra = _ROW_REQUIRED - set(raw), set(raw) - _ROW_KEYS
    _require(not missing, f"rows[{position}] is missing {sorted(missing)}")
    _require(not extra, f"rows[{position}] has unknown keys {sorted(extra)}")
    label = f"row {raw['original_index']}"
    _require(
        isinstance(raw["original_index"], int) and not isinstance(raw["original_index"], bool),
        f"rows[{position}].original_index must be an integer",
    )
    _require(raw["tier"] in TIERS, f"{label}: tier must be one of {TIERS}")
    _require(raw["kind"] in score.KINDS, f"{label}: kind must be one of {score.KINDS}")
    for key in ("problem", "answer", "topic", "answer_type"):
        _require(
            isinstance(raw[key], str) and raw[key].strip() != "", f"{label}: {key} must be non-empty text"
        )
    for key in ("multi_step", "requires_reasoning"):
        _require(isinstance(raw[key], bool), f"{label}: {key} must be true or false")
    _require(_is_str_list(raw["aliases"]), f"{label}: aliases must be a list of strings")
    _require(all(a.strip() for a in raw["aliases"]), f"{label}: an alias is empty")
    _require(len(set(raw["aliases"])) == len(raw["aliases"]), f"{label}: aliases repeat")
    unordered = raw.get("unordered", False)
    _require(isinstance(unordered, bool), f"{label}: unordered must be true or false")
    _require(not unordered or raw["kind"] == "list", f"{label}: unordered applies to lists only")
    _require(
        _is_str_list(raw["urls"]) and len(raw["urls"]) >= 1,
        f"{label}: urls must be a non-empty list of strings",
    )
    row = Row(
        original_index=raw["original_index"],
        tier=raw["tier"],
        kind=raw["kind"],
        problem=raw["problem"],
        answer=raw["answer"],
        aliases=tuple(raw["aliases"]),
        unordered=unordered,
        topic=raw["topic"],
        answer_type=raw["answer_type"],
        multi_step=raw["multi_step"],
        requires_reasoning=raw["requires_reasoning"],
        urls=tuple(raw["urls"]),
    )
    _require(row.gold_hosts() != frozenset(), f"{label}: no URL has a host")
    for target in row.targets:
        _require(
            score.matches(target, target, row.kind, unordered=row.unordered),
            f"{label}: {target!r} is not valid for kind {row.kind!r} (it must score as correct against itself)",
        )
    return row


def validate(document: Any) -> Pilot:
    """Check a parsed YAML document and return it as a ``Pilot``; raises ``DataError``."""
    _require(
        isinstance(document, dict) and set(document) == {"source", "rows"},
        "top level must be source and rows",
    )
    _validate_source(document["source"])
    _require(isinstance(document["rows"], list), "rows must be a list")
    rows = tuple(_validate_row(raw, i) for i, raw in enumerate(document["rows"]))
    ids = [r.original_index for r in rows]
    _require(
        len(set(ids)) == len(ids),
        f"original_index values repeat: {sorted(i for i in set(ids) if ids.count(i) > 1)}",
    )
    for tier in TIERS:
        count = sum(r.tier == tier for r in rows)
        _require(count == ROWS_PER_TIER, f"tier {tier!r} has {count} rows, expected {ROWS_PER_TIER}")
    _require(
        len(rows) == ROWS_PER_TIER * len(TIERS),
        f"expected {ROWS_PER_TIER * len(TIERS)} rows, got {len(rows)}",
    )
    return Pilot(source=document["source"], rows=rows)


def load(path: Path | str = PILOT_YAML) -> Pilot:
    """Read and validate the question file."""
    return validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def check_csv(
    pilot: Pilot, csv_path: Path | str | None = None, *, check_hash: bool = True
) -> list[str] | None:
    """Mismatches between the rows and the pinned CSV (an empty list means they agree).

    ``csv_path`` defaults to the copy in the Hugging Face cache; None is returned when there is none. Nothing is
    downloaded: run fetch.py first.
    """
    path = Path(csv_path) if csv_path is not None else fetch.cached_csv()
    if path is None or not path.is_file():
        return None
    problems: list[str] = []
    if check_hash and fetch.sha256_file(path) != pilot.source["csv_sha256"]:
        problems.append(f"{path.name}: sha256 differs from source.csv_sha256")
    with path.open(newline="", encoding="utf-8") as f:
        dataset = {int(r["original_index"]): r for r in csv.DictReader(f)}
    for row in pilot.rows:
        record = dataset.get(row.original_index)
        if record is None:
            problems.append(f"row {row.original_index}: not in the CSV")
            continue
        expected = {
            "problem": record["problem"],
            "answer": record["answer"],
            "topic": record["topic"],
            "answer_type": record["answer_type"],
            "multi_step": record["multi_step"] == "True",
            "requires_reasoning": record["requires_reasoning"] == "True",
            "urls": split_urls(record["urls"]),
        }
        for field, want in expected.items():
            have = getattr(row, field)
            have = list(have) if isinstance(have, tuple) else have
            if have != want:
                problems.append(f"row {row.original_index}: {field} is {have!r}, the CSV has {want!r}")
    return problems
