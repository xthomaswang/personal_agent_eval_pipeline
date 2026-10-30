"""The pilot question file: schema, tiers, ids, hand-set fields, and agreement with the pinned CSV when it is cached."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import pytest
import yaml
from web import data, fetch

EASY_IDS = [3503, 1337, 3531, 3010, 3561, 302, 3181, 3635, 2923, 3372, 1442, 3538]
MEDIUM_IDS = [1934, 378, 2146, 1963, 2849, 1608, 2783, 323, 3157, 1059, 69, 3352]


@pytest.fixture(scope="module")
def pilot() -> data.Pilot:
    return data.load()


@pytest.fixture
def raw() -> dict[str, Any]:
    """The parsed YAML, free to mutate."""
    return copy.deepcopy(yaml.safe_load(data.PILOT_YAML.read_text(encoding="utf-8")))


# -- the shipped file -------------------------------------------------------------------------------------------


def test_the_pinned_rows_and_tiers(pilot: data.Pilot) -> None:
    assert [r.original_index for r in pilot.tier("easy")] == EASY_IDS
    assert [r.original_index for r in pilot.tier("medium")] == MEDIUM_IDS
    assert len(pilot.rows) == 24
    assert len({r.id for r in pilot.rows}) == 24


def test_source_block_pins_the_dataset(pilot: data.Pilot) -> None:
    source = pilot.source
    assert source["dataset"] == "google/simpleqa-verified"
    assert source["revision"] == fetch.REVISION == "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"
    assert source["csv_sha256"] == "b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5"
    assert source["license"] == "MIT"
    assert "MIT License" in source["license_notice"]
    assert "@misc{" in source["citation"] and "2509.07968" in source["citation"]


def test_every_row_has_its_fields_and_gold_hosts(pilot: data.Pilot) -> None:
    for row in pilot.rows:
        assert row.kind in {"number", "year", "date", "string", "list"}
        assert row.problem.strip() and row.answer.strip()
        assert row.gold_hosts(), row.id
        assert len(row.urls) >= 2, row.id  # the dataset promises at least two supporting URLs
        assert all(isinstance(a, str) and a for a in row.aliases)


def test_gold_hosts_strip_quotes_and_www(pilot: data.Pilot) -> None:
    assert "en.wikipedia.org" in pilot.by_id(2783).gold_hosts()  # one dataset URL is wrapped in quotes
    assert "researchgate.net" in pilot.by_id(3352).gold_hosts()  # one dataset URL ends in a stray quote
    assert all(not h.startswith("www.") and "'" not in h for r in pilot.rows for h in r.gold_hosts())
    assert data.url_host("'https://WWW.Example.org/a?b=c'") == "example.org"
    assert data.url_host("not a url") == ""


def test_hand_aliases_of_the_four_pinned_rows(pilot: data.Pilot) -> None:
    list_row = pilot.by_id(3538)
    assert (list_row.answer, list_row.kind, list_row.unordered) == (
        "Painting, music, and poetry.",
        "list",
        True,
    )
    assert "Painting, music, poetry" in list_row.aliases

    alloy = pilot.by_id(3352)
    assert (alloy.answer, alloy.kind) == (
        "2,024",
        "string",
    )  # an alloy name: a number tolerance would accept 2014
    assert {"2024", "Al 2024"} <= set(alloy.aliases)

    village = pilot.by_id(2849)
    assert village.answer == "Prince Albert."
    assert "Prince Albert" in village.aliases

    position = pilot.by_id(69)
    assert position.answer == "15th place"
    assert {"15th", "15"} <= set(position.aliases)


def test_identifier_rows_are_strings_not_numbers(pilot: data.Pilot) -> None:
    # A number tolerance would accept an order number 7035 for 7034, or the alloy 2014 for 2024.
    assert pilot.by_id(3635).kind == "string"
    assert pilot.by_id(3352).kind == "string"


def test_only_list_rows_are_unordered(pilot: data.Pilot) -> None:
    assert [r.id for r in pilot.rows if r.unordered] == [3538]
    assert [r.id for r in pilot.rows if r.kind == "list"] == [3538]


def test_the_file_has_no_local_paths() -> None:
    text = data.PILOT_YAML.read_text(encoding="utf-8")
    assert "/Users/" not in text and "/home/" not in text


def test_row_helpers(pilot: data.Pilot) -> None:
    row = pilot.by_id(69)
    assert row.targets[0] == "15th place" and row.targets[1:] == row.aliases
    assert row.scoring() == {
        "gold": "15th place",
        "kind": "string",
        "aliases": row.aliases,
        "unordered": False,
    }
    with pytest.raises(KeyError):
        pilot.by_id(1)


# -- validation -------------------------------------------------------------------------------------------------


def test_load_accepts_the_shipped_document(raw: dict[str, Any]) -> None:
    assert len(data.validate(raw).rows) == 24


def test_duplicate_id_is_refused(raw: dict[str, Any]) -> None:
    raw["rows"][1]["original_index"] = raw["rows"][0]["original_index"]
    with pytest.raises(data.DataError, match="repeat"):
        data.validate(raw)


def test_unbalanced_tiers_are_refused(raw: dict[str, Any]) -> None:
    raw["rows"][0]["tier"] = "medium"
    with pytest.raises(data.DataError, match="tier"):
        data.validate(raw)


def test_wrong_row_count_is_refused(raw: dict[str, Any]) -> None:
    del raw["rows"][-1]
    with pytest.raises(data.DataError):
        data.validate(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tier", "hard", "tier"),
        ("kind", "regex", "kind"),
        ("aliases", "Delhi", "aliases"),
        ("aliases", [""], "alias"),
        ("aliases", ["a", "a"], "repeat"),
        ("unordered", True, "lists only"),
        ("urls", [], "urls"),
        ("urls", ["not a url"], "host"),
        ("multi_step", "False", "multi_step"),
        ("original_index", "3503", "integer"),
        ("answer", "", "answer"),
    ],
)
def test_bad_field_is_refused(raw: dict[str, Any], field: str, value: Any, message: str) -> None:
    raw["rows"][0][field] = value  # row 0 is a string row
    with pytest.raises(data.DataError, match=message):
        data.validate(raw)


def test_missing_and_unknown_row_keys_are_refused(raw: dict[str, Any]) -> None:
    del raw["rows"][0]["kind"]
    with pytest.raises(data.DataError, match="missing"):
        data.validate(raw)
    raw["rows"][0]["kind"] = "string"
    raw["rows"][0]["extra"] = 1
    with pytest.raises(data.DataError, match="unknown keys"):
        data.validate(raw)


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [("number", "aliases", ["two"]), ("year", "answer", "sometime in 1962"), ("string", "aliases", ["."])],
)
def test_gold_or_alias_invalid_for_its_kind_is_refused(
    raw: dict[str, Any], kind: str, field: str, value: Any
) -> None:
    next(r for r in raw["rows"] if r["kind"] == kind)[field] = value
    with pytest.raises(data.DataError, match=f"not valid for kind '{kind}'"):
        data.validate(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("revision", "0" * 40),
        ("csv_sha256", "abc"),
        ("dataset", "someone/else"),
        ("license", "Apache-2.0"),
        ("citation", "no arxiv id here"),
        ("license_notice", ""),
    ],
)
def test_bad_source_is_refused(raw: dict[str, Any], key: str, value: str) -> None:
    raw["source"][key] = value
    with pytest.raises(data.DataError):
        data.validate(raw)


def test_unknown_top_level_key_is_refused(raw: dict[str, Any]) -> None:
    raw["extra"] = 1
    with pytest.raises(data.DataError, match="top level"):
        data.validate(raw)


# -- the CSV ----------------------------------------------------------------------------------------------------


def write_csv(path: Path, pilot: data.Pilot, mutate: Any = None) -> Path:
    """A CSV with the pilot's rows in the dataset's layout (booleans as True/False, urls comma-joined)."""
    fields = [
        "original_index",
        "problem",
        "answer",
        "topic",
        "answer_type",
        "multi_step",
        "requires_reasoning",
        "urls",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in pilot.rows:
            record = {
                "original_index": row.original_index,
                "problem": row.problem,
                "answer": row.answer,
                "topic": row.topic,
                "answer_type": row.answer_type,
                "multi_step": str(row.multi_step),
                "requires_reasoning": str(row.requires_reasoning),
                "urls": ",".join(row.urls),
            }
            if mutate:
                record = mutate(record)
            if record:
                writer.writerow(record)
    return path


def test_check_csv_agrees_with_a_matching_file(pilot: data.Pilot, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "ok.csv", pilot)
    assert data.check_csv(pilot, path, check_hash=False) == []
    assert data.check_csv(pilot, path)  # a different file has a different sha256


def test_check_csv_reports_a_changed_answer(pilot: data.Pilot, tmp_path: Path) -> None:
    def mutate(record: dict[str, Any]) -> dict[str, Any]:
        if record["original_index"] == 3503:
            record["answer"] = "P. W. Botha"
        return record

    problems = data.check_csv(pilot, write_csv(tmp_path / "bad.csv", pilot, mutate), check_hash=False)
    assert problems is not None and len(problems) == 1
    assert "row 3503" in problems[0] and "answer" in problems[0]


def test_check_csv_reports_a_missing_row_and_changed_urls(pilot: data.Pilot, tmp_path: Path) -> None:
    def mutate(record: dict[str, Any]) -> dict[str, Any] | None:
        if record["original_index"] == 69:
            return None
        if record["original_index"] == 3010:
            record["urls"] = "https://example.org/"
        return record

    problems = data.check_csv(pilot, write_csv(tmp_path / "bad.csv", pilot, mutate), check_hash=False)
    assert problems is not None
    assert any("row 69: not in the CSV" in p for p in problems)
    assert any("row 3010" in p and "urls" in p for p in problems)


def test_check_csv_is_none_without_a_file(pilot: data.Pilot, tmp_path: Path) -> None:
    assert data.check_csv(pilot, tmp_path / "missing.csv") is None


def test_rows_match_the_pinned_csv_when_it_is_cached(pilot: data.Pilot) -> None:
    problems = data.check_csv(pilot)
    if problems is None:
        pytest.skip(
            "the pinned CSV is not in the Hugging Face cache; run: uv run python agent_ability/web/fetch.py"
        )
    assert problems == []


def test_split_urls_round_trips_the_awkward_fields() -> None:
    field = "https://a.org/x,'https://b.org/y',https://c.org/z',http://d.org/"
    assert data.split_urls(field) == [
        "https://a.org/x",
        "'https://b.org/y'",
        "https://c.org/z'",
        "http://d.org/",
    ]
    assert ",".join(data.split_urls(field)) == field


# -- fetch.py ---------------------------------------------------------------------------------------------------


def test_fetch_pins_one_revision() -> None:
    assert (fetch.REPO_ID, fetch.FILENAME) == ("google/simpleqa-verified", "simpleqa_verified.csv")
    assert len(fetch.REVISION) == 40 and set(fetch.REVISION) <= set("0123456789abcdef")


def test_fetch_sha256_and_empty_cache(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_bytes(b"abc")
    assert fetch.sha256_file(sample) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert fetch.cached_csv(tmp_path / "empty-cache") is None  # a lookup, never a download
