"""Declared-equivalent harness epochs (pins.toml `[[epoch.equivalent]]`, DESIGN §6.5) in RESULTS.md, the
comparison against the reference, and the results page's protocol ids: one epoch under its canonical id."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_speed_report import GATE_TASKS, cap_run

from modeleval import paths, publish, report

TREES = '["' + "4" * 40 + '", "' + "5" * 40 + '"]'


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """The reference's gate set in epoch e-old, a new model's in e-new: it loses five tasks in two harnesses."""
    root = tmp_path / "results"
    cap_run(root, "qwen-ref", "20261027-010000", {}, epoch="e-old", reference=True, display="Qwen ref")
    lost = {t: {"claude_code": False, "hermes": False} for t in GATE_TASKS[:5]}
    cap_run(root, "flash", "20261028-010000", lost, epoch="e-new", display="Flash")
    return root


def _declare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *decls: tuple[str, str]) -> None:
    backend = tmp_path / "backend"
    backend.mkdir(exist_ok=True)
    text = '[epoch]\nid = "e-new"\n'
    for new, same_as in decls:
        text += (f'\n[[epoch.equivalent]]\nnew = "{new}"\nsame_as = "{same_as}"\nreason = "a comment reworded"\n'
                 f"trees = {{ bench = {TREES} }}\n")
    (backend / "pins.toml").write_text(text)
    monkeypatch.setattr(paths, "REPO", backend)


def test_without_a_declaration_the_epochs_stay_apart(results: Path, tmp_path: Path, monkeypatch) -> None:
    _declare(tmp_path, monkeypatch)
    md = report.build_report(results)
    assert "harness epoch e-old)" in md and "harness epoch e-new)" in md
    assert "No reference run in this epoch" in md and "Declared-equivalent" not in md


def test_equivalent_epochs_share_one_table_and_are_compared(results: Path, tmp_path: Path, monkeypatch) -> None:
    _declare(tmp_path, monkeypatch, ("e-new", "e-old"))
    md = report.build_report(results)
    tables = md.split("## Setting and notes")[0]
    assert tables.count("## Capability on mac: gate set") == 1
    assert "## Capability on mac: gate set (protocol 1, harness epoch e-old)" in tables and "e-new" not in tables
    row = next(ln for ln in md.splitlines() if ln.startswith("| Flash |") and "pairs)" in ln)
    assert "| 22 (66 pairs) | 56 vs 66 | 0 | 5 | 17 |" in row and "**worse**" in row
    assert "harness epoch(s): e-old." in md
    assert md.count("e-new counts as e-old (a comment reworded)") == 1   # the one footnote


def test_a_chain_maps_to_its_oldest_id(results: Path, tmp_path: Path, monkeypatch) -> None:
    cap_run(results, "nanbeige", "20261029-010000", {}, epoch="e-newer", display="Nanbeige")
    _declare(tmp_path, monkeypatch, ("e-mid", "e-old"), ("e-new", "e-mid"), ("e-newer", "e-new"))
    md = report.build_report(results)
    tables = md.split("## Setting and notes")[0]
    assert tables.count("## Capability on mac: gate set") == 1 and "harness epoch e-old)" in tables
    assert "| Nanbeige |" in tables and "| Flash |" in tables
    assert {report.canonical_epoch(e) for e in ("e-old", "e-mid", "e-new", "e-newer")} == {"e-old"}


def test_protocol_ids_match_the_canonical_epoch(results: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publish, "CAP_PROTOCOLS", {"capability-9": {"protocol_version": 1, "epoch": "e-old",
                                                                    "title": "test"}})
    runs = {r.model_key: r for r in report.discover(results)}
    _declare(tmp_path, monkeypatch)
    assert publish.cap_protocol(runs["flash"]) is None
    _, msgs = publish.rows(runs["flash"])
    assert any("no capability protocol id" in m and "e-new" in m for m in msgs)
    _declare(tmp_path, monkeypatch, ("e-new", "e-old"))
    assert publish.cap_protocol(runs["flash"]) == publish.cap_protocol(runs["qwen-ref"]) == "capability-9"
    written, msgs = publish.rows(runs["flash"])
    [(_, row)] = written
    assert msgs == [] and row["protocol"] == "capability-9"
    assert row["notes"].count("Harness epoch e-new, declared equivalent to e-old (DESIGN §6.5).") == 1
    [(_, ref_row)] = publish.rows(runs["qwen-ref"])[0]
    assert "declared equivalent" not in ref_row.get("notes", "")
