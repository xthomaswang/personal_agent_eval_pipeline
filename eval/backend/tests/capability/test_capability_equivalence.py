"""Declared harness-epoch equivalences (DESIGN §6.5): pins.toml `[[epoch.equivalent]]`, the canonical
mapping, and the mechanical check that a declared tree pair differs in comments only. Every repository
here is a temporary one; no result may carry the text of a changed line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modeleval import cli, registry
from modeleval.capability import epoch as epochs

from .fakes import _git as git

SECRET = "zz-old-wording-zz"   # stands for the removed comment's text: it must never reach a result
BASE = {
    "bench/manifest.py": f'"""Doc."""\nimport os\n\n# see {SECRET} for details\nX = 1\nTEXT = """\n# inside a string\n"""\n',
    "bench/util.py": "def f():\n    return 1\n",
    "bench/README.md": "# Title\n",
}


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    _commit(repo, BASE)
    return repo, git(repo, "rev-parse", "HEAD:bench")


def _commit(repo: Path, files: dict[str, str | None]) -> str:
    """Write (or, for None, delete) files, commit, and return the bench tree."""
    for rel, text in files.items():
        if text is None:
            (repo / rel).unlink()
        else:
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    return git(repo, "rev-parse", "HEAD:bench")


def _edit(path: str, old: str, new: str) -> dict[str, str]:
    return {path: BASE[path].replace(old, new)}


def _pins(repo: Path, *decls: dict[str, Any], epoch_id: str = "bbbbbbbbbbbb") -> dict[str, Any]:
    return {"epoch": {"id": epoch_id, "equivalent": list(decls)}, "snapshot": {"repo": str(repo)}}


def _decl(old_tree: str, new_tree: str, new: str = "bbbbbbbbbbbb", same_as: str = "aaaaaaaaaaaa") -> dict[str, Any]:
    return {"new": new, "same_as": same_as, "reason": "one comment reworded; no code change",
            "trees": {"bench": [old_tree, new_tree]}}


def _verify(repo: Path, old: str, new: str, tmp_path: Path) -> dict[str, Any]:
    [result] = epochs.verify_equivalences(_pins(repo, _decl(old, new)), vendor_root=tmp_path / "vendor")
    assert SECRET not in json.dumps(result) and "X = " not in json.dumps(result)
    return result


def test_a_comment_only_diff_is_accepted(tmp_path: Path) -> None:
    repo, old = _repo(tmp_path)
    text = BASE["bench/manifest.py"].replace(f"# see {SECRET} for details", "# see the notes for details")
    new = _commit(repo, {"bench/manifest.py": text.replace("import os\n", "import os\n   \n")})   # + a blank line
    assert epochs.comment_only_problems(repo, old, new) == []
    result = _verify(repo, old, new, tmp_path)
    assert result["status"] == "verified", result
    assert "1 Python file, 3 changed lines, comments only" in result["detail"]
    assert "components not compared" in result["detail"]


@pytest.mark.parametrize("change, why", [
    (_edit("bench/manifest.py", "X = 1", "X = 2"), "2 of 2 changed lines are not comments or blank"),
    (_edit("bench/manifest.py", f"# see {SECRET} for details", "X = 3  # " + SECRET), "1 of 2 changed lines are not comments"),
    (_edit("bench/manifest.py", "# inside a string", "# inside a string, reworded"), "code tokens differ"),
    (_edit("bench/util.py", "    return 1", "    # return 1"), "are not comments or blank"),
    (_edit("bench/README.md", "# Title", "# Other title"), "not a Python file"),
    ({"bench/extra.py": "# only a comment\n"}, "not an edit of an existing text file"),
    ({"bench/util.py": None}, "not an edit of an existing text file"),
])
def test_a_code_diff_is_rejected(tmp_path: Path, change: dict[str, str | None], why: str) -> None:
    repo, old = _repo(tmp_path)
    new = _commit(repo, change)
    result = _verify(repo, old, new, tmp_path)
    assert result["status"] == "failed" and why in result["detail"], result["detail"]


def test_a_comment_that_changes_the_source_encoding_is_rejected(tmp_path: Path) -> None:
    """Line rule and tokens together: a coding cookie is a comment line that changes how the file is read."""
    repo, old = _repo(tmp_path)
    new = _commit(repo, {"bench/util.py": "def f():\n    return 1\n# trailing\n"})
    assert _verify(repo, old, new, tmp_path)["status"] == "verified"
    newer = _commit(repo, {"bench/util.py": "# -*- coding: latin-1 -*-\ndef f():\n    return 1\n# trailing\n"})
    result = _verify(repo, new, newer, tmp_path)
    assert result["status"] == "failed" and "code tokens differ" in result["detail"]


def test_a_missing_tree_object_leaves_the_reason_unverified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, old = _repo(tmp_path)
    gone = "1" * 40                                    # what a gc leaves: the old tree object is not there
    pins = _pins(repo, _decl(gone, old))
    [result] = epochs.verify_equivalences(pins, vendor_root=tmp_path / "vendor")
    assert result["status"] == "unverified" and "tree 11111111 is not in" in result["detail"]
    monkeypatch.setattr(registry, "load_pins", lambda path=None: pins)
    [(level, name, detail)] = cli.equivalence_checks()
    assert level == "WARN" and "bbbbbbbbbbbb declared equivalent to aaaaaaaaaaaa" in name
    assert "unverified" in detail and "recorded reason: one comment reworded; no code change" in detail


def test_the_snapshot_records_must_differ_in_the_listed_trees_only(tmp_path: Path) -> None:
    repo, old = _repo(tmp_path)
    new = _commit(repo, _edit("bench/manifest.py", f"# see {SECRET} for details", "# see the notes"))
    vendor = tmp_path / "vendor"
    old_comps = {"format": 1, "trees": {"bench": old, "smoke": "2" * 40, "shims": "3" * 40}, "claude_code": "2.1.302"}

    def record(comps: dict[str, Any]) -> str:
        epoch_id = epochs._sha256_json(comps)[: epochs.ID_LENGTH]
        (vendor / epoch_id).mkdir(parents=True, exist_ok=True)
        (vendor / epoch_id / epochs.EPOCH_FILE).write_text(json.dumps({"id": epoch_id, "components": comps}))
        return epoch_id

    old_id = epochs._sha256_json(old_comps)[: epochs.ID_LENGTH]   # only the new side has a record
    new_comps = json.loads(json.dumps(old_comps))
    new_comps["trees"]["bench"] = new
    new_id = record(new_comps)
    [ok] = epochs.verify_equivalences(_pins(repo, _decl(old, new, new_id, old_id)), vendor_root=vendor)
    assert ok["status"] == "verified" and f"only the listed trees differ (vendor/{new_id})" in ok["detail"]
    # another component changed as well (a new Claude Code): the same trees no longer explain the new id
    other = record({**new_comps, "claude_code": "2.1.303"})
    [bad] = epochs.verify_equivalences(_pins(repo, _decl(old, new, other, old_id)), vendor_root=vendor)
    assert bad["status"] == "failed" and "something else differs" in bad["detail"]


def test_canonical_ids_follow_the_chain_to_the_oldest() -> None:
    t = ["4" * 40, "5" * 40]
    raw = {"epoch": {"id": "c", "equivalent": [
        {"new": "b", "same_as": "a", "reason": "r", "trees": {"bench": t}},
        {"new": "c", "same_as": "b", "reason": "r", "trees": {"smoke": t}},
    ]}}
    assert epochs.canonical_map(raw) == {"b": "a", "c": "a"}
    assert [epochs.canonical(x, raw) for x in ("a", "b", "c", "d", None)] == ["a", "a", "a", "d", None]
    assert epochs.same_epoch("c", "a", raw) and epochs.same_epoch("b", "c", raw)
    assert not epochs.same_epoch("c", "d", raw) and not epochs.same_epoch(None, "a", raw)
    assert epochs.canonical_map({"epoch": "plain-id"}) == {} and epochs.canonical_map({}) == {}


@pytest.mark.parametrize("bad, message", [
    ({"new": "b", "same_as": "b"}, "two different epoch ids"),
    ({"trees": {"bench": ["4" * 40]}}, "two different full hashes"),
    ({"trees": {"bench": ["4" * 12, "5" * 12]}}, "two different full hashes"),
    ({"trees": {"docs": ["4" * 40, "5" * 40]}}, "not a snapshot part"),
    ({"reason": " "}, "reason must be"),
    ({"extra": 1}, "exactly the keys"),
])
def test_malformed_declarations_are_errors(bad: dict[str, Any], message: str) -> None:
    decl = {"new": "b", "same_as": "a", "reason": "r", "trees": {"bench": ["4" * 40, "5" * 40]}, **bad}
    with pytest.raises(epochs.EpochError, match=message):
        epochs.canonical_map({"epoch": {"equivalent": [decl]}})


def test_loops_and_duplicates_are_errors() -> None:
    t = {"bench": ["4" * 40, "5" * 40]}
    loop = [{"new": "a", "same_as": "b", "reason": "r", "trees": t}, {"new": "b", "same_as": "a", "reason": "r", "trees": t}]
    with pytest.raises(epochs.EpochError, match="loops back"):
        epochs.canonical_map({"epoch": {"equivalent": loop}})
    dup = [{"new": "a", "same_as": "b", "reason": "r", "trees": t}, {"new": "a", "same_as": "c", "reason": "r", "trees": t}]
    with pytest.raises(epochs.EpochError, match="more than once"):
        epochs.canonical_map({"epoch": {"equivalent": dup}})


def test_diff_and_cat_file_are_allowed_on_tree_objects_only(tmp_path: Path) -> None:
    for args in (("diff",), ("diff", "HEAD"), ("diff", "HEAD", "HEAD~1"), ("diff", "--cached", "4" * 40),
                 ("cat-file", "-p", "HEAD"), ("cat-file", "blob", "HEAD:x")):
        with pytest.raises(ValueError, match="not an allowed"):
            epochs._git(tmp_path, *args)


def test_the_repo_declaration_is_never_failed() -> None:
    """The committed pins.toml: ce99c7b5d0bd counts as the reference epoch 7582825160ad. Verified while both
    tree objects exist; after a gc only the recorded reason stands (unverified), never a failure."""
    raw = registry.load_pins()
    assert epochs.canonical("ce99c7b5d0bd", raw) == "7582825160ad"
    assert epochs.canonical(registry.epoch_id(raw), raw) == "4271c2f4a925"
    for result in epochs.verify_equivalences(raw):
        assert result["status"] in ("verified", "unverified"), result["detail"]
