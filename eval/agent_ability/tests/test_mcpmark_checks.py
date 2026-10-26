"""The MCPMark checker: every untouched start workspace fails verify.py, every scripted oracle passes,
and the start data is extracted safely and only after its sha256 matches the manifest."""

from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest
from marks import needs_data, needs_smoke_venv
from mcpmark_oracles import ORACLES

from smoke import checks, data, paths, tasks

pytestmark = [needs_data, needs_smoke_venv]


def test_oracles_cover_the_suite() -> None:
    assert sorted(ORACLES) == sorted(f"mcpmark/{n}" for n in tasks.MCPMARK_TASKS)


@pytest.mark.parametrize("name", tasks.MCPMARK_TASKS)
def test_untouched_fails_and_oracle_passes(name: str, tmp_path: Path) -> None:
    task = tasks.mcpmark_task(name)
    ws = tmp_path / "ws"
    tasks.build_workspace(task, ws)
    assert not (ws / "__MACOSX").exists()

    untouched = checks.check(task, ws, tmp_path / "untouched")
    assert not untouched["passed"], f"FLAG: {name} passes verify.py on the untouched start workspace"

    ORACLES[task.id](ws)
    solved = checks.check(task, ws, tmp_path / "solved")
    assert solved["passed"], f"{name}: the scripted oracle fails verify.py ({solved['detail']})"


def test_prompt_follows_the_mcpmark_recipe(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    task = tasks.mcpmark_task("file_property/largest_rename")
    description = (paths.MCPMARK_EASY / task.name / "description.md").read_text()

    text = tasks.prompt(task, ws)
    mcp_text = tasks.prompt(task, ws, mcp_arm=True)

    note = "\n\nNote: Based on your understanding, solve the task all at once by yourself, don't ask for my opinions on anything."
    assert text == f"{description}{note}\n\nWork in {ws}. You can run shell commands and read and write files there.\n"
    assert mcp_text == f"{description}{note}\n\nWork in {ws}.\n"


def _zip_with(entries: list[tuple[str, bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            zf.writestr(info, payload)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "entry",
    [
        ("cat/../escape.txt", b"x", stat.S_IFREG | 0o644),
        ("/cat/abs.txt", b"x", stat.S_IFREG | 0o644),
        ("cat/link", b"/etc/passwd", stat.S_IFLNK | 0o777),
        ("other/file.txt", b"x", stat.S_IFREG | 0o644),
    ],
)
def test_unsafe_zip_entries_are_refused(entry: tuple[str, bytes, int], tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_with([("cat/ok.txt", b"fine", stat.S_IFREG | 0o644), entry]))

    with pytest.raises(data.DataError):
        data.extract_category(archive, "cat", tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


def test_a_zip_that_differs_from_the_manifest_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    envs = tmp_path / "envs"
    envs.mkdir()
    (envs / "papers.zip").write_bytes(b"not the pinned bytes")
    (envs / "manifest.json").write_text('{"files": {"papers.zip": {"sha256": "' + "0" * 64 + '"}}}')
    monkeypatch.setattr(paths, "MCPMARK_ENVS", envs)
    data.verified_zip.cache_clear()
    try:
        with pytest.raises(data.DataError, match="sha256"):
            data.verified_zip("papers")
    finally:
        data.verified_zip.cache_clear()


def test_pinned_data_is_intact() -> None:
    assert data.integrity_problems(sorted({tasks.mcpmark_task(n).category for n in tasks.MCPMARK_TASKS})) == []
