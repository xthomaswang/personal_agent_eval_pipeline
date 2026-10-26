"""Access to the pinned, untrusted task data: integrity checks, exercise files, safe extraction.

Nothing here executes downloaded content. Files are copied only if they are regular files (never
symlinks), and zip entries are extracted only below the destination.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from smoke import paths


class DataError(RuntimeError):
    """The pinned data is missing, altered, or not what the suite expects."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> tuple[str | None, bool | None]:
    if not (repo / ".git").exists():
        return None, None
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, env=env
    )
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, env=env
    )
    if head.returncode != 0 or status.returncode != 0:
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def zip_manifest() -> dict:
    path = paths.MCPMARK_ENVS / "manifest.json"
    if not path.exists():
        raise DataError(f"{path} is missing; run scripts/fetch_data.py")
    return json.loads(path.read_text())


@functools.cache
def verified_zip(category: str) -> Path:
    """The category's start zip, after checking its sha256 against manifest.json (once per process)."""
    entry = zip_manifest()["files"].get(f"{category}.zip")
    path = paths.MCPMARK_ENVS / f"{category}.zip"
    if entry is None or not path.exists():
        raise DataError(f"{path} or its manifest entry is missing; run scripts/fetch_data.py")
    if sha256_file(path) != entry["sha256"]:
        raise DataError(f"{path}: sha256 differs from manifest.json")
    return path


def integrity_problems(categories: list[str]) -> list[str]:
    """Everything wrong with the pinned data, as messages (empty when it is all as pinned)."""
    problems = []
    for name, repo, sha in (
        ("polyglot-benchmark", paths.POLYGLOT_DIR, paths.POLYGLOT_SHA),
        ("mcpmark", paths.MCPMARK_DIR, paths.MCPMARK_SHA),
    ):
        head, dirty = _git_state(repo)
        if head is None:
            problems.append(f"{name}: no git checkout at {repo}; run scripts/fetch_data.py")
        elif head != sha or dirty:
            problems.append(f"{name}: {repo} is at {head} (dirty={dirty}), expected a clean {sha}")
    for category in categories:
        try:
            verified_zip(category)
        except DataError as exc:
            problems.append(str(exc))
    return problems


def copy_regular_file(src: Path, dst: Path) -> None:
    """Copy one file; refuse symlinks and anything that is not a regular file."""
    mode = src.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise DataError(f"refusing to copy {src}: not a regular file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _safe_relative(name: str) -> PurePosixPath:
    rel = PurePosixPath(name)
    if rel.is_absolute() or "\\" in name or any(part in ("..", "") for part in rel.parts):
        raise DataError(f"unsafe path {name!r}")
    return rel


def extract_category(zip_path: Path, category: str, dest: Path) -> None:
    """Unzip `<category>/...` into `dest` (a new directory) as MCPMark does: `unzip`, then drop
    `__MACOSX`. File mtimes from the zip are kept. Absolute paths, `..` and symlinks are refused."""
    dest.mkdir(parents=True)
    prefix = f"{category}/"
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("__MACOSX/"):
                continue
            if not name.startswith(prefix):
                raise DataError(f"{zip_path.name}: unexpected entry {name!r} outside {prefix}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise DataError(f"{zip_path.name}: refusing symlink entry {name!r}")
            rel_name = name[len(prefix) :].rstrip("/")
            if not rel_name:
                continue
            target = dest.joinpath(*_safe_relative(rel_name).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("xb") as out:
                shutil.copyfileobj(src, out)
            stamp = time.mktime((*info.date_time, 0, 0, -1))
            os.utime(target, (stamp, stamp))


@dataclass(frozen=True)
class Exercise:
    """One Aider polyglot Python exercise, as its .meta/config.json describes it."""

    name: str
    directory: Path
    solution: tuple[str, ...]
    test: tuple[str, ...]
    example: tuple[str, ...]
    editor: tuple[str, ...]

    @property
    def workspace_files(self) -> tuple[str, ...]:
        """What an agent may see: the stub, the tests and any editor helpers. Never .meta/.docs."""
        return self.solution + self.test + self.editor

    def instructions(self) -> str:
        """introduction + instructions + instructions.append, concatenated as aider's benchmark does."""
        docs = self.directory / ".docs"
        text = ""
        if (docs / "introduction.md").exists():
            text += (docs / "introduction.md").read_text()
        text += (docs / "instructions.md").read_text()
        if (docs / "instructions.append.md").exists():
            text += (docs / "instructions.append.md").read_text()
        return text


def exercise_names() -> list[str]:
    if not paths.POLYGLOT_PYTHON.is_dir():
        raise DataError(f"{paths.POLYGLOT_PYTHON} is missing; run scripts/fetch_data.py")
    return sorted(p.name for p in paths.POLYGLOT_PYTHON.iterdir() if p.is_dir())


@functools.cache
def exercise(name: str) -> Exercise:
    directory = paths.POLYGLOT_PYTHON / name
    config_path = directory / ".meta" / "config.json"
    if not config_path.is_file():
        raise DataError(f"unknown Aider exercise {name!r} (no {config_path})")
    files = json.loads(config_path.read_text())["files"]

    def names(key: str) -> tuple[str, ...]:
        return tuple(str(_safe_relative(n)) for n in files.get(key, []))

    return Exercise(
        name=name,
        directory=directory,
        solution=names("solution"),
        test=names("test"),
        example=names("example"),
        editor=names("editor"),
    )
