"""Fetch the pinned smoke-test data into ~/Developer/ml_source/datasets.

What it fetches:
- Aider-AI/polyglot-benchmark at one commit (a depth-1 fetch of exactly that commit);
- eval-sys/mcpmark at one commit (same);
- the MCPMark filesystem start zips that the easy tasks' meta.json files name (stateUrl), plus a
  manifest.json with each zip's URL, size, sha256 and HTTP Last-Modified.

Everything fetched is untrusted data: this script only downloads, checks out and hashes it, and
never executes any of it. It uses only the standard library and git, so it runs with `python -I`
(no script directory or current directory on sys.path). Re-running is safe: present, matching
data is left alone, and a mismatch is an error rather than an overwrite.

Usage (from the harness-smoke directory):
    .venv/bin/python -I scripts/fetch_data.py [--data-root DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

POLYGLOT = ("https://github.com/Aider-AI/polyglot-benchmark.git", "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f")
MCPMARK = ("https://github.com/eval-sys/mcpmark.git", "cd45b7f57923b9b3985467f5139927575f83141c")
ENV_CATEGORIES = (
    "file_context",
    "file_property",
    "folder_structure",
    "legal_document",
    "papers",
    "student_database",
)
ZIP_URL = "https://storage.mcpmark.ai/filesystem/{category}.zip"
EASY_TASKS = Path("tasks") / "filesystem" / "easy"

GIT_ENV = {
    **os.environ,
    "GIT_LFS_SKIP_SMUDGE": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], env=GIT_ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def _note(dataset_dir: Path, line: str) -> None:
    with (dataset_dir / "FETCHED_AT.txt").open("a") as f:
        f.write(line + "\n")


def fetch_repo(url: str, sha: str, dest: Path) -> str:
    """Check out exactly `sha` of `url` into the new directory `dest`."""
    if dest.exists():
        if not (dest / ".git").exists():
            raise SystemExit(f"{dest} exists but is not a git checkout; remove it first")
        head = _git("-C", str(dest), "rev-parse", "HEAD")
        dirty = _git("-C", str(dest), "status", "--porcelain")
        if head != sha or dirty:
            raise SystemExit(f"{dest} is at {head} (dirty={bool(dirty)}), expected a clean {sha}")
        return "present"
    dest.mkdir(parents=True)
    _git("init", "-q", str(dest))
    _git("-C", str(dest), "remote", "add", "origin", url)
    _git("-C", str(dest), "fetch", "-q", "--depth", "1", "--no-tags", "origin", sha)
    _git("-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", "FETCH_HEAD")
    head = _git("-C", str(dest), "rev-parse", "HEAD")
    if head != sha:
        raise SystemExit(f"{dest}: checked out {head}, expected {sha}")
    _note(dest.parent, f"{_now()} {url} {sha} -> {dest.name}")
    return "fetched"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_urls(mcpmark_dir: Path) -> dict[str, str]:
    """category -> stateUrl, read from the easy tasks' meta.json; must be the six expected URLs."""
    urls: dict[str, str] = {}
    for meta_path in sorted((mcpmark_dir / EASY_TASKS).glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        category = meta_path.parent.parent.name
        url = (meta.get("meta_data") or {}).get("stateUrl")
        if urls.setdefault(category, url) != url:
            raise SystemExit(f"{category}: tasks disagree on stateUrl ({urls[category]} vs {url})")
    expected = {c: ZIP_URL.format(category=c) for c in ENV_CATEGORIES}
    if urls != expected:
        raise SystemExit(f"unexpected stateUrl set:\n{json.dumps(urls, indent=2)}")
    return urls


def fetch_zips(urls: dict[str, str], dest: Path) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    status: dict[str, str] = {}
    for category, url in urls.items():
        name = f"{category}.zip"
        path = dest / name
        known = manifest["files"].get(name)
        if known and path.exists():
            if _sha256(path) != known["sha256"]:
                raise SystemExit(f"{path}: sha256 differs from manifest.json")
            status[name] = "present"
            continue
        part = path.with_suffix(".zip.part")
        digest = hashlib.sha256()
        size = 0
        request = urllib.request.Request(url, headers={"User-Agent": "harness-smoke-fetch/1"})
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            headers = response.headers
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
                digest.update(block)
                size += len(block)
        expected_size = headers.get("Content-Length")
        if expected_size is not None and int(expected_size) != size:
            part.unlink()
            raise SystemExit(f"{url}: got {size} bytes, Content-Length says {expected_size}")
        part.rename(path)
        manifest["files"][name] = {
            "url": url,
            "bytes": size,
            "sha256": digest.hexdigest(),
            "last_modified": headers.get("Last-Modified"),
            "etag": headers.get("ETag"),
            "fetched_at_utc": _now(),
        }
        status[name] = "fetched"
    manifest["source"] = (
        f"stateUrl in {MCPMARK[0]}@{MCPMARK[1]}:{EASY_TASKS.as_posix()}/*/*/meta.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if "fetched" in status.values():
        _note(dest, f"{_now()} {', '.join(n for n, s in status.items() if s == 'fetched')}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("HARNESS_SMOKE_DATA", Path.home() / "Developer" / "ml_source" / "datasets")),
    )
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    polyglot_dir = root / "aider-polyglot-benchmark" / POLYGLOT[1][:12]
    mcpmark_dir = root / "mcpmark" / MCPMARK[1][:12]
    print(f"polyglot-benchmark: {fetch_repo(*POLYGLOT, polyglot_dir)} at {polyglot_dir}")
    print(f"mcpmark: {fetch_repo(*MCPMARK, mcpmark_dir)} at {mcpmark_dir}")
    envs_dir = root / "mcpmark" / "filesystem-envs"
    for name, state in fetch_zips(zip_urls(mcpmark_dir), envs_dir).items():
        print(f"{name}: {state}")
    print(f"manifest: {envs_dir / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
