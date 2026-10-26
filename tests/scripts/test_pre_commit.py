"""scripts/pre-commit in a throwaway git repo (tmp_path): it judges the staged content."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
USERS_DIR = "/Us" + "ers/"      # assembled at run time, so this file passes the check itself

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="no git")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], env=_env(), capture_output=True, text=True,
                          check=True).stdout


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    for name in ("redact.py", "pre-commit"):
        shutil.copy2(REPO / "scripts" / name, r / "scripts" / name)
    git(r, "init", "-q")
    return r


def hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(repo / "scripts/pre-commit")], cwd=repo, env=_env(), capture_output=True, text=True,
                          timeout=60)


def test_nothing_staged_passes(repo):
    assert hook(repo).returncode == 0


def test_clean_files_pass_including_the_scripts_themselves(repo):
    (repo / "notes with space.md").write_text("plain text, ~/Developer paths only\n", encoding="utf-8")
    git(repo, "add", "-A")
    r = hook(repo)
    assert r.returncode == 0, r.stdout + r.stderr


def test_home_path_is_refused_with_file_and_line(repo):
    (repo / "a.txt").write_text(f"ok\nlog {USERS_DIR}someone/x.log\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    r = hook(repo)
    assert r.returncode == 1
    assert "redact: a.txt:2: home path" in r.stdout and "refused by scripts/redact.py --check" in r.stderr


def test_staged_content_is_judged_not_the_working_tree(repo):
    f = repo / "a.txt"
    f.write_text("clean\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    f.write_text(f"dirty {USERS_DIR}someone/x\n", encoding="utf-8")   # not staged
    assert hook(repo).returncode == 0
    git(repo, "add", "a.txt")
    f.write_text("clean again\n", encoding="utf-8")                   # the staged version is dirty
    assert hook(repo).returncode == 1


def test_symlinks_and_binary_files_are_refused(repo):
    (repo / "target.txt").write_text("x\n", encoding="utf-8")
    os.symlink("target.txt", repo / "link.txt")
    (repo / "blob.bin").write_bytes(b"\x00\xff\xfe")
    git(repo, "add", "-A")
    r = hook(repo)
    assert r.returncode == 1
    assert "redact: link.txt: symlink" in r.stdout and "redact: blob.bin: not UTF-8" in r.stdout


def test_capability_allowlist(repo):
    task = repo / "results/m/r1/capability/claude_code/transpose"
    task.mkdir(parents=True)
    (task / "calls.jsonl").write_text('{"ok": 1}\n', encoding="utf-8")
    (repo / "results/m/r1/capability/results.json").write_text("[]\n", encoding="utf-8")
    git(repo, "add", "-A")
    assert hook(repo).returncode == 0
    (task / "prompt.md").write_text("the task\n", encoding="utf-8")
    git(repo, "add", "-A")
    r = hook(repo)
    assert r.returncode == 1 and "capability/claude_code/transpose/prompt.md: not on the capability" in r.stdout


def test_deleting_a_bad_file_passes(repo):
    (repo / "old.txt").write_text(f"{USERS_DIR}someone\n", encoding="utf-8")
    git(repo, "add", "old.txt")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-q", "--no-verify", "-m", "x")
    git(repo, "rm", "-q", "old.txt")
    assert hook(repo).returncode == 0


def test_a_file_that_became_a_symlink_is_refused(repo):
    """A typechange (T) is not in --diff-filter=ACMR: the hook must still see the new symlink."""
    (repo / "f.json").write_text("{}\n", encoding="utf-8")
    git(repo, "add", "f.json")
    git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", "f", "--no-verify")
    (repo / "f.json").unlink()
    (repo / "f.json").symlink_to("scripts/redact.py")
    git(repo, "add", "f.json")
    assert "T\tf.json" in git(repo, "diff", "--cached", "--name-status")
    r = hook(repo)
    assert r.returncode == 1 and "f.json: symlink" in r.stdout
