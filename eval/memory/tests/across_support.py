"""Helpers the across-chats tests share: a launcher around the fake gateway, a fake localmelo checkout, a study."""
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from memory.across import client, data, data_v2, run, server
from memory.tests.across_fake import MINI_SAVE_ONLY, MINI_TOOL_WORDS, FakeGateway, FakeModelServer, mini_raw, mini_raw_v2

EVAL_ROOT = Path(__file__).resolve().parents[2]
ARMS3 = ("M2", "L1", "N0")


class FakeProc:
    """What server.launch returns, around an in-process FakeGateway."""

    def __init__(self, gateway, spec):
        self.gateway, self.spec = gateway, spec
        self.base_url = gateway.base_url
        self.process = type("P", (), {"returncode": None})()

    def alive(self):
        return not self.gateway.stopped

    def token(self):
        return client.read_token(self.spec.home)

    def stop(self):
        self.gateway.stop()
        return 0


class Launcher:
    def __init__(self, **kw):
        self.kw, self.gateways, self.cells = kw, [], []

    def __call__(self, spec):
        try:
            gateway = FakeGateway(spec.home, spec.x_pairs, port=spec.port, **self.kw)
        except ValueError as exc:  # `melo serve` refusing a setting exits at once
            raise server.LaunchError(f"melo serve exited with 2:\n{exc}") from exc
        self.gateways.append(gateway)
        self.cells.append(spec.home.parent.name)
        return FakeProc(gateway, spec)


def git(repo, *args):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True, env=env).stdout.strip()


def make_repo(tmp_path: Path) -> Path:
    """A clean git checkout with a fake `.venv/bin/melo` (ignored by git)."""
    root = tmp_path / "localmelo"
    (root / ".venv" / "bin").mkdir(parents=True)
    melo = root / ".venv" / "bin" / "melo"
    melo.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(EVAL_ROOT)!r})\n"
                    "from memory.tests.across_fake import melo_main\nsys.exit(melo_main(sys.argv[1:]))\n")
    melo.chmod(0o755)
    (root / ".gitignore").write_text(".venv/\n")
    git(root, "init", "-q")
    git(root, "add", ".gitignore")
    git(root, "commit", "-q", "-m", "init", "--no-gpg-sign")
    return root


def make_study(tmp_path, repo, models, launcher=None, tasks=None, **kw):
    if "expect_sha" not in kw:
        kw["expect_sha"] = git(repo, "rev-parse", "HEAD")[:8]
    options = dict(
        run_dir=tmp_path / "run", tasks=tasks or data.parse(mini_raw(), None), arms=run.ARMS, arm_names=ARMS3, repeats=1,
        localmelo=repo, poll_s=0.01, base_url=models.base_url, model="chat-model",
        embedding_model="embed-model", launcher=launcher or Launcher(), probe_rss=lambda: {"pid": 7, "rss_kb": 2048, "loaded": ["chat-model"]},
        warm=lambda study, arm: {"chat_ms": 10.0, **({"embed_ms": 5.0} if arm.embedding else {})},
        snapshot=lambda study: {"sw_vers": "ProductName: test", "xcodebuild_version": "Xcode test"},
        scratch=tmp_path, sleep=lambda s: None,
    )
    options.update(kw)
    return run.Study(**options)


def v2_launcher(**kw):
    """A launcher whose gateways keep the SQLite store, call read-tier tools on the filler and save only the mini set's notes."""
    return Launcher(**{"store": True, "tool_words": MINI_TOOL_WORDS, "save_only": MINI_SAVE_ONLY, "delay": 0.002, **kw})


def v2_study(tmp_path, repo, models, launcher=None, **kw):
    """A study of the mini v2 set (two long-range and two follow-up items) on N0, M2 and FC."""
    kw.setdefault("arm_names", ("N0", "M2", "FC"))
    return make_study(tmp_path, repo, models, launcher or v2_launcher(), tasks=data_v2.parse(mini_raw_v2(), None), **kw)


@dataclass
class V2Run:
    study: run.Study
    launcher: Launcher
    tasks: data_v2.TaskSet
    code: int


_V2_RUN: V2Run | None = None


def v2_run(tmp_path_factory) -> V2Run:
    """One study of the mini v2 set on N0, M2, B0 and FC, run once per pytest session (a few seconds, with the timing
    proxy and request bodies that hold the window) and read, never changed, by several test modules."""
    global _V2_RUN
    if _V2_RUN is None:
        base = tmp_path_factory.mktemp("v2run")
        home = base / "realhome"
        home.mkdir()
        patch = pytest.MonkeyPatch()
        patch.setenv("HOME", str(home))
        models = FakeModelServer(["chat-model", "embed-model"])
        try:
            launcher = v2_launcher(model_calls=True)
            study = v2_study(base, make_repo(base), models, launcher, arm_names=("N0", "M2", "B0", "FC"))
            code = run.run_study(study)
        finally:
            models.stop()
            patch.undo()
        assert list(home.iterdir()) == []  # the real HOME was never used
        _V2_RUN = V2Run(study, launcher, study.tasks, code)
    return _V2_RUN

