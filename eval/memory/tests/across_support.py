"""Helpers the across-chats tests share: a launcher around the fake gateway, a fake localmelo checkout, a study."""
import os
import subprocess
import sys
from pathlib import Path

from memory.across import client, data, run
from memory.tests.across_fake import FakeGateway, mini_raw

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
        gateway = FakeGateway(spec.home, spec.x_pairs, port=spec.port, **self.kw)
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


