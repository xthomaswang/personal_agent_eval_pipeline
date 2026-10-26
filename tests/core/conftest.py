"""Shared fixtures for the core tests: every path the pipeline touches is redirected into tmp_path.

Nothing here starts a real model server, takes the real GPU lock or binds a port outside the
ephemeral range: `paths.*` is patched to temporary directories and an ephemeral port, the lock
script, quietlib and `omlx` are fakes from tests/core/fakes/, and GPU_LOCK_PATH points into tmp_path.
"""
from __future__ import annotations

import os
import shutil
import socket
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeleval import paths, prepare, quiet
from modeleval.registry import parse_models

FAKES = Path(__file__).parent / "fakes"
FORBIDDEN = {8500, 8703} | set(range(8600, 8700))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port not in FORBIDDEN
    return port


def install_exec(src: Path, dest: Path, interpreter: str | None = None) -> Path:
    """Copy a fake into place as an executable, with `interpreter` as its shebang if given."""
    text = src.read_text()
    if interpreter:
        text = f"#!{interpreter} -B\n" + text
    dest.write_text(text)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def model_table(ckpt: Path, **over) -> dict:
    t = {"display": "Fake 1B + MTP", "served_id": "Fake-1B-mtp", "path": str(ckpt),
         "source": {"hf_repo": "org/Fake-1B", "revision": "0" * 40}, "quant": "oQ5", "spec": "mtp",
         "vision": False, "reference": True, "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
         "proxy_sampling": "inject:qwen3.8-gen-config",
         "server_log_expect": ["Speculative backend selected for .*: Lightning MTP", "MTP path activated"]}
    t.update(over)
    return t


@pytest.fixture
def env(tmp_path, monkeypatch) -> SimpleNamespace:
    """A sandbox: temp BASE/models/results/runs/scripts, fake omlx/quietlib/lock, ephemeral port."""
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    shutil.copy(FAKES / "quietlib.py", scripts / "quietlib.py")
    install_exec(FAKES / "gpu_lock.sh", scripts / "gpu_lock.sh")
    omlx_repo = root / "omlx_repo"
    (omlx_repo / ".venv/bin").mkdir(parents=True)
    exe = install_exec(FAKES / "fake_omlx.py", omlx_repo / ".venv/bin/omlx", sys.executable)
    ckpt = root / "models/oq/Fake-1B-mtp"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"\0" * 1024)
    models_dir = root / "models/hf-bench"
    models_dir.mkdir(parents=True)
    (models_dir / "Fake-1B-mtp").symlink_to(ckpt)
    port = free_port()
    results = root / "results"
    over = {
        "OMLX_REPO": omlx_repo, "OMLX_EXE": exe, "OMLX_PY": Path(sys.executable),
        "BASE": omlx_repo / ".modeleval-home", "MODELS_DIR": models_dir, "PORT": port,
        "BASE_URL": f"http://127.0.0.1:{port}/v1", "RESULTS": results, "UNVERIFIED": results / "_unverified",
        "RUNS": root / "runs", "VENDOR": root / "vendor", "SCRIPTS": scripts, "GPU_LOCK": root / "gpu-lock",
        "HOLD_FILE": root / "runs/hold", "HF_HOME": root / "hf", "WS_ROOT": root / "ws",
    }
    for k, v in over.items():
        monkeypatch.setattr(paths, k, v)
    # the agent-benchmark session's real port may be up while the tests run: convert's "no model
    # server" check looks at a free ephemeral port instead
    monkeypatch.setattr(prepare, "PEER_PORT", free_port())
    monkeypatch.setenv("GPU_LOCK_PATH", str(root / "gpu-lock"))
    monkeypatch.setenv("FAKE_LOCK_LOG", str(root / "lock.log"))
    monkeypatch.setenv("FAKE_QUIET_LOG", str(root / "quiet.log"))
    monkeypatch.setenv("FAKE_QUIET_CHECKS", str(root / "quiet-checks"))
    monkeypatch.setenv("FAKE_QUIET_DISTURB", str(root / "disturb"))
    monkeypatch.setenv("FAKE_OMLX_STATE", str(root / "omlx-requests"))
    monkeypatch.setenv("FAKE_OMLX_LINES", "Speculative backend selected for /x/Fake-1B-mtp: Lightning MTP "
                       "(model_type=qwen3_5, active)\nMTP path activated for uid=0 (batch=1)")
    monkeypatch.setattr(quiet, "MONITOR_INTERVAL", 0.2)
    monkeypatch.setattr(quiet, "MARGIN", 0.3)
    models = parse_models({"models": {"fake-1b": model_table(ckpt)}})
    return SimpleNamespace(root=root, port=port, exe=exe, ckpt=ckpt, models=models, model=models["fake-1b"],
                           scripts=scripts, results=results, runs=root / "runs", lock=root / "gpu-lock")


@pytest.fixture(autouse=True)
def _no_real_lock():
    """Belt and braces: no test may create or remove the shared GPU lock."""
    real = Path.home() / "Developer/mlsys/experiments/.gpu-lock"
    before = real.exists() and real.stat().st_mtime
    yield
    after = real.exists() and real.stat().st_mtime
    assert before == after, "a test touched the real GPU lock"
    assert os.environ.get("GPU_LOCK_PATH") != str(real)
