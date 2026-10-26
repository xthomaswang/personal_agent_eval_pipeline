"""The integration sandbox: the real orchestrator, the real speed module and the real runner's checks.

Everything the pipeline touches is redirected into tmp_path, as in tests/core: `paths.*` points at
temporary directories and an ephemeral port, the lock script and quietlib are tests/core's fakes, and
GPU_LOCK_PATH points into tmp_path. What differs from tests/core is what is real: `omlx` is
tests/integration/mock_omlx.py (agentbench's mock server), so run.py drives the real speed.py and
agentbench end to end, and the capability runner is a fake that checks run.py's spec and environment
with the real runner's own code.
"""
from __future__ import annotations

import os
import shutil
import socket
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeleval import paths, quiet, run, speed
from modeleval.registry import parse_models

HERE = Path(__file__).parent
CORE_FAKES = HERE.parent / "core" / "fakes"
FORBIDDEN = {8500, 8703} | set(range(8600, 8700))

sys.path.insert(0, str(HERE.parent / "speed"))   # speedkit and the tokenizing responder

os.environ["GIT_OPTIONAL_LOCKS"] = "0"           # agentbench headers run `git status` on this repo
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port not in FORBIDDEN
    return port


def install_exec(src: Path, dest: Path, interpreter: str | None = None) -> Path:
    text = src.read_text()
    if interpreter:
        text = f"#!{interpreter} -B\n" + text
    dest.write_text(text)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def model_table(ckpt: Path) -> dict:
    """A Nanbeige-shaped model: no speculative decoding, text only, its own sampling."""
    return {"display": "Mock 1B", "served_id": "Mock-1B", "path": str(ckpt),
            "source": {"hf_repo": "org/Mock-1B", "revision": "0" * 40}, "quant": "oQ5", "family": "mock_1b",
            "quant_dir": "oq5", "spec": "none",
            "vision": False, "reference": True, "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0},
            "proxy_sampling": "inject:omlx-default", "server_log_expect": ["Loaded model: Mock-1B "]}


@pytest.fixture(scope="session")
def mini_fixtures(tmp_path_factory) -> Path:
    from speedkit import mini_fixtures as build
    return build(tmp_path_factory.mktemp("fixtures"))


@pytest.fixture
def sandbox(tmp_path, monkeypatch, mini_fixtures) -> SimpleNamespace:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    shutil.copy(CORE_FAKES / "quietlib.py", scripts / "quietlib.py")
    install_exec(CORE_FAKES / "gpu_lock.sh", scripts / "gpu_lock.sh")
    omlx_repo = root / "omlx_repo"
    (omlx_repo / ".venv/bin").mkdir(parents=True)
    exe = install_exec(HERE / "mock_omlx.py", omlx_repo / ".venv/bin/omlx", sys.executable)
    ckpt = root / "models/oq/Mock-1B"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    models_dir = root / "models/hf-bench"
    models_dir.mkdir(parents=True)
    (models_dir / "Mock-1B").symlink_to(ckpt)
    port = free_port()
    results = root / "results"
    over = {
        "REPO": root, "OMLX_REPO": omlx_repo, "OMLX_EXE": exe, "OMLX_PY": Path(sys.executable),
        "BASE": omlx_repo / ".modeleval-home", "MODELS_DIR": models_dir, "PORT": port,
        "BASE_URL": f"http://127.0.0.1:{port}/v1", "RESULTS": results, "UNVERIFIED": results / "_unverified",
        "RUNS": root / "runs", "VENDOR": root / "vendor", "SCRIPTS": scripts, "GPU_LOCK": root / "gpu-lock",
        "HOLD_FILE": root / "runs/hold", "HF_HOME": root / "hf", "WS_ROOT": root / "ws",
    }
    for k, v in over.items():
        monkeypatch.setattr(paths, k, v)
    (root / "runs").mkdir()
    (root / "pins.toml").write_text('[epoch]\nid = "test-epoch"\n')
    for k, v in {"GPU_LOCK_PATH": root / "gpu-lock", "FAKE_LOCK_LOG": root / "lock.log",
                 "FAKE_QUIET_LOG": root / "quiet.log", "FAKE_QUIET_CHECKS": root / "quiet-checks",
                 "FAKE_QUIET_DISTURB": root / "disturb", "FAKE_QUIET_MINOR": root / "minor",
                 "MOCK_FIXTURES": mini_fixtures,
                 "MOCK_SPEED_TESTS": HERE.parent / "speed",
                 "BENCH_FAKE_RUNNER_LOG": root / "runner.log"}.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setattr(quiet, "MONITOR_INTERVAL", 0.2)
    monkeypatch.setattr(quiet, "MARGIN", 0.3)
    monkeypatch.setattr(quiet, "battery", lambda: (95, True))
    monkeypatch.setattr(run, "CAP_PYTHON", install_exec(HERE / "fake_runner.py", root / "cap-python", sys.executable))
    monkeypatch.setattr(run, "CAFFEINATE", False)
    monkeypatch.setattr(run, "SSD_CACHE_BYTES", 0)   # the host's free disk is not under test
    monkeypatch.setattr(run, "POLL_S", 0.01)
    monkeypatch.setattr(run, "ensure_epoch", lambda: "test-epoch")
    monkeypatch.setattr(run, "machine_info", lambda: {"sw_vers": {"BuildVersion": "test"}})
    monkeypatch.setattr(run, "sleep", lambda s: time.sleep(min(s, 0.01)))
    # the real speed module, on the mini fixtures, with the speed tests' fake renderer for G0
    from test_speed_e2e import fake_ids
    monkeypatch.setattr(speed, "FIXTURES_DIR", mini_fixtures)
    monkeypatch.setattr(speed, "independent_render",
                        lambda tok, msgs, tools, kw: {"ids": fake_ids(msgs, tools, kw), "transformers": "fake"})
    models = parse_models({"models": {"mock-1b": model_table(ckpt)}})
    return SimpleNamespace(root=root, port=port, models=models, model=models["mock-1b"], results=results,
                           runs=root / "runs", lock=root / "gpu-lock", disturb=root / "disturb",
                           minor=root / "minor")


@pytest.fixture(autouse=True)
def _no_real_lock():
    """No test may create or remove the shared GPU lock."""
    real = Path.home() / "Developer/mlsys/experiments/.gpu-lock"
    before = real.exists() and real.stat().st_mtime
    yield
    after = real.exists() and real.stat().st_mtime
    assert before == after, "a test touched the real GPU lock"
