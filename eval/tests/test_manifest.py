"""bench.manifest: which model server a run records (no oMLX; lsof is faked)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench import manifest


def _omlx_home(root: Path, alias: str) -> Path:
    home = root / "omlx-home"
    home.mkdir(parents=True)
    (home / "settings.json").write_text(json.dumps({"server": {"port": 8500}, "auth": {"api_key": "secret"}}))
    models = {"models": {f"mlx-community--{alias}": {"model_alias": alias, "max_context_window": 131072}}}
    (home / "model_settings.json").write_text(json.dumps(models))
    return home


@pytest.fixture
def served_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(manifest, "SERVED_DIR", runs)
    monkeypatch.setattr(manifest, "OMLX_BASE_URL", "http://127.0.0.1:8500/v1")
    monkeypatch.setattr(manifest, "BENCH_MODEL", "Qwen3.8-27B-8bit")
    return runs


def _fake_lsof(monkeypatch: pytest.MonkeyPatch, listener: int) -> None:
    real = manifest._run

    def run(cmd: list[str], timeout: float = 30.0) -> str | None:
        return str(listener) if cmd[0] == "lsof" else real(cmd, timeout)

    monkeypatch.setattr(manifest, "_run", run)


def test_the_server_serve_omlx_started_wins_over_the_configured_one(
    tmp_path: Path, served_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _omlx_home(tmp_path / "live", "Qwen3.8-27B-8bit")
    monkeypatch.setattr(manifest, "OMLX_HOME", tmp_path / "configured-home")
    record = {"server": "omlx", "repo": str(tmp_path / "live-repo"), "base_path": str(home), "pid": os.getpid()}
    (served_dir / ".server-8500.json").write_text(json.dumps(record))
    _fake_lsof(monkeypatch, os.getpid())

    server = manifest._model_server()

    assert server["verified"] is True and server["pid"] == os.getpid()
    assert server["base_path"] == str(home)
    assert server["repo"]["path"] == str(tmp_path / "live-repo")
    assert server["model_id"] == "mlx-community--Qwen3.8-27B-8bit"
    assert server["model_settings"]["max_context_window"] == 131072
    assert "secret" not in json.dumps(server)


def test_a_stale_serve_record_falls_back_to_the_configured_server(
    tmp_path: Path, served_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _omlx_home(tmp_path / "configured", "Qwen3.8-27B-8bit")
    monkeypatch.setattr(manifest, "OMLX_HOME", configured)
    record = {"server": "omlx", "repo": "/elsewhere", "base_path": "/elsewhere", "pid": 1}
    (served_dir / ".server-8500.json").write_text(json.dumps(record))
    _fake_lsof(monkeypatch, os.getpid())  # another process listens now

    server = manifest._model_server()

    assert server["verified"] is False and server["pid"] is None
    assert server["base_path"] == str(configured)
    assert server["model_id"] == "mlx-community--Qwen3.8-27B-8bit"


def test_the_model_record_has_defaults_without_direnv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manifest, "HF_HUB", tmp_path / "hub")

    model = manifest._model()

    assert model["id"] == manifest.BENCH_MODEL and model["path"] == manifest.BENCH_MODEL_PATH
    assert model["id"] and model["path"] and model["hf_repo"] is None


def test_a_converted_model_records_its_conversion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "chat" / "Qwen3.8-27B-oQ8-mtp"
    model_dir.mkdir(parents=True)
    record = {"name": model_dir.name, "source": {"hf_repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2"}}
    (model_dir / manifest.CONVERSION_RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(manifest, "BENCH_MODEL_HF_REPO", "")
    monkeypatch.setattr(manifest, "BENCH_MODEL_PATH", str(model_dir))

    model = manifest._model()

    assert model["path"] == str(model_dir) and model["conversion"] == record

    monkeypatch.setattr(manifest, "BENCH_MODEL_PATH", str(tmp_path / "chat" / "missing"))
    assert manifest._model()["conversion"] is None


def test_an_hf_model_records_its_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entry = tmp_path / "hub" / "models--mlx-community--Qwen3.8-27B-8bit"
    (entry / "refs").mkdir(parents=True)
    (entry / "refs" / "main").write_text("815b83c0\n")
    monkeypatch.setattr(manifest, "HF_HUB", tmp_path / "hub")
    monkeypatch.setattr(manifest, "BENCH_MODEL_HF_REPO", "mlx-community/Qwen3.8-27B-8bit")

    model = manifest._model()

    assert model["hf_repo"] == "mlx-community/Qwen3.8-27B-8bit" and model["revision"] == "815b83c0"
    assert model["snapshot_path"] == str(entry / "snapshots" / "815b83c0") and "conversion" not in model


def test_a_model_in_a_model_dir_is_found_by_its_directory_name(
    tmp_path: Path, served_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "omlx-home"
    home.mkdir()
    entry = {"max_context_window": 131072, "mtp_enabled": True}
    (home / "model_settings.json").write_text(json.dumps({"models": {"Qwen3.8-27B-oQ8-mtp": entry}}))
    monkeypatch.setattr(manifest, "OMLX_HOME", home)
    monkeypatch.setattr(manifest, "BENCH_MODEL", "Qwen3.8-27B-oQ8-mtp")

    server = manifest._model_server()

    assert server["model_id"] == "Qwen3.8-27B-oQ8-mtp" and server["model_settings"] == entry


def test_default_roots_follow_the_mlsys_layout():
    """<mlsys>/experiments/<pipeline>/eval: the env-free defaults (oMLX repo, models, agents) hang off <mlsys>."""
    if manifest.BENCH_ROOT.parent.parent.name != "experiments":
        pytest.skip("this checkout is not under <mlsys>/experiments")
    assert manifest.MLSYS_ROOT / "experiments" == manifest.BENCH_ROOT.parent.parent
    if "BENCH_SERVED_DIR" not in os.environ:
        assert manifest.SERVED_DIR == manifest.MLSYS_ROOT / "experiments" / "personal-agent-bench" / "runs"
