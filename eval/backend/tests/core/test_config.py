"""config.defaults.toml + config.toml: where things are on a machine, and the optional GPU-lock protocol."""
from __future__ import annotations

from pathlib import Path

import pytest

from modeleval import paths
from modeleval.paths import ConfigError


def test_the_committed_defaults_name_every_path_and_resolve_to_the_owners_layout():
    cfg = paths.load_config((paths.REPO / "config.defaults.toml",))
    assert set(cfg["paths"]) >= {"mlsys", "omlx_repo", "omlx_base", "models", "models_dir", "ws_root", "agent_root",
                                 "claude_versions", "datasets", "hf_home", "gpu_lock"}
    assert cfg["lock"]["mode"] in paths.LOCK_MODES and cfg["machine"]["id"] == ""
    assert "/Users/" not in (paths.REPO / "config.defaults.toml").read_text()
    assert "config.toml" in (paths.REPO / ".gitignore").read_text().splitlines()


def test_a_local_file_overrides_key_by_key_and_references_resolve(tmp_path, monkeypatch):
    (tmp_path / "defaults.toml").write_text('[paths]\nroot = "~/a"\nrepo = "${root}/serving"\nrel = "x/y"\n'
                                            '[lock]\nmode = "auto"\n')
    (tmp_path / "local.toml").write_text('[paths]\nroot = "/opt/m"\n')
    cfg = paths.load_config((tmp_path / "defaults.toml", tmp_path / "local.toml", tmp_path / "absent.toml"))
    assert cfg["paths"] == {"root": "/opt/m", "repo": "${root}/serving", "rel": "x/y"} and cfg["lock"]["mode"] == "auto"
    monkeypatch.setattr(paths, "CONFIG", cfg)
    assert paths.expand("${repo}/omlx") == Path("/opt/m/serving/omlx")
    assert paths.expand("${rel}") == paths.REPO / "x/y" and paths.expand("${rel}", tmp_path) == tmp_path / "x/y"
    assert paths.expand("~/z") == Path.home() / "z"
    with pytest.raises(ConfigError, match=r"no \[paths\] nope"):
        paths.expand("${nope}/x")


def test_lock_modes(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_LOCK_PATH", raising=False)
    monkeypatch.setattr(paths, "GPU_LOCK", tmp_path / "shared" / ".gpu-lock")
    for mode, exists, on in (("auto", False, False), ("auto", True, True), ("on", False, True), ("off", True, False)):
        if exists:
            (tmp_path / "shared").mkdir(exist_ok=True)
        monkeypatch.setattr(paths, "CONFIG", {"lock": {"mode": mode}})
        assert paths.lock_enabled() is on, (mode, exists)
        env = paths.lock_env()
        assert env["GPU_LOCK_PATH"] == str(tmp_path / "shared" / ".gpu-lock") and ("QUIET_LOCK" in env) is (not on)
    monkeypatch.setattr(paths, "CONFIG", {"lock": {"mode": "sometimes"}})
    with pytest.raises(ConfigError, match="mode must be one of"):
        paths.lock_enabled()


def test_machine_id_from_the_hardware_or_overridden(monkeypatch):
    paths.machine_id.cache_clear()
    monkeypatch.delenv("MODELEVAL_MACHINE", raising=False)
    try:
        derived = paths.machine_id()
        assert derived.count("-") == 2 and derived.endswith("gb") and "c-" in derived    # e.g. m5max-40c-128gb
        paths.machine_id.cache_clear()
        monkeypatch.setattr(paths, "CONFIG", {**paths.CONFIG, "machine": {"id": "mini-10c-16gb"}})
        assert paths.machine_id() == "mini-10c-16gb"
        paths.machine_id.cache_clear()
        monkeypatch.setenv("MODELEVAL_MACHINE", "a/b")
        with pytest.raises(ConfigError, match="one directory name"):
            paths.machine_id()
    finally:
        paths.machine_id.cache_clear()
