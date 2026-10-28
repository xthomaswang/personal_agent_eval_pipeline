from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time

import httpx
import pytest

from modeleval import paths
from modeleval.server import (MemorySampler, Pidfile, Server, ServerError, group_alive, model_settings,
                              parse_server_log, parse_vm_stat, port_listening)


def _server(env, **kw) -> Server:
    return Server(env.model, env.runs / "fake-1b/r1", 1, caffeinate=False, **kw)


def test_write_settings_sets_every_protocol_value(env):
    s = _server(env)
    s.write_settings()
    st = json.loads((paths.BASE / "settings.json").read_text())
    assert oct((paths.BASE / "settings.json").stat().st_mode & 0o777) == "0o600"
    assert st["server"]["port"] == env.port and st["server"]["host"] == "127.0.0.1"
    assert st["server"]["server_aliases"] == ["localhost", "127.0.0.1"]
    assert st["scheduler"] == {"max_concurrent_requests": 8, "embedding_batch_size": 32, "chunked_prefill": True,
                               "prefill_priority": "context", "decode_fairness": True}
    assert st["cache"]["hot_cache_max_size"] == "12GB" and st["cache"]["ssd_cache_max_size"] == "128GB"
    assert st["cache"]["ssd_cache_dir"] == str(s.ssd_cache) and s.ssd_cache.is_dir()
    assert st["memory"]["memory_guard_tier"] == "custom"
    assert st["memory"]["memory_guard_custom_ceiling_gb"] == 90.0
    assert (st["memory"]["soft_threshold"], st["memory"]["hard_threshold"]) == (0.85, 0.95)
    assert st["server"]["burst_decode_mode"] == "balanced"
    assert st["sampling"]["max_tokens"] == 32768 and st["sampling"]["top_k"] == 0
    assert (st["sampling"]["temperature"], st["sampling"]["top_p"]) == (1.0, 0.95)
    assert st["model"]["model_dirs"] == [str(paths.BASE / "empty-models")]
    assert (paths.BASE / "empty-models").is_dir() and not any((paths.BASE / "empty-models").iterdir())
    assert st["huggingface"]["hf_cache_enabled"] is False
    secret = st["auth"]["secret_key"]
    assert len(secret) == 64
    ms = json.loads((paths.BASE / "model_settings.json").read_text())
    assert ms == {"version": 1, "models": {"Fake-1B-mtp": {"max_context_window": 131072, "mtp_enabled": True}}}
    # the manifest copy never carries auth or aliases
    assert "auth" not in s.written_settings and "server_aliases" not in s.written_settings["server"]
    s.write_settings()   # the secret survives a rewrite, so oMLX never regenerates and re-saves the file
    assert json.loads((paths.BASE / "settings.json").read_text())["auth"]["secret_key"] == secret


def test_protocol_v1_settings_remain_reproducible(tmp_path):
    from modeleval.server import protocol_settings

    v1 = protocol_settings(tmp_path, 8703, tmp_path / "ssd", version=1)
    v2 = protocol_settings(tmp_path, 8703, tmp_path / "ssd", version=2)
    assert v1["cache"]["hot_cache_max_size"] == "20GB"
    assert v1["memory"]["memory_guard_tier"] == "balanced"
    assert v1["memory"]["memory_guard_custom_ceiling_gb"] == 0.0
    assert v2["cache"]["hot_cache_max_size"] == "12GB"
    assert v2["memory"]["memory_guard_tier"] == "custom"
    assert v2["memory"]["memory_guard_custom_ceiling_gb"] == 90.0


def test_model_settings_per_spec_and_no_max_tokens(env):
    from modeleval.registry import parse_models
    from .conftest import model_table
    ms = parse_models({"models": {
        "a": model_table(env.ckpt),
        "b": model_table(env.ckpt, served_id="NoSpec", spec="none", reference=False, family="nospec",
                         model_settings={"qwen4_ple_ssd_offload": True})}})
    out = model_settings(list(ms.values()))["models"]
    assert out["NoSpec"] == {"max_context_window": 131072, "qwen4_ple_ssd_offload": True}
    assert "mtp_enabled" not in out["NoSpec"]


def test_write_settings_refused_while_port_listens(env):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", env.port))
        srv.listen()
        with pytest.raises(ServerError, match="listening"):
            _server(env).write_settings()
    assert not (paths.BASE / "settings.json").exists()


def test_start_warm_up_log_checks_and_stop(env):
    s = _server(env)
    s.start()
    try:
        assert s.alive() and port_listening(env.port)
        assert s.base_url == f"http://127.0.0.1:{env.port}/v1"
        assert httpx.get(f"{s.base_url}/models").json()["data"][0]["id"] == "Fake-1B-mtp"
        reqs = (env.root / "omlx-requests").read_text().splitlines()
        assert "chat Fake-1B-mtp" in reqs   # the 8-token warm-up loaded the model
        eff = s.effective_settings
        assert eff["version"] == "0.7.1rc1" and eff["memory_guard"] == "on (tier: custom)"
        assert eff["ssd_cache"]["max"] == "128.0GB" and eff["hot_cache"] == "12.0GB"
        assert eff["ssd_cache_manager"]["auto_size"] == "False"
        assert eff["loaded_model"]["id"] == "Fake-1B-mtp"
        assert set(eff["log_expect"]) == set(env.model.server_log_expect)
        assert [e["role"] for e in Pidfile(s.run_dir / "pidfile").entries()] == ["omlx"]
    finally:
        s.stop()
    assert not port_listening(env.port) and not group_alive(s.pid)
    assert Pidfile(s.run_dir / "pidfile").entries() == []
    rec = s.record()
    assert rec["launch"] == 1 and rec["started"] and rec["stopped"]


def test_missing_log_expectation_refuses_and_stops(env, monkeypatch):
    monkeypatch.setenv("FAKE_OMLX_LINES", "nothing about speculative decoding")
    s = _server(env)
    with pytest.raises(ServerError, match="no server-log line matches"):
        s.start()
    assert not port_listening(env.port) and not group_alive(s.pid)


def test_refuses_to_start_while_port_listens(env):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", env.port))
        srv.listen()
        with pytest.raises(ServerError, match="listening"):
            _server(env).start()


def test_early_exit_is_reported(env, monkeypatch):
    monkeypatch.setenv("FAKE_OMLX_MODE", "die")
    with pytest.raises(ServerError, match="exited with code 3"):
        _server(env).start()


def test_stop_escalates_to_sigkill_and_kills_the_whole_group(env, monkeypatch):
    monkeypatch.setenv("FAKE_OMLX_MODE", "ignore-sigint")
    monkeypatch.setenv("FAKE_OMLX_CHILD", "1")
    s = _server(env)
    s.STOP_TIMEOUT = 1.0
    s.start()
    t0 = time.time()
    s.stop()
    assert time.time() - t0 < 15
    assert not group_alive(s.pid) and not port_listening(env.port)   # the sleep child went with the group


def test_second_launch_appends_to_the_log_and_parses_only_its_own_part(env):
    s1 = _server(env)
    s1.start()
    s1.stop()
    s2 = Server(env.model, env.runs / "fake-1b/r1", 2, caffeinate=False)
    s2.start()
    s2.stop()
    text = (env.runs / "fake-1b/r1/server.log").read_text()
    assert "modeleval launch 1" in text and "modeleval launch 2" in text
    assert "modeleval launch 1" not in s2.log_text()


def test_pidfile_kill_all_spares_reused_pids(tmp_path):
    pf = Pidfile(tmp_path / "pidfile")
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    pf.add(p.pid, "monitor", "sleep")
    pf.add(p.pid + 100000, "gone", "x")          # not a live group
    q = subprocess.Popen(["sleep", "61"], start_new_session=True)
    pf.add(q.pid, "omlx", "/nonexistent/omlx")   # alive but not what we started: left alone
    out = pf.kill_all(timeout=2)
    assert any("monitor" in x and "stopped" in x for x in out)
    assert p.wait(timeout=5) is not None
    assert q.poll() is None
    q.kill()
    q.wait()
    assert pf.entries() == []


def test_parse_server_log_paths_are_tilded():
    home = str(paths.HOME)
    text = (f"\x1b[33m└─ Version: 0.7.1rc1\x1b[0m\nBase path: {home}/x\n"
            f"paged SSD cache: {home}/c (max: 128.0GB)\n"
            "2026 - omlx.patches.x - INFO - Qwen4-Exp PLE mode for /m: mmap\n"
            "2026 - omlx.engine_pool - WARNING - Qwen4-Exp PLE forced to SSD for M: resident\n")
    eff = parse_server_log(text)
    assert eff["base_path"] == "~/x" and eff["ssd_cache"] == {"dir": "~/c", "max": "128.0GB"}
    assert eff["ple_mode"] == "mmap" and eff["ple_forced_to_ssd"] is True


def test_parse_vm_stat():
    text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages wired down:  10.\n"
            "File-backed pages:  20.\nPages occupied by compressor:  3.\nSwapouts:  7.\n")
    assert parse_vm_stat(text) == {"wired": 10 * 16384, "compressor": 3 * 16384, "file_backed": 20 * 16384,
                                   "swapouts": 7}


def test_memory_sampler_writes_rows_with_labels(tmp_path):
    p = subprocess.Popen(["sleep", "30"])
    try:
        ms = MemorySampler(tmp_path, interval=0.1, launch_no=3)
        ms.start(p.pid)
        ms.mark("after-load")
        time.sleep(0.5)
        ms.mark("m3-110k")
        time.sleep(0.4)
        summary = ms.stop()
    finally:
        p.kill()
        p.wait()
    lines = (tmp_path / "speed/memory.tsv").read_text().splitlines()
    assert lines[0].split("\t")[:4] == ["iso", "epoch", "launch", "label"]
    assert len(lines) >= 3
    assert summary["launch"] == 3 and summary["samples"] == len(lines) - 1
    assert summary["after_load_tree_bytes"] > 0 and summary["peak"]["tree_bytes"] > 0
    assert "m3-110k" in summary["by_label"] and summary["source"] in ("footprint", "rss")
