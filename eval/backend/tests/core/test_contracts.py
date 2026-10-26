"""The orchestrator against copies of the real scripts it drives (gpu_lock.sh + lib.sh), in the sandbox."""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys

import pytest

from modeleval import paths, run
from modeleval.server import Pidfile, group_alive


def test_lock_calls_against_the_real_gpu_lock_script(env):
    for name in ("gpu_lock.sh", "lib.sh"):
        shutil.copy(paths.REPO / "scripts" / name, env.scripts / name)
    r = run.Run(run.Options(model="fake-1b", notice_wait=0), env.models)
    try:
        r.acquire()
    except run.Stop as e:   # the agent-benchmark server is up on 8500 right now: acquire exits 3
        assert "8500" in e.reason
        assert not env.lock.exists()
        pytest.skip("port 8500 is up; acquire correctly refused and released")
    text = env.lock.read_text()
    assert f"run: {r.run_id}" in text and f"purpose: modeleval fake-1b speed (timed)" in text
    assert "holder: local_ai-benchmark session" in text
    r.quiet_window_end()
    assert "purpose: modeleval fake-1b capability, accuracy only (untimed)" in env.lock.read_text()
    assert f"run: {r.run_id}" in env.lock.read_text()
    other = run.Run(run.Options(model="fake-1b", notice_wait=0), env.models)
    other.run_id = "someone-else"
    other.lock_held = True
    other.release()                        # a different run id never drops our lock
    assert env.lock.exists()
    r.release()
    if env.lock.exists():                  # release refuses while something listens on 8703
        pytest.skip("port 8703 is listening on this machine; release correctly refused")
    assert not r.lock_held


def test_pidfile_recognises_a_retitled_omlx_by_its_port(env):
    """oMLX retitles itself, so a stale entry is also ours when its group listens on our port."""
    code = ("import socket, time; s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
            f"s.bind(('127.0.0.1', {env.port})); s.listen(); time.sleep(60)")
    p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        for _ in range(50):
            with socket.socket() as c:
                if c.connect_ex(("127.0.0.1", env.port)) == 0:
                    break
            import time
            time.sleep(0.1)
        pf = Pidfile(env.root / "pidfile")
        pf.add(p.pid, "omlx", "/no/such/omlx")
        out = pf.kill_all(timeout=3)
        assert any("stopped" in x for x in out), out
        assert p.wait(timeout=5) is not None and not group_alive(p.pid)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
