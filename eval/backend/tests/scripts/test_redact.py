"""redact.py: the --check rules, the app-name rewrite, and the scripts' own files passing the check."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/redact.py"
SYSTEM_PY = Path("/usr/bin/python3")
# Sensitive-looking strings are assembled at run time, so this file passes the check itself.
USERS_DIR = "/Us" + "ers/"
UUID = "-".join(["1b4e28ba", "2fa1", "11d2", "883f", "0016d3cca427"])
APP, APP2 = "We" + "Chat", "Google " + "Drive"    # two of redact.APPS


def ip(*octets: int) -> str:
    return ".".join(map(str, octets))


@pytest.fixture
def hosts(redact):
    return redact.host_pattern(["tm9x.local", "tm9x"])


def problems(redact, hosts, text: str) -> list[str]:
    return [what for _, what in redact.scan_text(text, hosts)]


@pytest.mark.parametrize("text, bad", [
    (f"log at {USERS_DIR}alice/Developer/x.log", True),
    (f"cwd={USERS_DIR}Shared/x", True),
    (f"the check fails on `{USERS_DIR}`, the host name", False),   # a bare mention, as in DESIGN.md
    ("~/Developer/mlsys/models", False),
    ("/usr/bin/python3", False),
])
def test_home_paths(redact, hosts, text, bad):
    assert (problems(redact, hosts, text) == ["home path (write ~ or a relative path)"]) is bad


@pytest.mark.parametrize("text, bad", [
    ("served by TM9X.local:8703", True), ("hostname tm9x", True), ("(tm9x)", True),
    ("tm9xa", False), ("atm9x", False), ("tm9x-other", False), ("localhost", False),
])
def test_host_names(redact, hosts, text, bad):
    assert ("host name" in problems(redact, hosts, text)) is bad


def test_hostnames_of_this_machine(redact):
    names = redact.hostnames()
    assert "localhost" not in [n.lower() for n in names]
    full = socket.gethostname()
    if len(full) >= 3 and not full.lower().startswith("localhost"):
        assert full in names


@pytest.mark.parametrize("text, found", [
    (f"server at {ip(10, 0, 0, 7)}:8703", [ip(10, 0, 0, 7)]),
    (f"peer {ip(172, 16, 4, 1)} and {ip(172, 31, 255, 254)}.", [ip(172, 16, 4, 1), ip(172, 31, 255, 254)]),
    (f"lan {ip(192, 168, 1, 20)}", [ip(192, 168, 1, 20)]),
    (f"loopback {ip(127, 0, 0, 1)}", []),
    (f"public {ip(172, 32, 0, 1)} {ip(8, 8, 8, 8)} {ip(192, 169, 1, 1)}", []),
    (f"not an address {ip(10, 0, 0, 300)} or {ip(10, 1, 2, 3)}.4 or 1.{ip(10, 1, 2, 3)}", []),
    ("version 1.10.0.1", []),
])
def test_private_ipv4(redact, text, found):
    assert redact.private_ipv4(text) == found


@pytest.mark.parametrize("text, bad", [
    ('{"session_id": "' + UUID + '"}', True),
    ('"sessionId":"' + UUID.upper() + '"', True),
    ("machineID=" + UUID, True),
    ("CLAUDE_SESSION_ID=" + UUID, True),
    ("user-id: " + UUID, True),
    ('{\\"device_id\\": \\"' + UUID + '\\"}', True),         # JSON inside a JSON string
    ('"userID": "' + "ab" * 32 + '"', True),                  # a 64-digit hex user id
    ('"run_id": "' + UUID + '"', False),                      # other ids pass
    ('"session_id": null', False),
    ("trace " + UUID, False),
    ('"session_id": "20261026-0300"', False),
])
def test_keyed_ids(redact, hosts, text, bad):
    assert ("machine/user/session/device id" in problems(redact, hosts, text)) is bad


def test_app_names_are_reported_by_check(redact, hosts):
    text = f"busy: {APP}(123) 40%\nok line\ntop: kernel 2.0, {APP2} 1.5\n"
    assert redact.scan_text(text, hosts) == [(1, "third-party app name (run redact.py without --check)"),
                                             (3, "third-party app name (run redact.py without --check)")]


@pytest.mark.parametrize("path, ok", [
    ("results/qwen/20261026-0300/capability/results.json", True),
    ("results/qwen/20261026-0300/capability/claude_code/transpose/calls.jsonl", True),
    ("results/qwen/20261026-0300/capability/hermes/wordy/summary.json", True),
    ("results/qwen/20261026-0300/capability/localmelo/react/check.txt", True),
    ("results/qwen/20261026-0300/capability/localmelo/react/row.json", False),
    ("results/qwen/20261026-0300/capability/claude-code/transpose/calls.jsonl", False),
    ("results/qwen/20261026-0300/capability/claude_code/transpose/prompt.md", False),
    ("results/qwen/20261026-0300/capability/claude_code/transpose/workspace/x.py", False),
    ("results/qwen/20261026-0300/capability/claude_code/calls.jsonl", False),
    ("results/qwen/20261026-0300/capability/manifest.json", False),
    ("results/qwen/20261026-0300/capability/raw/stdout.log", False),
    ("/abs/agent-model-eval/results/qwen/r1/capability/hermes/y/check.txt", True),
    ("/abs/agent-model-eval/results/qwen/r1/capability/x/y/check.txt", False),
    ("/abs/agent-model-eval/results/qwen/r1/capability/hermes/y/stdout", False),
    ("results/qwen/20261026-0300/speed/monitor.tsv", True),
    ("results/qwen/20261026-0300/manifest.json", True),
    ("docs/capability/notes.md", True),
    # the results contract's layout, relative to the repository root (what the pre-commit hook passes)
    ("docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952/capability/results.json", True),
    ("docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952/capability/hermes/wordy/summary.json", True),
    ("docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952/capability/claude_code/transpose/prompt.md", False),
    ("docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952/capability/manifest.json", False),
    ("docs/benchmark/m5max/qwen3.8_27b/oq5/backend/r1/capability/hermes/y/stdout", False),
    ("docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952/speed/monitor.tsv", True),
    ("docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952/README.md", True),
    ("docs/benchmark/qwen3.8_27b/oq5/agent_ability/20261026-062952/summary.json", True),
])
def test_capability_allowlist(redact, path, ok):
    assert (redact.capability_problem(path) is None) is ok


def test_check_file_kinds(redact, hosts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("good.txt").write_text("fine\n", encoding="utf-8")
    Path("latin1.txt").write_bytes(b"caf\xe9\n")
    os.symlink("good.txt", "link.txt")
    os.symlink("nowhere.txt", "broken.txt")
    Path("dir").mkdir()
    assert redact.check_file("good.txt", hosts) == []
    assert redact.check_file("latin1.txt", hosts) == ["latin1.txt: not UTF-8 (byte 3)"]
    assert redact.check_file("link.txt", hosts) == ["link.txt: symlink (commit the file itself, or nothing)"]
    assert redact.check_file("broken.txt", hosts) == ["broken.txt: symlink (commit the file itself, or nothing)"]
    assert redact.check_file("dir", hosts) == ["dir: not a regular file"]
    assert redact.check_file("missing.txt", hosts) == ["missing.txt: missing"]
    cap = Path("results/m/r/capability/h/t")
    cap.mkdir(parents=True)
    (cap / "prompt.md").write_text(f"see {USERS_DIR}bob/x\n", encoding="utf-8")
    out = redact.check_file(str(cap / "prompt.md"), hosts)
    assert out[0].endswith("not on the capability export allowlist (results.json; <harness>/<run>/calls.jsonl, "
                           "summary.json, check.txt; harness claude_code, hermes, localmelo)")
    assert out[1] == "results/m/r/capability/h/t/prompt.md:1: home path (write ~ or a relative path)"


def test_main_check_and_rewrite(redact, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(redact, "hostnames", lambda: ["tm9x"])
    Path("m.tsv").write_text(f"a\tbusy: {APP}(12) 40%\r\nb\tquiet\r\n", encoding="utf-8")
    Path("ip.txt").write_text(f"x\nhost tm9x at {ip(10, 2, 3, 4)}\n", encoding="utf-8")
    assert redact.main(["--check", "m.tsv", "ip.txt"]) == 1
    out = capsys.readouterr().out
    assert "redact: m.tsv:1: third-party app name" in out
    assert "redact: ip.txt:2: host name" in out and f"redact: ip.txt:2: private IPv4 address {ip(10, 2, 3, 4)}" in out
    assert redact.main(["m.tsv"]) == 0                    # rewrite mode: names only, line endings kept
    assert Path("m.tsv").read_bytes() == b"a\tbusy: <app>(12) 40%\r\nb\tquiet\r\n"
    assert redact.main(["--check", "m.tsv"]) == 0
    os.symlink("m.tsv", "l.tsv")
    assert redact.main(["l.tsv"]) == 1 and "skipped" in capsys.readouterr().out
    assert redact.main(["--bogus", "m.tsv"]) == 2
    assert redact.main(["--check"]) == 0


@pytest.mark.parametrize("python", [sys.executable, str(SYSTEM_PY)])
def test_cli_flags_this_machines_host_name(tmp_path, python):
    if not Path(python).exists():
        pytest.skip(f"no {python}")
    full = socket.gethostname()
    if len(full) < 3 or full.lower().startswith("localhost"):
        pytest.skip("generic host name")
    f = tmp_path / "manifest.json"
    f.write_text('{"host": "' + full + '"}\n', encoding="utf-8")
    r = subprocess.run([python, "-I", "-B", str(SCRIPT), "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 1 and f"{f}:1: host name" in r.stdout
    f.write_text('{"host": "<machine>"}\n', encoding="utf-8")
    r = subprocess.run([python, "-I", "-B", str(SCRIPT), "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_owned_files_pass_their_own_check():
    """The hook checks every staged file, these included: they must not trip it themselves."""
    owned = [REPO / "scripts" / n for n in ("quietlib.py", "quiet_check.sh", "gpu_lock.sh", "lib.sh", "redact.py",
                                            "pre-commit")]
    owned += sorted((REPO / "tests/scripts").glob("*.py"))
    py = str(SYSTEM_PY) if SYSTEM_PY.exists() else sys.executable
    r = subprocess.run([py, "-I", "-B", str(SCRIPT), "--check", *map(str, owned)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


GH = "gh" + "p_" + "A1b2C3d4" * 5          # assembled at run time (see USERS_DIR)
HF = "hf" + "_" + "Ab12" * 8


@pytest.mark.parametrize("text, bad", [
    (f"git clone https:/{'/'}{GH}@github.com/org/private", True),
    (f"git clone https:/{'/'}user:secret@example.invalid/x.git", True),
    (f"export TOKEN={GH}", True),
    (f"hf download org/model --{'token'} {HF}", True),
    (f"key {'sk'}-proj-{'Ab12' * 6}", True),
    ("git clone https://github.com/org/repo.git", False),
    ("ssh git@github.com:org/repo", False),
    ("pip install scikit-learn; sk-learn is a nickname", False),
    ("uses hf_transfer and hf_cache_enabled", False),
    (f"--{'token'}-file ~/.config/x", False),
])
def test_credentials(redact, hosts, text, bad):
    assert any("credential" in p for p in problems(redact, hosts, text)) is bad
