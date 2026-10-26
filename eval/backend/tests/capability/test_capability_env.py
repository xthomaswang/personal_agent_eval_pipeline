"""The runner's environment allowlist and the workspace-root preflight (DESIGN §6.2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from modeleval.capability import runner


def test_allowed_env_keeps_the_allowlist_and_drops_credentials() -> None:
    environ = {
        "PATH": "/usr/bin", "HOME": "/home/x", "USER": "x", "LOGNAME": "x", "SHELL": "/bin/zsh", "TMPDIR": "/tmp/x",
        "TERM": "xterm", "LANG": "en_US.UTF-8", "LC_ALL": "C", "HF_HOME": "/hf", "NVM_DIR": "/nvm",
        "BENCH_MODEL": "m", "OMLX_BASE_URL": "u", "PAB_ROOT": "/p", "HARNESS_SMOKE_DATA": "/d",
        # dropped: not on the allowlist, or credential-shaped
        "DEVELOPER_DIR": "/Applications/Xcode-beta.app", "SSH_AUTH_SOCK": "/s", "CLAUDECODE": "1",
        "CLAUDE_CODE_SESSION": "s", "ANTHROPIC_API_KEY": "k", "OPENAI_BASE_URL": "u", "GH_TOKEN": "t",
        "BENCH_UPSTREAM_TOKEN_CMD": "rec auth token", "AWS_SECRET_KEY": "k", "github_token": "t",
    }  # fmt: skip
    env = runner.allowed_env(environ, {"BENCH_EXTRA": "1"})
    assert set(env) == {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TERM", "LANG", "LC_ALL", "HF_HOME", "NVM_DIR",
        "BENCH_MODEL", "OMLX_BASE_URL", "PAB_ROOT", "HARNESS_SMOKE_DATA", "BENCH_EXTRA",
        "PYTHONDONTWRITEBYTECODE", "GIT_OPTIONAL_LOCKS",
    }  # fmt: skip
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["GIT_OPTIONAL_LOCKS"] == "0"
    assert runner.forbidden_names(env) == []
    assert runner.forbidden_names(environ) == sorted(
        ["CLAUDECODE", "CLAUDE_CODE_SESSION", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "GH_TOKEN",
         "BENCH_UPSTREAM_TOKEN_CMD", "AWS_SECRET_KEY", "github_token"]
    )  # fmt: skip


@pytest.mark.parametrize("name", ["CLAUDE_CONFIG_DIR", "OPENAI_API_KEY", "HF_TOKEN", "SOME_KEY"])
def test_allowed_env_refuses_credential_extras(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        runner.allowed_env({"PATH": "/usr/bin"}, {name: "x"})


def test_forced_values_win_over_the_callers() -> None:
    env = runner.allowed_env({"PYTHONDONTWRITEBYTECODE": "", "GIT_OPTIONAL_LOCKS": "1"})
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["GIT_OPTIONAL_LOCKS"] == "0"


def test_ws_root_preflight(tmp_path: Path) -> None:
    clean = tmp_path / "a" / "b" / "c" / "ws"
    assert runner.ws_root_problems(clean) == []
    assert clean.is_dir()

    (tmp_path / "a" / "b" / "Makefile").write_text("all:\n")
    (tmp_path / "a" / "b" / "c" / "AGENTS.md").write_text("x\n")
    problems = runner.ws_root_problems(clean)
    assert len(problems) == 2 and "Makefile" in " ".join(problems) and "AGENTS.md" in " ".join(problems)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    inside = repo / "x" / "y" / "z" / "ws"
    assert any("git repository" in p for p in runner.ws_root_problems(inside))


def test_the_production_ws_root_passes() -> None:
    from modeleval import paths

    # The runner creates ws_root; until then check from its parent, which sees the same ancestors
    ws_root = paths.WS_ROOT if paths.WS_ROOT.exists() else paths.WS_ROOT.parent
    assert runner.ws_root_problems(ws_root) == []


def test_boundary_rule_is_the_users_battery_rule() -> None:
    rule = runner.boundary_rule
    assert [rule("task", p) for p in (100, 50, 49, 20, 19)] == ["go", "go", "pause", "pause", "stop"]
    assert [rule("harness-block", p) for p in (60, 59, 10)] == ["go", "pause", "pause"]
    assert rule("task", None) == "go"


def test_localmelo_args_stop_a_minute_before_the_kill() -> None:
    assert runner.localmelo_args(3600) == ["--request-timeout", "1800", "--timeout", "3540", "--output-format", "json"]
    assert runner.localmelo_args(120.0)[3] == "60"
