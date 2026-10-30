"""Run one harness on one task in a workspace, with the recorder proxy in between.

Each run gets its own proxy instance on an OS-assigned port, its own copy of the harness home, and
a run directory holding manifest.json, the proxy's calls.jsonl and proxy.log, the harness's
stdout.log and stderr.log, and summary.json. The generic harness smoke test uses this now; the
T1-T4 runner will build on it later.

The caller is responsible for the shared GPU lock and for a running oMLX.

Usage:
    uv run python -m bench.harness --harness claude_code --workspace <dir> \
        --prompt-file <task.md> --run-dir runs/<id> [--timeout 900] [--max-steps 40]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from bench import manifest
from bench.proxy.steps import load_calls, step_counts

AGENT_ROOT = manifest.MLSYS_ROOT / "agent"
CLAUDE_ENV = AGENT_ROOT / "configs" / "claude-code" / "claude-code.env"
CLAUDE_HOME = AGENT_ROOT / "homes" / "claude-code"
HERMES_BIN = AGENT_ROOT / ".venv" / "bin" / "hermes"
HERMES_HOME = AGENT_ROOT / "homes" / "hermes"
MODEL = manifest.BENCH_MODEL
UPSTREAM = manifest.OMLX_BASE_URL.removesuffix("/v1")
UPSTREAM_TOKEN_CMD_ENV = "BENCH_UPSTREAM_TOKEN_CMD"

# Built-in tools each harness may use in the generic smoke test: no web, no MCP
CLAUDE_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
HERMES_TOOLSETS = "terminal,file"

# Hermes state that is large or shared is linked into the per-run copy; logs and caches are not copied
HERMES_SHARED = ("tools", "installs")
HERMES_SKIP = ("logs", "sessions", "cache", "audio_cache", "image_cache", "backups")

Command = tuple[list[str], dict[str, str]]


def _claude_code(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> Command:
    home = run_dir / "claude-home"
    shutil.copytree(CLAUDE_HOME, home)
    # claude-code.env exports the 8500 base URL and the shared config dir; override both after sourcing it
    script = (
        '. "$1"; shift; '
        'export ANTHROPIC_BASE_URL="$PAB_BASE_URL" CLAUDE_CONFIG_DIR="$PAB_CONFIG_DIR"; '
        'exec "$CLAUDE_BIN" "$@"'
    )
    args = [
        "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--max-turns", str(max_turns),
        "--tools", CLAUDE_TOOLS,
        "--allowedTools", CLAUDE_TOOLS,
        "--permission-mode", "dontAsk",
        "--permission-prompts", "none",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
        "--no-session-persistence",
        "--exclude-dynamic-system-prompt-sections",
    ]  # fmt: skip
    env = {"PAB_BASE_URL": base_url, "PAB_CONFIG_DIR": str(home)}
    return ["/bin/sh", "-c", script, "sh", str(CLAUDE_ENV), *args], env


def _hermes_config(text: str, base_url: str) -> str:
    """The template's config.yaml with its provider URL replaced by the proxy and its model by MODEL.

    The template names oMLX on 8500 and the oMLX model; a run may serve another engine (OMLX_BASE_URL,
    BENCH_MODEL). Replacing text keeps the template's comments.
    """
    config = yaml.safe_load(text) or {}
    model = config.get("model") or {}
    api = config["providers"][model.get("provider", "omlx")]["api"]
    text = text.replace(api.removesuffix("/v1"), base_url)
    if model.get("default"):
        text = text.replace(model["default"], MODEL)
    if f"{base_url}/v1" not in text:
        raise RuntimeError(f"Hermes config names no proxy URL after rewriting {api}")
    return text


def _hermes(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> Command:
    home = run_dir / "hermes-home"
    shutil.copytree(
        HERMES_HOME,
        home,
        symlinks=True,
        ignore=shutil.ignore_patterns(*HERMES_SHARED, *HERMES_SKIP),
    )
    for name in HERMES_SHARED:
        if (HERMES_HOME / name).exists():
            (home / name).symlink_to(HERMES_HOME / name)
    config = home / "config.yaml"
    config.write_text(_hermes_config(config.read_text(), base_url))
    # No --yolo: in one-shot mode Hermes denies commands it flags as dangerous, which is what we want
    cmd = [
        str(HERMES_BIN), "chat", "--query-file", str(run_dir / "prompt.md"), "--oneshot",
        "--format", "stream-json",
        "--max-turns", str(max_turns),
        "-t", HERMES_TOOLSETS,
    ]  # fmt: skip
    return cmd, {"HERMES_HOME": str(home), "HERMES_DISABLE_LAZY_INSTALLS": "1"}


def _accepts_no_save_session(repo: Path) -> bool:
    """Whether the localmelo checkout at repo knows --no-save-session, i.e. its __main__.py names the option.
    A checkout without a readable __main__.py is treated as one that does not know it."""
    try:
        return "--no-save-session" in (repo / "__main__.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _localmelo(prompt: str, run_dir: Path, base_url: str, max_turns: int) -> Command:
    # Direct mode has no turn limit of its own; its hard cap is 30 steps, plus the proxy's --max-steps.
    # Persisting memory per run gives history.db, whose steps table is localmelo's own step count.
    # "--" ends the options, so a task text that starts with "-" stays the task.
    # A blank task requests localmelo's interactive UI, which exits 2 without a TTY (since U2a.1);
    # refuse it here so it cannot pass as a run.
    if not prompt.strip():
        raise ValueError("localmelo needs a non-blank task text")
    # localmelo records each direct run as a worker session in the user's cache directory by default
    # (since the release that added --no-save-session), and the bench opts out whenever the checkout
    # knows the option.
    opt_out = ["--no-save-session"] if _accepts_no_save_session(manifest.LOCALMELO_REPO) else []
    cmd = [
        str(manifest.LOCALMELO_REPO / ".venv" / "bin" / "melo"),
        "--base-url", f"{base_url}/v1",
        "--chat-model", MODEL,
        *opt_out,
        "--", prompt,
    ]  # fmt: skip
    env = {"LOCALMELO_PERSIST_MEMORY": "1", "LOCALMELO_MEMORY_DIR": str(run_dir / "mem")}
    return cmd, env


BUILDERS: dict[str, Callable[[str, Path, str, int], Command]] = {
    "claude_code": _claude_code,
    "hermes": _hermes,
    "localmelo": _localmelo,
}

# Harnesses whose command line takes extra flags from run(harness_args=...): localmelo's options
# (--tool-exposure, --builtins, --request-timeout, ...) vary per experiment, the others' do not.
HARNESS_ARGS = ("localmelo",)


def _with_harness_args(cmd: list[str], prompt: str, harness_args: Sequence[str]) -> list[str]:
    """cmd with harness_args before its trailing prompt positional (and the "--" in front of it, if
    any), or at the end when it has none."""
    if cmd[-2:] == ["--", prompt]:
        return [*cmd[:-2], *harness_args, "--", prompt]
    if cmd and cmd[-1] == prompt:
        return [*cmd[:-1], *harness_args, prompt]
    return [*cmd, *harness_args]


def _start_proxy(
    run_dir: Path,
    harness: str,
    thinking: str,
    max_steps: int | None,
    random_seed: bool = False,
    one_reasoning_field: bool = False,
    log_bodies: bool = False,
) -> tuple[subprocess.Popen[str], str]:
    cmd = [
        sys.executable, "-m", "bench.proxy",
        "--port", "0",
        "--upstream", UPSTREAM,
        "--harness", harness,
        "--run-id", run_dir.name,
        "--run-dir", str(run_dir),
        "--thinking", thinking,
    ]  # fmt: skip
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    if random_seed:
        cmd.append("--random-seed")
    if one_reasoning_field:
        cmd.append("--one-reasoning-field")
    if log_bodies:
        cmd.append("--log-bodies")
    # An upstream that needs a bearer token (Recursive's rec engine): the proxy runs this command
    # itself, so the token reaches neither the harness's environment nor the run records.
    if os.environ.get(UPSTREAM_TOKEN_CMD_ENV):
        cmd += ["--upstream-token-cmd", os.environ[UPSTREAM_TOKEN_CMD_ENV]]
    log_path = run_dir / "proxy.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=manifest.BENCH_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        found = re.search(r"listening on (http://\S+)", log_path.read_text())
        if found:
            return proc, found.group(1)
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"proxy did not start; see {log_path}")


def _without(environ: Any, name: str) -> dict[str, str]:
    return {k: v for k, v in environ.items() if k != name}


def _stop(proc: subprocess.Popen[Any], group: bool = False) -> int:
    """SIGTERM, then SIGKILL after 10 s. Returns the exit code."""
    send = (lambda s: os.killpg(proc.pid, s)) if group else proc.send_signal
    if proc.poll() is None:
        try:
            send(signal.SIGTERM)
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
        except ProcessLookupError:
            pass
    return proc.wait()


def _events(stdout: Path) -> list[dict[str, Any]]:
    events = []
    for line in stdout.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _sqlite_value(db: Path, query: str) -> Any:
    """The first column of the first row, or None when the database, the table or the row is missing."""
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            row = conn.execute(query).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def _sqlite_scalar(db: Path, query: str) -> int | None:
    value = _sqlite_value(db, query)
    return int(value) if value is not None else None


def _harness_report(harness: str, run_dir: Path) -> dict[str, Any]:
    """What the harness itself says about the run: its own step count and final status."""
    events = _events(run_dir / "stdout.log")
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    if harness == "claude_code":
        return {
            "num_turns": result.get("num_turns"),
            "subtype": result.get("subtype"),
            "is_error": result.get("is_error"),
            "permission_denials": len(result.get("permission_denials") or []),
            "duration_ms": result.get("duration_ms"),
        }
    if harness == "hermes":
        db = run_dir / "hermes-home" / "state.db"
        return {
            "tool_use_events": sum(1 for e in events if e.get("type") == "tool_use"),
            "api_call_count": _sqlite_scalar(db, "SELECT SUM(api_call_count) FROM sessions"),
            "exit_code": result.get("exit_code"),
        }
    if harness == "localmelo":
        db = run_dir / "mem" / "history.db"
        # With --output-format json, stdout is one line: the run's RunResult record. The answer
        # (`result`) stays in stdout.log only.
        record = next((e for e in reversed(events) if "task_id" in e and "status" in e), None)
        return {
            # One row per tool call, so it pairs with the proxy's n_tool
            "steps": _sqlite_scalar(db, "SELECT COUNT(*) FROM steps"),
            # localmelo's own verdict: "failed" also when its reflection gate stopped the run (exit code 0),
            # "running" when it crashed
            "task_status": _sqlite_value(db, "SELECT status FROM tasks ORDER BY rowid DESC LIMIT 1"),
            "run_record": None if record is None else {k: v for k, v in record.items() if k != "result"},
        }
    return {"result_event": {k: result[k] for k in ("type", "num_turns") if k in result} or None}


def run(
    harness: str,
    prompt: str,
    workspace: Path,
    run_dir: Path,
    *,
    timeout_s: float = 900,
    max_steps: int | None = None,
    max_turns: int = 60,
    thinking: str = "off",
    random_seed: bool = False,
    one_reasoning_field: bool = False,
    log_bodies: bool = False,
    harness_args: Sequence[str] = (),
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """extra_env is merged into the harness's environment last, e.g. a PATH with a task's tools first.

    harness_args go on the harness's command line before the prompt (localmelo only); log_bodies is
    the proxy's --log-bodies, which proxy.log records with the proxy's other settings.
    """
    if harness_args and harness not in HARNESS_ARGS:
        raise ValueError(f"harness_args are for {', '.join(HARNESS_ARGS)} only, not {harness}")
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.collect(), indent=2, ensure_ascii=False) + "\n"
    )
    (run_dir / "prompt.md").write_text(prompt)
    proxy, base_url = _start_proxy(
        run_dir, harness, thinking, max_steps, random_seed, one_reasoning_field, log_bodies
    )
    try:
        cmd, builder_env = BUILDERS[harness](prompt, run_dir, base_url, max_turns)
        if harness_args:
            cmd = _with_harness_args(cmd, prompt, harness_args)
        started_at = time.time()
        t0 = time.monotonic()
        with (run_dir / "stdout.log").open("w") as out, (run_dir / "stderr.log").open("w") as err:
            proc = subprocess.Popen(
                cmd,
                cwd=workspace,
                env={**_without(os.environ, UPSTREAM_TOKEN_CMD_ENV), **builder_env, **(extra_env or {})},
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                exit_code = proc.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                exit_code = _stop(proc, group=True)
                timed_out = True
        t_end = time.monotonic()
    finally:
        _stop(proxy)
    calls = run_dir / "calls.jsonl"
    summary = {
        "run_id": run_dir.name,
        "harness": harness,
        "workspace": str(workspace),
        "started_at": started_at,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_s": round(t_end - t0, 3),
        "harness_args": list(harness_args),
        "steps": step_counts(load_calls(calls), t0, t_end) if calls.exists() else None,
        "harness_report": _harness_report(harness, run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--workspace", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--thinking", default="off")
    parser.add_argument("--random-seed", action="store_true", help="the proxy's --random-seed")
    parser.add_argument("--one-reasoning-field", action="store_true", help="the proxy's --one-reasoning-field")
    parser.add_argument("--log-bodies", action="store_true", help="the proxy's --log-bodies")
    parser.add_argument(
        "--harness-args",
        default="",
        help='extra flags for the harness command line, shell-quoted (localmelo only): --harness-args="--x 1"',
    )
    args = parser.parse_args()

    text = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    summary = run(
        args.harness,
        text,
        args.workspace.resolve(),
        args.run_dir.resolve(),
        timeout_s=args.timeout,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        thinking=args.thinking,
        random_seed=args.random_seed,
        one_reasoning_field=args.one_reasoning_field,
        log_bodies=args.log_bodies,
        harness_args=shlex.split(args.harness_args),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
