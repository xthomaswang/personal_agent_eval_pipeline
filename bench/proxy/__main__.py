"""Run the model-call recording proxy.

Usage:
    uv run python -m bench.proxy --port 8600 --harness claude-code --thinking off \
        --run-dir runs/<run-id> --run-id <run-id>
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from . import thinking
from .config import (
    BUDGET_MODES,
    DEFAULT_PORT,
    DEFAULT_READ_TIMEOUT,
    ProxyConfig,
    default_run_dir,
    new_run_id,
    upstream_from_env,
    upstream_token,
)
from .recorder import CALLS_FILE
from .sampling import DEFAULT_SAMPLING, POLICIES_HELP, SamplingPolicy
from .thinking import MODES_HELP, ThinkingMode


def _thinking(value: str) -> ThinkingMode:
    try:
        return ThinkingMode.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _sampling(value: str) -> SamplingPolicy:
    try:
        return SamplingPolicy.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _steps(value: str) -> int:
    steps = int(value)
    if steps < 0:
        raise argparse.ArgumentTypeError("--max-steps must not be negative")
    return steps


def _upstream_token(command: str | None) -> str | None:
    try:
        return upstream_token(command) if command else None
    except ValueError as exc:
        raise SystemExit(f"--upstream-token-cmd: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.proxy",
        description="Record every model call a harness makes, relaying it to oMLX.",
    )
    parser.add_argument(
        "--upstream",
        help="model server root URL; a trailing /v1 is stripped "
        "(default: OMLX_BASE_URL without /v1, else http://127.0.0.1:8500)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"listen port; real runs use 8600-8699, 0 picks a free one (default {DEFAULT_PORT})",
    )
    parser.add_argument("--run-id", help="run identifier written into every record (default: generated)")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="directory for calls.jsonl and bodies/ (default: $BENCH_RUNS_DIR/<run-id>, else runs/<run-id>)",
    )
    parser.add_argument("--harness", help="harness label written into every record, e.g. hermes")
    parser.add_argument(
        "--thinking",
        type=_thinking,
        default=ThinkingMode("passthrough"),
        metavar="MODE",
        help=f"thinking level forced on every model request: {MODES_HELP} (default passthrough)",
    )
    parser.add_argument(
        "--sampling",
        type=_sampling,
        default=DEFAULT_SAMPLING,
        metavar="POLICY",
        help=f"sampling fields of every generation request: {POLICIES_HELP} (default {DEFAULT_SAMPLING})",
    )
    parser.add_argument(
        "--max-steps",
        type=_steps,
        help="step budget: successful main calls (n_llm) allowed in a run (default: no budget)",
    )
    parser.add_argument(
        "--budget-mode",
        choices=BUDGET_MODES,
        default="reject",
        help="a main call beyond --max-steps is refused with HTTP 400 bench_budget_exhausted "
        "(reject, default) or forwarded and only flagged (record)",
    )
    parser.add_argument(
        "--log-bodies",
        action="store_true",
        help="also store full request and response bodies under <run-dir>/bodies/",
    )
    parser.add_argument(
        "--inject-stream-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add stream_options.include_usage=true to streaming OpenAI requests (default on)",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=DEFAULT_READ_TIMEOUT,
        help=f"seconds without upstream data before a call fails (default {DEFAULT_READ_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="set a fresh random seed on every OpenAI generation request, after --sampling; for servers whose "
        "default seed is fixed (rec), where every unseeded request would follow the same sample path",
    )
    parser.add_argument(
        "--one-reasoning-field",
        action="store_true",
        help="send assistant messages that carry both reasoning_content and reasoning with only one of them "
        "(rec rejects both; Hermes sends both to local endpoints)",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print a line per call to stderr")
    parser.add_argument(
        "--upstream-token-cmd",
        metavar="CMD",
        help="command printing a bearer token for an upstream that needs one, e.g. 'rec auth token'; "
        "run once at startup, sent upstream in place of the client's Authorization, never recorded",
    )
    return parser


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    return sock


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or new_run_id()
    config = ProxyConfig(
        upstream=args.upstream or upstream_from_env(),
        run_dir=args.run_dir or default_run_dir(run_id),
        run_id=run_id,
        harness=args.harness,
        thinking=args.thinking,
        sampling=args.sampling,
        max_steps=args.max_steps,
        budget_mode=args.budget_mode,
        log_bodies=args.log_bodies,
        inject_stream_usage=args.inject_stream_usage,
        random_seed=args.random_seed,
        one_reasoning_field=args.one_reasoning_field,
        read_timeout=args.read_timeout,
        echo=not args.quiet,
        upstream_token=_upstream_token(args.upstream_token_cmd),
    )
    app = create_app(config)
    # Bind first so that the real port is known (and printed) for --port 0.
    sock = _bind(args.host, args.port)
    port = sock.getsockname()[1]

    def say(line: str) -> None:
        print(f"[proxy] {line}", file=sys.stderr, flush=True)

    for warning in thinking.startup_warnings(config.thinking):
        say(f"warning: {warning}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        say(f"warning: listening on {args.host}; the proxy has no authentication")
    say(f"listening on http://{args.host}:{port} -> upstream {config.upstream}")
    say(
        f"run_id={config.run_id} harness={config.harness or '-'} thinking={config.thinking} "
        f"sampling={config.sampling}{'+random-seed' if config.random_seed else ''} "
        f"max_steps={config.max_steps if config.max_steps is not None else '-'} ({config.budget_mode}) "
        f"log_bodies={'on' if config.log_bodies else 'off'} "
        f"inject_stream_usage={'on' if config.inject_stream_usage else 'off'}"
        f"{' one_reasoning_field=on' if config.one_reasoning_field else ''}"
    )
    if config.upstream_token:
        say(f"upstream auth: bearer token from {args.upstream_token_cmd!r}")
    say(f"records -> {config.run_dir / CALLS_FILE}")

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="warning",
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=5,
        )
    )
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
