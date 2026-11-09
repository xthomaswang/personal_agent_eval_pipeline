"""One task through localmelo's gateway chat: the path a person uses, as a command for the bench harness.

    python -m bench.gateway_turn --melo PATH --home DIR --run-dir DIR --base-url URL/v1 --model ID \\
        --prompt-file FILE [--embedding-model ID] [--timeout S] [-X KEY=VALUE ...] [--set table.key=value ...]

It starts ``melo serve`` on its own HOME with the shipped defaults (the chat model is reached through the bench's
recording proxy), opens a chat whose workspace is the current directory, sends the task as the first message, allows
every approval the turn asks for, as a person doing the task would (each one is counted), waits for the turn to end and
stops the gateway.

It writes ``<run-dir>/events.jsonl`` (the turn's event stream, one event a line), ``<run-dir>/gateway_turn.json``
(status, cause, wall time, the approvals it allowed) and the reply on stdout.
Exit 0: the turn completed. 124: it did not finish in time and was asked to stop. 1: any other end, or the gateway did
not start. On TERM or INT the gateway is stopped before the exit (it runs in its own process group).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memory.across import client, server

EXIT_TIMEOUT = 124


def setting(text: str) -> tuple[str, Any]:
    """``table.key=value`` of ``--set``: the value is read as JSON when it is JSON (a number, true, a list), else as text."""
    key, sep, value = text.partition("=")
    if not sep or "." not in key:
        raise argparse.ArgumentTypeError(f"--set takes table.key=value, got {text!r}")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def stamped(events: Sequence[dict[str, Any]], started_epoch: float) -> list[dict[str, Any]]:
    """The events with a Unix ``ts`` each: the gateway's own, else the client's ``t_ms`` after the message was sent."""
    out = []
    for event in events:
        if not isinstance(event.get("ts"), (int, float)) and isinstance(event.get("t_ms"), (int, float)):
            event = {**event, "ts": round(started_epoch + event["t_ms"] / 1000.0, 3)}
        out.append(event)
    return out


def run(args: argparse.Namespace) -> int:
    run_dir, home = args.run_dir.resolve(), args.home.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("the task text is blank")
    server.write_config(home, server.config_document(args.base_url, args.model, args.embedding_model, None, dict(args.set)))
    spec = server.LaunchSpec(args.melo, home, server.free_port(), tuple(args.x), run_dir / "gateway.log")
    record: dict[str, Any] = {"status": "not_started", "workspace": os.getcwd(), "timeout_s": args.timeout, "x": list(args.x),
                              "settings": dict(args.set), "embedding_model": args.embedding_model}
    gateway = server.launch(spec)

    def stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt  # unwinds through the finally below, which stops the gateway

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        gw = server.wait_ready(gateway, timeout=args.ready_timeout)
        result = client.run_turn(gw, prompt, f"bench-{uuid.uuid4().hex[:16]}", workspace=os.getcwd(), timeout_s=args.timeout,
                                 deny_approvals=False, allow_all=True)
        events = stamped(result.events, result.started_epoch)
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as out:
            for event in events:
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
        record.update(status=result.status, cause=result.cause, error=result.error, wall_ms=result.wall_ms, wall_source=result.wall_source,
                      ttfe_ms=result.ttfe_ms, usage=result.usage, session_id=result.session_id, turn_id=result.turn_id,
                      approvals_allowed=[a["tool"] for a in result.allowed], cancelled=result.cancelled, events=len(events))
        sys.stdout.write(result.answer + ("\n" if result.answer and not result.answer.endswith("\n") else ""))
        code = 0 if result.status == "completed" else EXIT_TIMEOUT if result.status == "timeout" else 1
    except KeyboardInterrupt:
        record.update(status="stopped", error="stopped by a signal")
    except (server.LaunchError, client.GatewayError, OSError) as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        print(f"gateway_turn: {record['error']}", file=sys.stderr)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)  # nothing may cut the stop short
        record["gateway_exit"] = gateway.stop()
        (run_dir / "gateway_turn.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--melo", type=Path, required=True, help="the melo command of the checkout under test")
    ap.add_argument("--home", type=Path, required=True, help="the HOME the gateway runs with (its config is written here)")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True, help="the chat model's server up to /v1 (the recording proxy)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--embedding-model", default=None, help="when given, memory recall uses it through the same server")
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds the turn may take (default %(default)s)")
    ap.add_argument("--ready-timeout", type=float, default=120.0, help="seconds the gateway may take to answer (default %(default)s)")
    ap.add_argument("-X", dest="x", action="append", default=[], metavar="KEY=VALUE", help="a `melo serve -X` pair; may repeat")
    ap.add_argument("--set", action="append", default=[], type=setting, metavar="TABLE.KEY=VALUE", help="a config.toml setting; may repeat")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
