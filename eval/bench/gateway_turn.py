"""One task through localmelo's gateway chat: the path a person uses, as a command for the bench harness.

    python -m bench.gateway_turn --melo PATH --home DIR --run-dir DIR --base-url URL/v1 --model ID \\
        --prompt-file FILE [--embedding-model ID] [--timeout S] [-X KEY=VALUE ...] [--set table.key=value ...]

It starts ``melo serve`` on its own HOME with the shipped defaults (the chat model is reached through the bench's
recording proxy), opens a chat whose workspace is the current directory, sends the task as the first message, allows
every approval the work asks for, as a person doing the task would (each one is counted), waits for the turn to end and
stops the gateway.

A chat may give its work to a background task (the model starts one, or the turn hands the rest of its work off and
ends at once). The work is then not over when the turn is: the command also waits until every task of the gateway has
ended, allows the approvals they ask for, and asks them to stop when the time is up.

It writes ``<run-dir>/events.jsonl`` (the event stream of the turn and of every task, one event a line, each with its
``source``), ``<run-dir>/gateway_turn.json`` (status, cause, wall time, the approvals it allowed, the tasks and how
each ended) and the replies on stdout.
Exit 0: the turn completed and every task ended ``done``. 124: the work did not finish in time and was asked to stop.
1: any other end, or the gateway did not start. On TERM or INT the gateway is stopped before the exit (it runs in its
own process group).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from memory.across import client, server

EXIT_TIMEOUT = 124
TASK_FINISHED = ("done", "failed", "cancelled", "expired")  # the task states that do not change by themselves
QUIET_S = 3.0  # nothing running for this long: the work is over (an ended task can start one more that goes on with it)
STOP_WAIT_S = 20.0  # what tasks asked to stop get to end


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


def tasks_of(gw: Any) -> list[dict[str, Any]]:
    """The gateway's tasks, each with the approvals it waits for; none on a build without tasks."""
    status, body = gw.request("GET", "/api/tasks")
    rows = body.get("tasks") if status == 200 and isinstance(body, dict) else None
    return [r for r in rows if isinstance(r, dict) and r.get("task_id")] if isinstance(rows, list) else []


def follow_tasks(
    gw: Any, budget_s: float, *, expect_task: bool = False, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait until every task of the gateway has ended, allowing each approval a task asks for once.

    The gateway is the run's own, so every task on it belongs to the run. With no task at all it returns at once, unless
    ``expect_task`` says the turn handed its work off (the task is then given ``QUIET_S`` to show). After ``budget_s``
    the tasks still going are asked to stop, as the page's Stop does. Returns ``tasks`` (id and last state, in the
    order they appeared), ``approvals_allowed`` (tool names) and ``timed_out``."""
    start, quiet, stopping_since = clock(), None, None
    seen: dict[str, str] = {}
    allowed: list[str] = []
    answered: set[str] = set()
    while True:
        now = clock()
        tasks = tasks_of(gw)
        for task in tasks:
            seen[str(task["task_id"])] = str(task.get("state"))
        going = [t for t in tasks if t.get("state") not in TASK_FINISHED]
        if stopping_since is None and going and now - start >= budget_s:
            stopping_since = now
        if stopping_since is not None:
            for task in going:  # again each round: a task the stop let start is stopped too
                gw.request("PATCH", f"/api/tasks/{task['task_id']}", {"state": "stop"})
            if not going or now - stopping_since >= STOP_WAIT_S:
                break
        else:
            for task in going:
                for card in task.get("pending_approvals") or []:
                    key = str(card.get("request_id"))
                    if key not in answered and gw.allow_once(key):
                        allowed.append(str(card.get("tool_name") or ""))
                    answered.add(key)
            if going:
                quiet = None
            elif quiet is None:
                quiet = now
            if quiet is not None and ((not seen and not expect_task) or now - quiet >= QUIET_S):
                break
        sleep(1.0)
    for task in tasks_of(gw) if seen else ():  # the states after the last wait
        seen[str(task["task_id"])] = str(task.get("state"))
    return {"tasks": [{"task_id": k, "state": v} for k, v in seen.items()], "approvals_allowed": allowed,
            "timed_out": stopping_since is not None}


def task_events(gw: Any, task_id: str) -> list[dict[str, Any]]:
    """The stored events of a task's own runs (a task is a session of its own); none when they cannot be read."""
    try:
        return [{**e, "source": "task"} for e in gw.events_once(task_id)]
    except (client.GatewayError, OSError):
        return []


def task_reply(gw: Any, task_id: str) -> str:
    try:
        row = gw.last_turn(task_id)
    except (client.GatewayError, OSError, KeyError):
        return ""
    return str((row or {}).get("answer") or "")


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
        t0 = time.monotonic()
        result = client.run_turn(gw, prompt, f"bench-{uuid.uuid4().hex[:16]}", workspace=os.getcwd(), timeout_s=args.timeout,
                                 deny_approvals=False, allow_all=True)
        row = gw.last_turn(result.session_id) or {}
        handed_off = bool((row.get("result") or {}).get("handed_off")) if isinstance(row.get("result"), dict) else False
        # the work a chat gave to tasks goes on after its turn: wait for it within what is left of the time
        followed = follow_tasks(gw, max(args.timeout - (time.monotonic() - t0), 0.0), expect_task=handed_off)
        events = [{**e, "source": "chat"} for e in stamped(result.events, result.started_epoch)]
        replies = [result.answer]
        for task in followed["tasks"]:
            events += task_events(gw, task["task_id"])
            replies.append(task_reply(gw, task["task_id"]))
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as out:
            for event in events:
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
        record.update(status=result.status, cause=result.cause, error=result.error, wall_ms=result.wall_ms, wall_source=result.wall_source,
                      ttfe_ms=result.ttfe_ms, usage=result.usage, session_id=result.session_id, turn_id=result.turn_id,
                      approvals_allowed=[a["tool"] for a in result.allowed] + followed["approvals_allowed"], cancelled=result.cancelled,
                      events=len(events), handed_off=handed_off, tasks=followed["tasks"], tasks_timed_out=followed["timed_out"],
                      work_ms=round((time.monotonic() - t0) * 1000.0, 1))
        for reply in replies:
            sys.stdout.write(reply + ("\n" if reply and not reply.endswith("\n") else ""))
        if result.status == "timeout" or followed["timed_out"]:
            code = EXIT_TIMEOUT
        else:
            code = 0 if result.status == "completed" and all(t["state"] == "done" for t in followed["tasks"]) else 1
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
