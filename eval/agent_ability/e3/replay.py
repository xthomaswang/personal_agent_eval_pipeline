"""Replay the recorded reflection requests of E3-style studies to measure how often a reflection is unreadable.

    cd eval
    uv run python agent_ability/e3/replay.py --out <file.jsonl> --arms S0,V64,O20 --localmelo <clone> <dir> [<dir> ...]

A study's own reflections are few (five to eight per arm), too few to compare unreadable rates. Here each
reflection of the given arms (as the report counts them: the first request of a chain, not its retries) is
sent again `--repeats` times, exactly as the recorder proxy forwarded it (its sampling, tools and chat-template
settings included), straight to the model server, one request at a time. The order is repeat-major, so the arms
interleave and a stopped replay has a similar count for every reflection. Each reply is classified:

- readable: it holds a JSON object with a recommended_action key (the report's has_decision);
- tool_call: no text, a tool call instead;
- invalid: it names recommended_action but no JSON object around it parses;
- no_json: anything else;
- error: the request failed (retried on the next run).

A reply that is not readable is kept in full (`reply`), so a repair of it can be checked offline.
With `--full-replies`, every non-error row also keeps its full content and tool calls (`content`, `tool_calls`).

The output file is appended to and read back on a rerun, so a stopped replay resumes. The operator takes the
GPU lock and starts the server; this only checks that the lock is held by agent-benchmark. It stops before the
next request when a STOP file is next to the output file, and after three failed requests in a row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

AGENT_ABILITY = Path(__file__).resolve().parent.parent
EVAL_ROOT = AGENT_ABILITY.parent
for _root in (EVAL_ROOT, AGENT_ABILITY):  # bench and e3
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from e3 import report as rp  # noqa: E402
from e3 import run as e3_run  # noqa: E402

CLASSES = ("readable", "tool_call", "invalid", "no_json", "error")
MAX_ERRORS_IN_A_ROW = 3

Post = Callable[[str, dict[str, Any]], dict[str, Any]]


def classify(reply: dict[str, Any]) -> str:
    """readable, tool_call, invalid or no_json, for a chat-completions response."""
    message = ((reply.get("choices") or [{}])[0]).get("message") or {}
    text = rp._text(message.get("content"))
    if rp.has_decision(text):
        return "readable"
    if not text.strip() and message.get("tool_calls"):
        return "tool_call"
    return "invalid" if "recommended_action" in text else "no_json"


def content_and_tool_calls(reply: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The assistant's content as text, and its tool calls, each with its name and its full arguments string."""
    message = ((reply.get("choices") or [{}])[0]).get("message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": str(fn.get("arguments") or "")})
    return rp._text(message.get("content")), calls


def full_reply(reply: dict[str, Any]) -> str:
    """The reply as text: its content, then one line per tool call with the full arguments."""
    text, calls = content_and_tool_calls(reply)
    lines = [text] if text else []
    lines += [f"[tool_call] {call['name']}({call['arguments']})" for call in calls]
    return "\n".join(lines)


def _json(raw: tuple[bytes, str] | None) -> Any:
    try:
        return json.loads(raw[0]) if raw else None
    except json.JSONDecodeError:
        return None


def reflections(study: Path, arms: Sequence[str], prompt: str) -> list[dict[str, Any]]:
    """The reflections of a study's runs of the given arms, each with the body the proxy forwarded for it."""
    state = json.loads((study / "e3.json").read_text())
    found = []
    for record in sorted(state["runs"].values(), key=lambda r: r["index"]):
        if record.get("arm") not in arms or record.get("state") not in ("finished", "interrupted"):
            continue
        if not record.get("run_dir"):
            continue
        run_dir = study / record["run_dir"]
        previous_unreadable = False
        for call in rp._calls(run_dir):
            body = _json(rp._body(run_dir, call, "request"))
            if not isinstance(body, dict) or rp.is_reflection_request(body, prompt) is not True:
                previous_unreadable = False
                continue
            response = rp._body(run_dir, call, "response")
            readable = response is not None and rp.has_decision(rp.reply_text(*response))
            start, previous_unreadable = not previous_unreadable, not readable
            forwarded = rp._body(run_dir, call, "forwarded")
            if start and isinstance(_json(forwarded), dict):
                found.append(
                    {
                        "study": study.name, "run_id": record["run_id"], "arm": record["arm"],
                        "exercise": record["exercise"], "body": forwarded[1],
                        "recorded": "readable" if readable else "unreadable", "forwarded": _json(forwarded),
                    }
                )  # fmt: skip
    return found


def http_post(base_url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({**body, "stream": False}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def _key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return row["study"], row["run_id"], row["body"], row["repeat"]


def load(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []


def latest(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The last row of each (reflection, repeat): a retried error gives way to its retry."""
    return list({_key(r): r for r in rows}.values())


def replay(
    items: Sequence[dict[str, Any]],
    repeats: int,
    out: Path,
    base_url: str,
    post: Post = http_post,
    full_replies: bool = False,
) -> str | None:
    """Send what is not done yet; returns why it stopped early, or None when every reply is in."""
    done = {_key(r) for r in load(out) if r["class"] != "error"}
    errors_in_a_row = 0
    for repeat in range(1, repeats + 1):
        for item in items:
            if (item["study"], item["run_id"], item["body"], repeat) in done:
                continue
            if (out.parent / "STOP").exists():
                return "STOP file found"
            started = time.monotonic()
            try:
                reply, error = post(base_url, item["forwarded"]), None
            except (OSError, ValueError) as exc:  # urllib's errors are OSErrors; a bad JSON reply is a ValueError
                reply, error = {}, f"{type(exc).__name__}: {exc}"
            choice = (reply.get("choices") or [{}])[0]
            cls = "error" if error else classify(reply)
            row = {
                **{k: item[k] for k in ("study", "run_id", "arm", "exercise", "body", "recorded")},
                "repeat": repeat,
                "class": cls,
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": (reply.get("usage") or {}).get("completion_tokens"),
                "excerpt": None if error else rp.reply_excerpt(json.dumps(reply).encode(), "reply.json"),
                "reply": None if cls in ("readable", "error") else full_reply(reply),
                "seconds": round(time.monotonic() - started, 3),
                "error": error,
            }
            if full_replies and not error:
                row["content"], row["tool_calls"] = content_and_tool_calls(reply)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            errors_in_a_row = errors_in_a_row + 1 if error else 0
            if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                return f"{errors_in_a_row} failed requests in a row (last: {error})"
    return None


def summary(rows: Sequence[dict[str, Any]], arms: Sequence[str]) -> str:
    lines = [
        "| Arm | Reflections | Replies | Readable | tool_call | invalid | no_json | error | Unreadable rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = latest(rows)
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        counts = Counter(r["class"] for r in mine)
        replies = len(mine) - counts["error"]
        rate = f"{100 * (replies - counts['readable']) / replies:.1f}%" if replies else "-"
        n_reflections = len({(r["study"], r["run_id"], r["body"]) for r in mine})
        cells = " | ".join(str(counts[c]) for c in CLASSES)
        lines.append(f"| {arm} | {n_reflections} | {replies} | {cells} | {rate} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, post: Post = http_post) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("studies", type=Path, nargs="+", help="E3-style output directories (each has e3.json)")
    ap.add_argument("--out", type=Path, required=True, help="the replies, one JSON line each (appended; resumes)")
    ap.add_argument("--arms", required=True, help="comma-separated arms to replay")
    ap.add_argument("--repeats", type=int, default=6, help="replies per reflection (default 6)")
    ap.add_argument("--localmelo", type=Path, default=rp.VENDOR, help="the studies' clone, for REFLECTION_PROMPT")
    ap.add_argument("--base-url", default=None, help="the model server (default: bench.manifest's OMLX_BASE_URL)")
    ap.add_argument("--full-replies", action="store_true", help="also keep each reply's full content and tool calls")
    args = ap.parse_args(argv)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    prompt = rp.reflection_prompt(args.localmelo)
    items = []
    for study in args.studies:
        found = reflections(study.resolve(), arms, prompt)
        if not found:
            print(f"warning: no reflections of {', '.join(arms)} in {study} (the wrong --localmelo?)", file=sys.stderr)
        items += found
    if post is http_post:
        problem = e3_run.lock_problem()
        if problem:
            print(problem, file=sys.stderr)
            return 2
    if args.base_url is None:
        from bench import manifest

        args.base_url = manifest.OMLX_BASE_URL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stopped = replay(items, args.repeats, args.out, args.base_url, post, full_replies=args.full_replies)
    print(summary(load(args.out), arms))
    if stopped:
        print(f"stopped: {stopped}; run it again to resume", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
