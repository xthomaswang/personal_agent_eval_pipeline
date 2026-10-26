"""Check localmelo's bench contract (Phase 1a and later) on every localmelo run of a smoke run root.

Usage: python3 -I check_localmelo_contract.py [--reflection-retry auto|on|off] <harness-smoke run root>...

Criteria 1-6 are the ones localmelo_main set for its Phase 1a handoff; 7 comes with PR #31:
1. stdout is exactly one JSON line (`--output-format json`), and the exit code matches its status:
   0 completed, 1 crashed, 2 any other status;
2. no crashed run, and no traceback on stderr;
3. no task row in history.db left as `running`;
4. every model request carries X-Localmelo-Attempt (needs the proxy's --log-bodies); per run, requests
   with tools = n_main, requests without = n_aux, successful responses = n_llm, failed ones = llm_errors;
5. with `--tool-exposure all`, every planning request offers the same tools, sorted by name;
6. planning requests do not exceed --max-steps;
7. a reflection (a request without tools whose system message opens with localmelo's REFLECTION_PROMPT)
   whose reply holds no JSON object is asked for once more, right after it: the same system message,
   then one user message with the turns written out as text, the same request and REFLECTION_RETRY_NOTE.
   No other reply is asked for again, a retry's included. Whether a run's localmelo retries comes from
   its manifest (the commit's melo/agent/_chat.py defines REFLECTION_RETRY_NOTE) unless
   --reflection-retry says so; when the manifest cannot tell, only the retries that happened are checked.
   With the reflection gate's `flags` setting (localmelo's gate PR), a reply is asked for again also
   when its JSON objects hold no valid recommended_action, such as a tool call written as text; under
   `utility` and before the gate PR, only a reply without any JSON object is. The setting comes from
   the JSON line's config, else the run's --reflection-gate flag, else the commit's default.
A run the bench killed at its time budget (SIGTERM) prints no JSON line by design; it is reported as
killed, and only criteria 3-7 apply to it. Exit status 1 when any run fails a criterion.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sqlite3
import subprocess
import sys

ATTEMPT_HEADER = "x-localmelo-attempt"
# The opening of localmelo's REFLECTION_PROMPT, and its REFLECTION_TRANSCRIPT_INTRO and
# REFLECTION_RETRY_NOTE (melo/agent/_chat.py, PR #31)
REFLECTION_START = "The current attempt has ended without a final answer."
TRANSCRIPT_INTRO = "The end of this attempt, written out as text:"
RETRY_NOTE = (
    "Reply with only the JSON object described above. Do not call any tool: tools are not available "
    "for this reply."
)
# The decisions a reflection may recommend (melo/agent/agent.py REFLECTION_ACTIONS at 382cb58)
REFLECTION_ACTIONS = ("continue", "stop", "decompose", "done")


def expected_exit(status: str) -> int:
    return {"completed": 0, "crashed": 1}.get(status, 2)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _strip_fence(text: str) -> str:
    """localmelo's _strip_code_fences: the reply without a Markdown code fence around it."""
    body = text.strip()
    if body.startswith("```"):
        newline = body.find("\n")
        body = body[newline + 1 :] if newline != -1 else body[3:]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def holds_reflection_object(text: str) -> bool:
    """localmelo's _holds_reflection_object: the reply, without a code fence around it, is a JSON
    object, or is not JSON as a whole and holds one (the first `{` that decodes)."""
    body = _strip_fence(text)
    try:
        return isinstance(json.loads(body), dict)
    except (ValueError, RecursionError):
        pass
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            return isinstance(decoder.raw_decode(body, start)[0], dict)
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
        except (ValueError, RecursionError):  # a number the decoder cannot convert ends the search, as in localmelo
            break
    return False


def reflection_object(text: str) -> dict | None:
    """The reflection a reply holds under the gate's `flags` setting (gate spec, requirement 4): the
    first JSON object whose recommended_action, as a string, is one of REFLECTION_ACTIONS. Like
    _decode_reflection, the reply without a fence is read whole first; when it is not JSON, each `{`
    that decodes is tried in order, and the search goes on after an object without a valid action.
    A bench copy of localmelo's _find_reflection (gate branch 0f712f5), which gave the same answer on 87
    replies (E4's, the reflection probes' and 17 edge cases); None when the reply holds none."""
    body = _strip_fence(text)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        pass
    else:
        return value if isinstance(value, dict) and str(value.get("recommended_action")) in REFLECTION_ACTIONS else None
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
            continue
        except (ValueError, RecursionError):  # as in localmelo's _find_reflection
            break
        if str(value.get("recommended_action")) in REFLECTION_ACTIONS:
            return value
        start = body.find("{", end)
    return None


def holds_reflection(text: str) -> bool:
    return reflection_object(text) is not None


def reflection_kind(body: dict) -> str | None:
    """'first' for a reflection request, 'retry' for its retry, None for any other request without tools."""
    messages = body.get("messages") or []
    if len(messages) < 2 or messages[0].get("role") != "system":
        return None
    if not _text(messages[0].get("content")).startswith(REFLECTION_START):
        return None
    return "retry" if _text(messages[-1].get("content")).endswith(RETRY_NOTE) else "first"


def _retry_shape(first: dict, retry: dict) -> list[str]:
    sent, messages = first["body"]["messages"], retry["body"]["messages"]
    call = retry["call"]
    if len(messages) != 2 or messages[1].get("role") != "user":
        return [f"7: the retry at call {call} is not one system and one user message"]
    problems = []
    if messages[0] != sent[0]:
        problems.append(f"7: the retry at call {call} changes the system message")
    request, text = _text(sent[-1].get("content")), _text(messages[1].get("content"))
    # render_transcript writes a paragraph for every turn but an assistant one with neither text nor calls
    turns = [m for m in sent[1:-1] if m.get("role") != "assistant" or _text(m.get("content")).strip() or m.get("tool_calls")]
    if not text.endswith(f"{request}\n\n{RETRY_NOTE}"):
        problems.append(f"7: the retry at call {call} does not end with the reflection's request and the note")
    elif turns and not text.startswith(f"{TRANSCRIPT_INTRO}\n\n"):
        problems.append(f"7: the retry at call {call} does not open with the turns written out as text")
    elif not turns and text != f"{request}\n\n{RETRY_NOTE}":
        problems.append(f"7: the retry at call {call} has text beyond the request and the note")
    return problems


def check_retries(seq: list[dict], expected: bool | None, killed: bool, gate: str | None = None) -> list[str]:
    """Criterion 7 on a run's model requests in order. Each item has the call index, the request's
    reflection_kind (None for a planning request), its body, and the reply's text (None if not read).
    *gate* is the run's reflection gate setting: under "flags" a reply is asked for again unless it
    holds a reflection (holds_reflection); otherwise unless it holds any JSON object."""
    readable, what = (holds_reflection, "a reflection") if gate == "flags" else (holds_reflection_object, "a JSON object")
    problems = []
    for i, item in enumerate(seq):
        if item["kind"] == "retry":
            prev = seq[i - 1] if i else None
            if prev is None or prev["kind"] != "first":
                problems.append(f"7: the retry at call {item['call']} does not follow a reflection")
            elif prev["reply"] is not None and readable(prev["reply"]):
                problems.append(f"7: call {item['call']} asks again although call {prev['call']}'s reply holds {what}")
            else:
                problems.extend(_retry_shape(prev, item))
            if expected is False:
                problems.append(f"7: a retry at call {item['call']}, from a localmelo without the retry")
            continue
        if item["kind"] != "first" or item["reply"] is None or readable(item["reply"]):
            continue
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if nxt is not None and nxt["kind"] == "retry":
            continue  # checked at the retry
        if expected and not (nxt is None and killed):
            where = f"call {nxt['call']} is not its retry" if nxt else "no retry follows"
            problems.append(f"7: call {item['call']}'s reply holds no {what.removeprefix('a ')}, and {where}")
    return problems


@functools.lru_cache
def _defines(path: str, commit: str, file: str, token: str) -> bool | None:
    """Whether *file* at *commit* of the repo at *path* contains *token*; None when git cannot tell."""
    out = subprocess.run(["git", "-C", path, "show", f"{commit}:{file}"], capture_output=True, text=True, check=False)
    return token in out.stdout if out.returncode == 0 else None


def _manifest_localmelo(run: pathlib.Path) -> dict | None:
    """The run manifest's localmelo entry when it names a clean commit, else None."""
    try:
        localmelo = json.loads((run / "manifest.json").read_text())["harnesses"]["localmelo"]
        if localmelo.get("dirty") or not localmelo["commit"] or not localmelo["path"]:
            return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return localmelo


def retry_expected(run: pathlib.Path, mode: str) -> bool | None:
    if mode != "auto":
        return mode == "on"
    localmelo = _manifest_localmelo(run)
    if localmelo is None:
        return None
    return _defines(localmelo["path"], localmelo["commit"], "melo/agent/_chat.py", "REFLECTION_RETRY_NOTE")


def reflection_gate(record: dict, summary: dict, localmelo: dict | None) -> str | None:
    """The run's reflection gate setting: the JSON line's config.reflection_gate, else the value of the
    run's --reflection-gate flag, else "flags" when the commit has the option (its default), else None
    (a localmelo before the gate PR, which behaves as "utility")."""
    gate = (record.get("config") or {}).get("reflection_gate")
    if gate:
        return str(gate)
    args = [str(a) for a in summary.get("harness_args") or []]
    for i, arg in enumerate(args):
        if arg == "--reflection-gate" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--reflection-gate="):
            return arg.split("=", 1)[1]
    if localmelo and _defines(localmelo["path"], localmelo["commit"], "melo/agent/options.py", "reflection_gate"):
        return "flags"
    return None


def _reply_text(run: pathlib.Path, call: dict) -> str | None:
    path = (call.get("bodies") or {}).get("response")
    if path is None or call.get("status") != 200 or call.get("error"):
        return None
    try:
        message = json.loads((run / path).read_text())["choices"][0]["message"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None  # a streamed or unreadable body
    return _text(message.get("content"))


def check_run(run: pathlib.Path, retry_mode: str = "auto") -> tuple[list[str], dict]:
    summary = json.loads((run / "summary.json").read_text())
    problems: list[str] = []
    killed = bool(summary.get("timed_out"))
    lines = [line for line in (run / "stdout.log").read_text(errors="replace").splitlines() if line.strip()]
    record: dict = {}
    if killed:
        if lines:
            problems.append(f"1: killed at the budget, yet stdout has {len(lines)} line(s)")
    elif len(lines) != 1:
        problems.append(f"1: stdout has {len(lines)} lines, not one JSON line")
    if lines and not killed:
        try:
            record = json.loads(lines[-1])
        except ValueError:
            problems.append("1: the stdout line is not JSON")
    status = record.get("status")
    if record and summary.get("exit_code") != expected_exit(status):
        problems.append(f"1: exit {summary.get('exit_code')} with status {status} (expected {expected_exit(status)})")
    if status == "crashed":
        problems.append("2: status crashed")
    stderr = (run / "stderr.log").read_text(errors="replace")
    if "Traceback (most recent call last)" in stderr:
        problems.append("2: traceback on stderr")
    db = run / "mem" / "history.db"
    if db.exists():
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            running = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'running'").fetchone()[0]
        if running:
            problems.append(f"3: {running} task row(s) still running")
    else:
        problems.append("3: no history.db")

    calls = [json.loads(line) for line in (run / "calls.jsonl").read_text().splitlines() if line.strip()]
    chat = [c for c in calls if c.get("endpoint") == "chat_completions" and not c.get("handled_locally")]
    main = [c for c in chat if c.get("request_tool_names")]
    aux = [c for c in chat if not c.get("request_tool_names")]
    ok = [c for c in chat if c.get("status") == 200 and not c.get("error")]
    missing_header = []
    for c in chat:
        headers_path = (c.get("bodies") or {}).get("headers")
        if headers_path is None:
            missing_header.append(f"{c['call_index']} (no headers logged)")
            continue
        pairs = json.loads((run / headers_path).read_text())["request"]  # paths are relative to the run
        if not any(k.lower() == ATTEMPT_HEADER for k, _ in pairs):
            missing_header.append(str(c["call_index"]))
    if missing_header:
        problems.append(f"4: no {ATTEMPT_HEADER} on calls {', '.join(missing_header[:5])}")
    counts = {"n_main": len(main), "n_aux": len(aux), "n_llm": len(ok), "llm_errors": len(chat) - len(ok)}
    if record:
        for key, seen in counts.items():
            if record.get(key) != seen:
                problems.append(f"4: {key} {record.get(key)} in the JSON line, {seen} in the proxy records")
    config = record.get("config") or {}
    if config.get("tool_exposure") == "all" and main:
        names = [tuple(c["request_tool_names"]) for c in main]
        hashes = {c.get("request_tools_sha256") for c in main}
        if any(list(n) != sorted(n) for n in names):
            problems.append("5: tools not sorted by name")
        if len(hashes) != 1:
            problems.append(f"5: {len(hashes)} different tools arrays across planning requests")
    max_steps = config.get("max_steps")
    if max_steps is not None and len(main) > max_steps:
        problems.append(f"6: {len(main)} planning requests over --max-steps {max_steps}")
    seq = []
    for c in sorted(chat, key=lambda c: c["call_index"]):
        request_path = (c.get("bodies") or {}).get("request")
        body = json.loads((run / request_path).read_text()) if request_path and not c.get("request_tool_names") else None
        kind = reflection_kind(body) if body else None
        reply = _reply_text(run, c) if kind == "first" else None
        seq.append({"call": c["call_index"], "kind": kind, "body": body if kind else None, "reply": reply})
    expected = retry_expected(run, retry_mode)
    gate = reflection_gate(record, summary, _manifest_localmelo(run))
    problems.extend(check_retries(seq, expected, killed, gate))
    facts = {
        "exit": summary.get("exit_code"),
        "killed": killed,
        "status": status if record else summary.get("harness_report", {}).get("task_status"),
        **counts,
        "reflections": sum(1 for s in seq if s["kind"] == "first"),
        "retries": sum(1 for s in seq if s["kind"] == "retry"),
        "retry_expected": expected,
        "reflection_gate": gate,
        "line": lines[-1][:400] if lines else None,
    }
    return problems, facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=pathlib.Path, help="harness-smoke run roots")
    parser.add_argument(
        "--reflection-retry",
        choices=["auto", "on", "off"],
        default="auto",
        help="whether localmelo asks a reflection again (criterion 7); auto reads each run's manifest",
    )
    args = parser.parse_args()
    failed = total = unknown = 0
    for root in args.roots:
        for summary in sorted(root.glob("localmelo/*/summary.json")):
            run = summary.parent
            total += 1
            problems, facts = check_run(run, args.reflection_retry)
            unknown += facts["retry_expected"] is None
            label = f"{run.relative_to(root.parent)}: exit {facts['exit']}, status {facts['status']}"
            label += ", killed at the budget" if facts["killed"] else ""
            label += f", main {facts['n_main']} aux {facts['n_aux']} ok {facts['n_llm']} errors {facts['llm_errors']}"
            if facts["reflections"] or facts["retries"]:
                label += f", reflections {facts['reflections']} retried {facts['retries']}"
            label += f", gate {facts['reflection_gate']}" if facts["reflection_gate"] else ""
            if problems:
                failed += 1
                print(f"FAIL {label}")
                for p in problems:
                    print(f"     {p}")
                print(f"     JSON line: {facts['line']}")
            else:
                print(f"ok   {label}")
    print(f"{total - failed} of {total} localmelo runs meet criteria 1-7")
    if unknown:
        print(f"{unknown} run(s) with no clean localmelo commit in the manifest: criterion 7 checked only the retries that happened")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
