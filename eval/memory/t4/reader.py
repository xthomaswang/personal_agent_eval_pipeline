"""T4's fixed reader: one model call per probe, answering from the memory store alone.

    uv run python -m memory.t4.reader --run-dir <run>                    # <run>/t4/store.json
    uv run python -m memory.t4.reader --run-dir <dir> --store <export>   # an exported store
    uv run python -m memory.t4.reader --run-dir <dir> --gold             # the gold store (the ceiling)
    uv run python -m memory.t4.reader --run-dir <dir> --filter           # empty store x 3 -> drop_list.yaml

The reader is a bench call, not an agent call: it runs through its own recorder proxy instance
(harness t4-reader, sampling passthrough, thinking off) and sends temperature 0. Answers go to
<run>/reader/answers.jsonl, the proxy's records to <run>/reader/. The caller holds the GPU lock and a
running oMLX (OMLX_BASE_URL, BENCH_MODEL; BENCH_UPSTREAM_TOKEN_CMD when the server needs a key).

--filter is T4's empty-memory filter: every probe is asked 3 times with an empty store, and an
answerable probe answered right at least once is dropped. Abstention probes are never dropped (an empty
store answers them right by construction). The list is written next to the fixtures with the reader's
setting, and is frozen with the fixture version.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from bench.harness import UPSTREAM, UPSTREAM_TOKEN_CMD_ENV, _stop
from memory.t4 import store as st
from memory.t4.oracle import ANSWERABLE, FIXTURES, Fixtures, score

ROOT = Path(__file__).resolve().parents[2]
MAX_TOKENS = 200
FORMATS = {
    "date": 'a date as "YYYY-MM-DD"',
    "datetime": 'a local date and time as "YYYY-MM-DD HH:MM" (24-hour clock)',
    "time": 'a local time as "HH:MM" (24-hour clock)',
    "money": "an amount in US dollars as a number, for example 12.5",
    "entity": "a name, as the store writes it",
    "list": "a JSON array of names",
}
SYSTEM = """You answer questions about {owner}'s life using only the memory store below. An assistant wrote \
the store from {owner}'s messages, emails and calendar between {start} and {end}.

Use only what the store says. If the store does not contain the answer, answer null.
Reply with one JSON object, {{"answer": ...}}, and nothing else.

Memory store:
{store}"""


def start_proxy(out: Path, run_id: str) -> tuple[subprocess.Popen[str], str]:
    cmd = [sys.executable, "-m", "bench.proxy", "--port", "0", "--upstream", UPSTREAM, "--harness", "t4-reader",
           "--run-id", run_id, "--run-dir", str(out), "--thinking", "off", "--sampling", "passthrough"]
    if os.environ.get(UPSTREAM_TOKEN_CMD_ENV):
        cmd += ["--upstream-token-cmd", os.environ[UPSTREAM_TOKEN_CMD_ENV]]
    log_path = out / "proxy.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        found = re.search(r"listening on (http://\S+)", log_path.read_text())
        if found:
            return proc, found.group(1)
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    proc.terminate()
    raise SystemExit(f"proxy did not start; see {log_path}")


def parse(text: str) -> tuple[Any, str | None]:
    """The answer from a reply, or an error when the reply holds no JSON object with "answer"."""
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"], None
    return None, "no JSON object with an answer"


def ask(client: httpx.Client, url: str, model: str, system: str, probe: dict[str, Any]) -> dict[str, Any]:
    body = {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"Question: {probe['question']}\nAnswer with "
                                                     f"{FORMATS[probe['type']]}, or null."}]}
    t0 = time.monotonic()
    try:
        r = client.post(f"{url}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"answer": None, "raw": None, "error": f"request: {e}", "latency_s": time.monotonic() - t0}
    msg = data["choices"][0]["message"]
    raw = msg.get("content") or ""
    answer, error = parse(raw)
    return {"answer": answer, "raw": raw, "error": error, "latency_s": round(time.monotonic() - t0, 3),
            "usage": data.get("usage")}


def system_info() -> dict[str, str | None]:
    def run(cmd: list[str]) -> str | None:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    xcode = run(["xcodebuild", "-version"])
    repo = os.environ.get("OMLX_REPO")
    return {"server": f"oMLX {run(['git', '-C', repo, 'describe', '--tags', '--always'])}" if repo else None,
            "macos_build": run(["sw_vers", "-buildVersion"]),
            "xcode_build": xcode.splitlines()[-1].split()[-1] if xcode else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--store", type=Path, help="an exported store (default: <run>/t4/store.json)")
    src.add_argument("--gold", action="store_true", help="the gold store, for the ceiling")
    src.add_argument("--filter", action="store_true", help="the empty-memory filter: empty store, 3 reps")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = ap.parse_args()

    fx = Fixtures(args.fixtures)
    if args.filter:
        entries, reps = [], 3
    elif args.gold:
        entries, reps = st.gold(fx.facts), args.reps
    else:
        entries, reps = st.load(args.store or args.run_dir / "t4" / "store.json"), args.reps
    tl = fx.facts["timeline"]
    end = dt.date.fromisoformat(tl["start"]) + dt.timedelta(days=tl["days"] - 1)
    system = SYSTEM.format(owner=fx.facts["entities"][fx.owner]["name"], start=tl["start"], end=end.isoformat(),
                           store=st.render(entries))
    model = os.environ.get("BENCH_MODEL")
    if not model:
        raise SystemExit("BENCH_MODEL is not set")

    out = args.run_dir / "reader"
    out.mkdir(parents=True, exist_ok=True)
    proc, url = start_proxy(out, f"{args.run_dir.name}-reader")
    answers = []
    try:
        with httpx.Client(timeout=300) as client, (out / "answers.jsonl").open("w") as f:
            for rep in range(reps):
                for p in fx.probes:
                    row = {"probe_id": p["id"], "rep": rep, **ask(client, url, model, system, p)}
                    row["correct"] = row["error"] is None and fx.correct(p, row["answer"])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    answers.append(row)
    finally:
        _stop(proc)

    errors = sum(a["error"] is not None for a in answers)
    if args.filter:
        right: dict[str, int] = {}
        for a in answers:
            right[a["probe_id"]] = right.get(a["probe_id"], 0) + bool(a["correct"])
        cat = {p["id"]: p["category"] for p in fx.probes}
        dropped = sorted(i for i, n in right.items() if n and cat[i] in ANSWERABLE)
        info = {
            "version": 1,
            "rule": "an answerable probe answered right at least once in 3 reps from an empty store is dropped; "
                    "abstention probes are never dropped",
            "reader": {"model": model, "thinking": "off", "temperature": 0, "max_tokens": MAX_TOKENS, "reps": reps,
                       "date": dt.date.today().isoformat(), **system_info()},
            "errors": errors,
            "dropped": dropped,
            "right_from_empty": {i: n for i, n in sorted(right.items()) if n},
        }
        (args.fixtures / "drop_list.yaml").write_text(yaml.safe_dump(info, sort_keys=False, width=120))
        print(f"dropped {len(dropped)} of {sum(c in ANSWERABLE for c in cat.values())} answerable probes; "
              f"{errors} errors -> {args.fixtures / 'drop_list.yaml'}")
        return 1 if errors else 0
    s = score(Fixtures(args.fixtures), answers)
    cats = " ".join(f"{c}={v['correct']}/{v['n']}" for c, v in s["per_category"].items())
    print(f"answerable {s['answerable']['correct']}/{s['answerable']['n']}  {cats}  errors {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
