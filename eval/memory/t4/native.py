"""Drive localmelo's native T4 trial, then evaluate both exported stores.

    BENCH_LOCALMELO_REPO=memory/vendor/localmelo uv run python -m memory.t4.native --run-dir <trial>

The caller owns the GPU lock for real-model runs. Interrupted ingestion sessions are recorded, not
replayed against an already-mutated memory; evaluation phases can be retried safely. See
memory/native_t4_and_d11.md, section 1. R5/R6-specific admission and recall defaults are preserved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from memory.t4 import routing
from memory.t4.oracle import FIXTURES, Fixtures

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "memory" / "vendor" / "localmelo"
PROMPT = (
    "Today is {date}. Read today's records with the feed_day tool and remember what they establish about "
    "Jamie's plans, people, places, dates and amounts, so that you can answer questions about them later. "
    "Use absolute dates, and when something changes, remember the new value. Call the submit tool with "
    'status "done" when you are finished.'
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def module_command(module: str, *args: str) -> list[str]:
    # eval is deliberately not installed as a package. Name its trusted root explicitly under -I,
    # rather than relying on the harness workspace or an inherited PYTHONPATH to find pab.
    code = f"import runpy,sys; sys.path.insert(0, {str(ROOT)!r}); runpy.run_module({module!r}, run_name='__main__')"
    return [str(ROOT / ".venv" / "bin" / "python"), "-I", "-B", "-c", code, *args]


def mcp_config(trial: Path, day: int, fixtures: Path) -> dict[str, Any]:
    command = module_command("memory.pab", "--run-dir", str(trial), "--session", str(day),
                             "--task", "t4-native", "--fixtures", str(fixtures))
    return {"mcpServers": {"pab": {"command": command[0], "args": command[1:], "cwd": str(ROOT)}}}


def _run_day(prompt: str, workspace: Path, run_dir: Path, config: Path, trial: Path,
             localmelo: Path) -> dict[str, Any]:
    from bench import harness, manifest

    if manifest.LOCALMELO_REPO.resolve() != localmelo.resolve():
        raise RuntimeError("Set BENCH_LOCALMELO_REPO before importing bench.harness")
    return harness.run(
        "localmelo", prompt, workspace, run_dir, max_steps=30, timeout_s=900, thinking="off",
        harness_args=["--mcp-only", "--mcp-config", str(config), "--reasoning", "compress",
                      "--keep-decision-text", "--output-format", "json"],
        extra_env={"LOCALMELO_PERSIST_MEMORY": "1", "LOCALMELO_MEMORY_DIR": str(trial / "mem"),
                   "HOME": str(trial / "home"), "PYTHONDONTWRITEBYTECODE": "1"},
    )


def log_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a banner or the last line of an interrupted process
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def session_result(day: int, run_dir: Path, summary: dict[str, Any] | None) -> dict[str, Any]:
    from bench.proxy.steps import step_counts

    records = log_records(run_dir / "stdout.log")
    record = next((r for r in reversed(records) if "task_id" in r and "status" in r), {})
    calls = log_records(run_dir / "calls.jsonl")
    counts = (summary.get("steps") if summary else None) or step_counts(calls)
    timed_out = bool(summary and summary.get("timed_out"))
    budget = timed_out or any(c.get("budget_exhausted") for c in calls)
    return {"day": day, "state": "finished" if summary else "interrupted", "task_id": record.get("task_id"),
            "status": record.get("status", "interrupted"), "exit_code": summary.get("exit_code") if summary else None,
            "budget_exhausted": budget, "timed_out": timed_out,
            "model_calls": counts["n_llm"],
            "wall_s": summary.get("wall_s") if summary else None}


def rebuild_everything(decisions: list[dict[str, Any]], out: Path) -> int:
    """Keep every candidate item, including ignored duplicates, without policy or text rewriting."""
    entries = []
    for n, row in enumerate(decisions, 1):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"decision {row.get('seq', n)} has no text; ingestion needs --keep-decision-text")
        entries.append({"id": f"decision-{row.get('seq', n)}", "text": row["text"],
                        "source_ids": row.get("source_ids") or []})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    tmp.replace(out)
    return len(entries)


def _export(localmelo: Path, trial: Path, flag: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    missing = not (trial / "mem" / "long_term.db").exists()
    command = [str(localmelo / ".venv" / "bin" / "melo-retrieve"), flag, str(trial / "mem")]
    errors = target.with_suffix(".err")
    with tmp.open("w") as out, errors.open("w") as err:
        result = subprocess.run(command, stdout=out, stderr=err,
                                env={**os.environ, "HOME": str(trial / "home")})
    save(target.with_suffix(".diagnostic.json"), {"exit_code": result.returncode,
                                                "stderr": errors.read_text(), "missing_store": missing})
    if result.returncode and not missing:
        raise subprocess.CalledProcessError(result.returncode, command)
    if missing:  # Older pins exit 2, newer pins exit 0: neither means the agent remembered anything.
        tmp.write_text("")
    tmp.replace(target)


def _evaluate(trial: Path, store: Path, fixtures: Path, tokenizer: Path | None) -> None:
    args = ["--run-dir", str(trial), "--store", str(store), "--fixtures", str(fixtures)]
    for module in ("memory.t4.reader", "memory.t4.oracle"):
        cmd = module_command(module, *args)
        if module.endswith("oracle") and tokenizer is not None:
            cmd += ["--tokenizer", str(tokenizer)]
        with (trial / (module.rsplit(".", 1)[-1] + ".out")).open("w") as out, \
                (trial / (module.rsplit(".", 1)[-1] + ".err")).open("w") as err:
            # The reader keeps the real HOME: its proxy's upstream token command may need the login keychain.
            subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, check=True)


def identity(fixtures: Path, localmelo: Path, tokenizer: Path | None) -> dict[str, Any]:
    files = [fixtures / "facts.yaml", fixtures / "probes.yaml", fixtures / "drop_list.yaml",
             *sorted((fixtures / "days").glob("*.jsonl"))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    sha = subprocess.run(["git", "-C", str(localmelo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return {"protocol": "native-t4-v1", "reasoning": "compress",
            "budget": {"max_steps": 30, "timeout_s": 900, "thinking": "off"},
            "fixtures_sha256": digest.hexdigest(), "localmelo_sha": sha,
            "localmelo_repo": str(localmelo), "model": os.environ.get("BENCH_MODEL", "Qwen3.8-27B-oQ5-mtp"),
            "tokenizer": str(tokenizer) if tokenizer else None}


def run_trial(trial: Path, fixtures: Path = FIXTURES, tokenizer: Path | None = None) -> dict[str, Any]:
    trial, fixtures = trial.resolve(), fixtures.resolve()
    localmelo = Path(os.environ.get("BENCH_LOCALMELO_REPO", VENDOR)).expanduser().resolve()
    os.environ["BENCH_LOCALMELO_REPO"] = str(localmelo)
    fx = Fixtures(fixtures)
    config = identity(fixtures, localmelo, tokenizer)
    os.environ.setdefault("BENCH_MODEL", config["model"])
    checkpoint = trial / "t4" / "native.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["config"] != config:
            raise ValueError("Trial build, fixtures, model or tokenizer changed; use a fresh trial directory")
    else:
        if trial.exists() and any(trial.iterdir()):
            raise ValueError("Non-empty trial directory has no native checkpoint; use a fresh directory")
        state = {"config": config, "sessions": {}, "phases": {}}
        save(checkpoint, state)
    for name in ("mem", "home", "events"):
        (trial / name).mkdir(exist_ok=True)
    (trial / "events" / "pab.jsonl").touch(exist_ok=True)
    for day in range(1, fx.facts["timeline"]["days"] + 1):
        key = str(day)
        session = trial / "sessions" / f"day{day:02d}"
        run_dir = session / "run"
        if key in state["sessions"]:
            if state["sessions"][key]["state"] == "started":
                summary_path = run_dir / "summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
                state["sessions"][key] = session_result(day, run_dir, summary)
                save(checkpoint, state)
            continue
        workspace = session / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        cfg = session / "pab.json"
        save(cfg, mcp_config(trial, day, fixtures))
        date = dt.date.fromisoformat(fx.facts["timeline"]["start"]) + dt.timedelta(days=day - 1)
        state["sessions"][key] = {"day": day, "state": "started"}
        save(checkpoint, state)
        summary = _run_day(PROMPT.format(date=date.isoformat()), workspace, run_dir, cfg, trial, localmelo)
        state["sessions"][key] = session_result(day, run_dir, summary)
        save(checkpoint, state)

    t4 = trial / "t4"
    for flag, name in (("--export-store", "export.jsonl"), ("--export-decisions", "decisions.jsonl")):
        if not state["phases"].get(name):
            _export(localmelo, trial, flag, t4 / name)
            state["phases"][name] = True
            save(checkpoint, state)
    decisions = lines(t4 / "decisions.jsonl")
    for session in state["sessions"].values():
        task_id = session.get("task_id")
        session["admitted_by"] = dict(Counter(r.get("admitted_by") or "unspecified" for r in decisions
                                               if task_id and r.get("task_id") == task_id and routing.admitted(r)))
    save(t4 / "routing.json", routing.check(fx.facts, decisions))
    everything = t4 / "store-everything.jsonl"
    rebuild_everything(decisions, everything)
    comparator = trial / "comparators" / "store-everything"
    comparator.mkdir(parents=True, exist_ok=True)
    (comparator / "home").mkdir(exist_ok=True)
    if not (comparator / "events").is_symlink():
        (comparator / "events").symlink_to(trial / "events", target_is_directory=True)
    for name, run_dir, store in (("native", trial, t4 / "export.jsonl"),
                                  ("store-everything", comparator, everything)):
        if not state["phases"].get(name):
            _evaluate(run_dir, store, fixtures, tokenizer)
            state["phases"][name] = True
            save(checkpoint, state)
    state["complete"] = True
    save(checkpoint, state)
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--tokenizer", type=Path, default=Path(os.environ["BENCH_MODEL_PATH"]) / "tokenizer.json"
                    if os.environ.get("BENCH_MODEL_PATH") else None)
    args = ap.parse_args()
    state = run_trial(args.run_dir, args.fixtures, args.tokenizer)
    print(json.dumps({"complete": state["complete"], "sessions": len(state["sessions"]),
                      "run_dir": str(args.run_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
