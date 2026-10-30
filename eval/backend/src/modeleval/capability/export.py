"""The only writer of capability files under a run's results directory <run> (DESIGN §8): for a verified run
docs/benchmark/<family>/<quant>/backend/<run-id>/ at the repository root.

A harness run directory holds transcripts, the agent homes, the task workspace, the prompt with its
absolute workspace path and per-run manifests. None of that may reach the public repo, and redaction
after the fact is not a safe filter, so this module copies an allowlist instead:

    <run>/capability/results.json                      rows, written by the runner
    <run>/capability/<harness>/<run-name>/calls.jsonl  counts and timings per call
                                     .../summary.json  bench.harness's summary
                                     .../check.txt     the checker's output

Of localmelo's run config, summary.json keeps only the keys in LOCALMELO_CONFIG_FIELDS, so memory_dir and
tools never reach the public repo.

Every string is rewritten so a path names its declared root, not this machine (paths.portable: ws/<task>,
eval/backend/..., models/...), and any other home path reads `~`; a calls.jsonl written with --log-bodies
is refused. Never exported: prompt.md (its sha256 is in the row), manifest.json, stdout/stderr,
claude-home/, hermes-home/, mem/, workspace/, check/, the env file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from modeleval import paths

RESULTS_NAME = "results.json"
RUN_FILES = ("calls.jsonl", "summary.json", "check.txt")
HARNESSES = ("claude_code", "hermes", "localmelo")
_RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# What a results row may carry (see runner.py for their meaning)
ROW_FIELDS = (
    "set", "harness", "task", "kind", "scored", "attempt", "run_dir", "launch",
    "verdict", "smoke_verdict", "valid", "solved", "solved_within_1200", "invalid_reasons", "anomalies",
    "wall_s", "model_s", "timed_out", "exit_code", "localmelo_status",
    "n_llm", "n_tool", "n_llm_aux", "n_llm_err", "native_steps", "turns",
    "check", "calls", "prompt_sha256", "t_start", "t_end", "battery_start", "template_errors", "error", "flags",
)  # fmt: skip
CHECK_FIELDS = (
    "kind", "passed", "returncode", "timed_out", "duration_s", "detail",
    "tests_modified", "modified_test_files", "extra_files", "missing_solution_files",
)  # fmt: skip

# The localmelo run config keys that summary.json keeps (run_record.config in bench/harness.py).
# memory_dir is a machine path and tools can name installed tool servers, so neither is kept.
LOCALMELO_CONFIG_FIELDS = (
    "max_steps", "attempt_policy", "steps_per_attempt", "max_attempts", "reflection_gate", "tool_exposure",
    "mode", "builtins", "memory", "timeout_s", "window_eviction", "fixed_prefix", "reflect_tool_guard",
    "cache_layout", "window_messages", "brief_items",
)  # fmt: skip


class ExportError(RuntimeError):
    """A file would leak something, or is not on the allowlist."""


def scrub_text(text: str, home: Path = paths.HOME) -> str:
    """Paths under a declared root relative to it (paths.portable); any other home path as `~` (the resolved
    spelling too, e.g. /private/... variants of the same home)."""
    text = paths.portable(text)
    for prefix in {str(home), os.path.realpath(home)}:
        text = text.replace(prefix + "/", "~/").replace(prefix, "~")
    return text


def scrub(obj: Any, home: Path = paths.HOME) -> Any:
    """Every string in a JSON-like value, keys included, through scrub_text."""
    if isinstance(obj, str):
        return scrub_text(obj, home)
    if isinstance(obj, dict):
        return {scrub_text(str(k), home): scrub(v, home) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, home) for v in obj]
    return obj


def _assert_clean(text: str, where: str, home: Path = paths.HOME) -> None:
    if str(home) in text or os.path.realpath(home) in text:
        raise ExportError(f"{where} still names the home directory after scrubbing")


def clean_row(record: dict[str, Any]) -> dict[str, Any]:
    """A runner record reduced to the allowlisted row fields, paths portable (scrub_text)."""
    row = {key: record.get(key) for key in ROW_FIELDS}
    check = record.get("check") or {}
    row["check"] = {key: check.get(key) for key in CHECK_FIELDS if key in check}
    return scrub(row)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_results(path: Path, doc: dict[str, Any]) -> Path:
    """results.json (the runner writes it after every task), scrubbed and checked."""
    if path.name != RESULTS_NAME:
        raise ExportError(f"{path.name} is not {RESULTS_NAME}")
    text = json.dumps(scrub(doc), indent=2, ensure_ascii=False) + "\n"
    _assert_clean(text, RESULTS_NAME)
    _write_atomic(path, text)
    return path


def _export_calls(src: Path, dest: Path) -> None:
    if (src.parent / "bodies").exists():
        raise ExportError(f"{paths.tilde(src.parent)} has bodies/: request and response bodies were logged")
    lines = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated last record of a killed proxy
        if isinstance(record, dict) and record.get("bodies"):
            raise ExportError(f"{paths.tilde(src)}: call {record.get('call_index')} logged bodies")
        lines.append(json.dumps(scrub(record), ensure_ascii=False, separators=(",", ":")))
    text = "\n".join(lines) + ("\n" if lines else "")
    _assert_clean(text, str(dest))
    _write_atomic(dest, text)


def _allowlist_run_config(summary: Any) -> Any:
    """summary with harness_report.run_record.config cut down to LOCALMELO_CONFIG_FIELDS.
    Each level must exist and be a dict; anything else is returned unchanged."""
    report = summary.get("harness_report") if isinstance(summary, dict) else None
    record = report.get("run_record") if isinstance(report, dict) else None
    config = record.get("config") if isinstance(record, dict) else None
    if isinstance(config, dict):
        record["config"] = {key: value for key, value in config.items() if key in LOCALMELO_CONFIG_FIELDS}
    return summary


def export_run(run_dir: Path, dest: Path) -> list[Path]:
    """Copy one harness run's allowlisted files from runs/ into results/; returns what was written."""
    written = []
    if (run_dir / "calls.jsonl").exists():
        _export_calls(run_dir / "calls.jsonl", dest / "calls.jsonl")
        written.append(dest / "calls.jsonl")
    if (run_dir / "summary.json").exists():
        summary = scrub(_allowlist_run_config(json.loads((run_dir / "summary.json").read_text())))
        text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        _assert_clean(text, str(dest / "summary.json"))
        _write_atomic(dest / "summary.json", text)
        written.append(dest / "summary.json")
    if (run_dir / "check.log").exists():
        text = scrub_text((run_dir / "check.log").read_text(encoding="utf-8", errors="replace"))
        _assert_clean(text, str(dest / "check.txt"))
        _write_atomic(dest / "check.txt", text)
        written.append(dest / "check.txt")
    return written


def export_capability(run_root: Path, results_json: Path) -> list[Path]:
    """Export every run results.json lists (rows, superseded attempts and warm-ups)."""
    doc = json.loads(results_json.read_text())
    written = []
    for row in [*doc.get("rows", []), *doc.get("superseded", []), *doc.get("warmups", [])]:
        rel = row["run_dir"]
        if not is_allowlisted(f"{rel}/calls.jsonl"):
            raise ExportError(f"unexpected run directory {rel!r}")
        written += export_run(run_root / rel, results_json.parent / rel)
    return written


def is_allowlisted(rel: str) -> bool:
    """Whether a path relative to a run's capability/ directory may exist there (redact.py)."""
    parts = Path(rel).parts
    if parts == (RESULTS_NAME,):
        return True
    return (
        len(parts) == 3
        and parts[0] in HARNESSES
        and bool(_RUN_NAME.match(parts[1]))
        and parts[1] not in (".", "..")
        and parts[2] in RUN_FILES
    )
