"""Frozen TM60/TM60b/TM68 pair comparison check. Run only under the pinned vendor Python.

No dataset text or model reply prose is printed or persisted. A completed check
exits 0 even when its statistical gate fails; setup/integrity errors exit 1.
STOP, NIGHT_DEADLINE and TERM/INT stop between pairs with exit 75.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from localmelo.melo.memory.retrieval.llm import send_from_provider
from localmelo.support.backends.registry import providers_for
from localmelo.support.config import Config
from localmelo.support.memory import hippo_encode, hippo_judge, hippo_route
from localmelo.support.memory.hippo_route import Entry
from localmelo.support.memory.index import ITEM_KINDS, MemoryItem
from localmelo.support.memory.note_judge import JUDGE_TIMEOUT_S
from localmelo.support.providers._http import switching_for

SET_SHA256 = "b55702119a1623f86c4c9efd3410628c3b9459d1efe671c8ec9109aadb091cc3"
VENDOR_SHA = "10420c74c33c469c39e43073f6f8e762713dd65b"
OLD_DATE = "2026-10-01"
NEW_DATE = "2026-11-01"
LABELS = ("same", "changes", "other")
CJK = re.compile(r"[㐀-䶿一-鿿]")
_HTTP_COUNTS: ContextVar[dict | None] = ContextVar("tm60_http_counts", default=None)
TM60B_SET_SHA256 = "4aa4d5f255e9fe987f13bbf4ee0b994f572445c07fdfd460a43811b5a9d73e9a"
TM60B_VENDOR_SHA = "4157a0d0cba2897b3e76a2f9ecefe2c133f48477"
FROZEN_COUNTS = {"same": 15, "changes": 20, "other": 25}
# TM68: a dated occurrence beside a schedule change. Its pairs carry a group, and group A two more labels.
TM68_SET_SHA256 = "19f97b673cfc2eafd9cd8e8a78294dc5d877217deefa90787ff078b0359a8f27"
TM68_VENDOR_SHA = "2e53cca4712ea2126b4be628c6e2bc87fdc34a10"
TM68_GROUPS = {"A": ("other", 20), "B": ("changes", 10), "C": ("changes", 10)}  # group: (label, frozen count)
TM68_OCCURRENCE = ("ahead", "past")  # group A: the old occurrence lies after or before the new item's date
TM68_READING = {"a_false_changes_max": 2, "c_changes_min": 8}  # fixed in the brief before any judge saw the set
GATE_LIMITS = {"false_same_max": 1, "false_changes_max": 2, "same_recall_min": 10, "changes_recall_min": 14}
# A gated profile pins its frozen set and build. TM60 reads all four gate lines in both phases; TM60b reads
# precision in both phases and recall after support only. "none" writes the tables without any gate line.
PROFILES = {
    "tm60": {"title": "TM60 pair check", "protocol": "tm60-pairs-v2", "set_sha256": SET_SHA256,
             "vendor_sha": VENDOR_SHA, "gated": True, "recall_phases": ("raw", "after support"), "overall": False},
    "tm60b": {"title": "TM60b pair check: held-out (gates)", "protocol": "tm60b-pairs-v1",
              "set_sha256": TM60B_SET_SHA256, "vendor_sha": TM60B_VENDOR_SHA, "gated": True,
              "recall_phases": ("after support",), "overall": True},
    "none": {"title": "Pair check: development (old set, no gate)", "protocol": "tm60b-pairs-v1",
             "set_sha256": None, "vendor_sha": None, "gated": False, "recall_phases": (), "overall": False},
    # TM68 runs exactly as TM60b (same protocol) and is read per group, by the reading fixed in its brief.
    "tm68": {"title": "TM68 pair check: a dated occurrence beside a schedule change", "protocol": "tm60b-pairs-v1",
             "set_sha256": TM68_SET_SHA256, "vendor_sha": TM68_VENDOR_SHA, "gated": False, "recall_phases": (),
             "overall": False, "groups": True},
}


def group_fields_ok(pair: dict) -> bool:
    """A TM68 pair: its group fixes its label; group A alone says whether the new item names the old occurrence and when it lies."""
    group = pair.get("group")
    if group not in TM68_GROUPS or pair.get("label") != TM68_GROUPS[group][0]:
        return False
    if group == "A":
        return (set(pair) == {"label", "old", "new", "group", "says_stands", "occurrence"}
                and isinstance(pair["says_stands"], bool) and pair["occurrence"] in TM68_OCCURRENCE)
    return set(pair) == {"label", "old", "new", "group"}


def load_pairs(path: Path, expected_sha: str = SET_SHA256, frozen_counts: dict | None = None, *, groups: bool = False) -> list[dict]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("pair set SHA256 mismatch")
    pairs = json.loads(payload)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair set must be a nonempty list")
    if groups and (not all(isinstance(p, dict) and group_fields_ok(p) for p in pairs)
                   or Counter(p["group"] for p in pairs) != {g: n for g, (_, n) in TM68_GROUPS.items()}):
        raise ValueError("group schema or frozen group counts mismatch")
    for pair in pairs:
        if groups:
            pair = {key: pair[key] for key in ("label", "old", "new")}
        if not isinstance(pair, dict) or set(pair) != {"label", "old", "new"} or pair["label"] not in LABELS:
            raise ValueError("pair schema mismatch")
        for side in ("old", "new"):
            item = pair[side]
            if (not isinstance(item, dict) or set(item) != {"kind", "text"}
                    or not isinstance(item["text"], str) or item["kind"] not in ITEM_KINDS):
                raise ValueError("item schema mismatch")
    if frozen_counts is not None and Counter(p["label"] for p in pairs) != frozen_counts:
        raise ValueError("frozen label counts changed")
    return pairs


def make_provider(base_url: str, model: str):
    # Same public-inference oMLX backend and no inference API key as job 59's
    # gateway config. Its admin-token helper is only for admin HTTP routes;
    # this check uses none. Do not load the person's config or keychain.
    if urlsplit(base_url).scheme != "http" or urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("a local HTTP model server is required")
    cfg = Config(chat_backend="vllm", chat_alias="omlx", embedding_backend="none", request_timeout_s=JUDGE_TIMEOUT_S)
    cfg.vllm.chat_url, cfg.vllm.chat_model = base_url, model
    provider, _ = providers_for(cfg)

    async def request_sent(request):
        counts = _HTTP_COUNTS.get()
        if counts is not None:
            counts["http_requests"] += 1
            if counts.pop("switching_pending", False):
                counts["switching_retries"] += 1

    async def response_received(response):
        counts = _HTTP_COUNTS.get()
        if counts is not None and response.status_code == 503:
            # The provider will read this non-streaming body anyway. Cache it
            # for its unchanged parser, and use its own switching classifier.
            await response.aread()
            if switching_for(response) is not None:
                counts["switching_responses"] += 1
                counts["switching_pending"] = True

    # Observational hooks only: leave the shipped retry/backoff/model-following
    # behavior alone. _quiet shares this HTTP client with the original provider.
    hooks = provider._http.event_hooks
    provider._http.event_hooks = {
        **hooks, "request": [*hooks.get("request", []), request_sent],
        "response": [*hooks.get("response", []), response_received],
    }
    return provider


async def evaluate_pair(pair: dict, pair_id: str, send) -> dict:
    old, new = pair["old"], pair["new"]
    item = MemoryItem(session_id="tm60", title="", text=new["text"], kind=new["kind"],
                      created_at=datetime.fromisoformat(NEW_DATE).timestamp())
    fields = {"id": 1, "text": old["text"], "kind": old["kind"],
              "created_at": datetime.fromisoformat(OLD_DATE).timestamp(), "sources": (), "vector": None,
              "provenance": {}, "subject": (), "tokens": frozenset(), "unit": None}
    if "folded" in {field.name for field in dataclasses.fields(Entry)}:
        # Later builds keep each item's normalised text on the entry, as the library's own builder does.
        # The judge request does not read it.
        fields["folded"] = hippo_route.normalise(old["text"])
    entry = Entry(**fields)
    said = [new["text"]]
    request = hippo_judge._request(item, (entry,), said)
    counts = {"http_requests": 0, "switching_retries": 0, "switching_responses": 0}
    token = _HTTP_COUNTS.set(counts)
    started = time.perf_counter()
    try:
        raw, why = await hippo_encode._ask(
            send, request, hippo_encode.Stats(), system_prompt=hippo_judge.SYSTEM_PROMPT,
            parse_reply=hippo_judge._reader(1), attempts=1, timeout_s=JUDGE_TIMEOUT_S,
        )
    finally:
        seconds = time.perf_counter() - started
        _HTTP_COUNTS.reset(token)
    counts.pop("switching_pending", None)
    failure = None
    if raw is None:
        if why.startswith("the model failed: "):
            failure = why.removeprefix("the model failed: ")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", failure):
                failure = "provider_error"
        else:
            failure = "output_limit" if "ran out of room" in why else "unreadable_reply"
    after = raw["verdict"] if raw else "other"
    support = None
    if after == "changes":
        support = hippo_judge.supported(new["text"], old["text"], said)
        if not support:
            after = "other"
    return {
        "pair_id": pair_id, "expected": pair["label"], "raw_verdict": raw,
        "verdict_after_support": after, "supported": support, "failure_type": failure, "seconds": seconds,
        **counts,
        "chinese": bool(CJK.search(old["text"]) or CJK.search(new["text"])),
        "cross_kind": old["kind"] != new["kind"],
        # TM68's labels of the pair (never text); absent in the other sets
        **{key: pair[key] for key in ("group", "says_stands", "occurrence") if key in pair},
    }


def gates(rows: list[dict], phase: str) -> dict:
    predicted = lambda r: (r["raw_verdict"] or {"verdict": "other"})["verdict"] if phase == "raw" else r["verdict_after_support"]
    counts = {
        "false_same": sum(r["expected"] != "same" and predicted(r) == "same" for r in rows),
        "false_changes": sum(r["expected"] != "changes" and predicted(r) == "changes" for r in rows),
        "same_recall": sum(r["expected"] == "same" and predicted(r) == "same" for r in rows),
        "changes_recall": sum(r["expected"] == "changes" and predicted(r) == "changes" for r in rows),
    }
    limits = {"false_same": 1, "false_changes": 2, "same_recall": 10, "changes_recall": 14}
    return {key: {"count": count, "limit": limits[key],
                  "pass": count <= limits[key] if key.startswith("false_") else count >= limits[key]}
            for key, count in counts.items()}


def _predicted(row: dict, phase: str) -> str:
    return (row["raw_verdict"] or {"verdict": "other"})["verdict"] if phase == "raw" else row["verdict_after_support"]


def group_reading(rows: list[dict], complete: bool) -> list[str]:
    """TM68's tables per group and its reading, as fixed in the brief before the run (counts only, no text)."""
    out = ["## Groups", "",
           ("A: the old item is one dated occurrence that still stands beside a later schedule change (expected `other`). "
            "B: that very occurrence moved or was cancelled (expected `changes`). C: a standing fact got a new value "
            "(expected `changes`, the controls)."), "",
           "| Group | Expected | Pairs | Phase | same | changes | other | Failed requests |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for group, (label, _) in TM68_GROUPS.items():
        selected = [r for r in rows if r.get("group") == group]
        for phase in ("raw", "after support"):
            counts = Counter(_predicted(r, phase) for r in selected)
            out.append(f"| {group} | {label} | {len(selected)} | {phase} | {counts['same']} | {counts['changes']} | "
                       f"{counts['other']} | {sum(bool(r['failure_type']) for r in selected)} |")
    a = [r for r in rows if r.get("group") == "A"]
    out += ["", "### Group A, split", "", "| Split | Pairs | false changes, raw | false changes, after support | false same, after support |",
            "|---|---:|---:|---:|---:|"]
    splits = [("the new item says the occurrence stands", [r for r in a if r.get("says_stands") is True]),
              ("the new item does not mention it", [r for r in a if r.get("says_stands") is False]),
              ("occurrence after 2026-11-01 (ahead)", [r for r in a if r.get("occurrence") == "ahead"]),
              ("occurrence before 2026-11-01 (past)", [r for r in a if r.get("occurrence") == "past"]),
              ("Chinese", [r for r in a if r["chinese"]])]
    for name, selected in splits:
        out.append(f"| {name} | {len(selected)} | {sum(_predicted(r, 'raw') == 'changes' for r in selected)} | "
                   f"{sum(_predicted(r, 'after support') == 'changes' for r in selected)} | "
                   f"{sum(_predicted(r, 'after support') == 'same' for r in selected)} |")
    a_false = sum(_predicted(r, "after support") == "changes" for r in a)
    c_right = sum(_predicted(r, "after support") == "changes" for r in rows if r.get("group") == "C")
    b_right = sum(_predicted(r, "after support") == "changes" for r in rows if r.get("group") == "B")
    false_same = [r["pair_id"] for r in rows if r["expected"] != "same"
                  and "same" in (_predicted(r, "raw"), _predicted(r, "after support"))]
    controls_ok = c_right >= TM68_READING["c_changes_min"]
    if not complete:
        primary = "PENDING (the run is not complete)"
    elif not controls_ok:
        primary = "NOT READ (the controls are below their line: the run or the set is off)"
    elif a_false <= TM68_READING["a_false_changes_max"]:
        primary = "judge-2 stays (the live case counts as a rare error)"
    else:
        primary = "a judge change comes before the next live run and the arms"
    out += ["", "### Reading (fixed before the run)", "",
            (f"- Controls: `changes` on group C after support {c_right}/{len([r for r in rows if r.get('group') == 'C'])} "
             f"(line: at least {TM68_READING['c_changes_min']}): {'met' if controls_ok else 'NOT met'}."),
            (f"- Primary: false `changes` on group A after support {a_false}/{len(a)} "
             f"(line: at most {TM68_READING['a_false_changes_max']}): **{primary}**."),
            f"- Group B (reported, not gated): `changes` after support {b_right}/{len([r for r in rows if r.get('group') == 'B'])}.",
            f"- False `same` anywhere (raw or after support): {', '.join(false_same) or 'none'}.",
            (f"- Group A pairs judged `changes` after support: "
             f"{', '.join(r['pair_id'] for r in a if _predicted(r, 'after support') == 'changes') or 'none'}; "
             f"raw only (support took them back): "
             f"{', '.join(r['pair_id'] for r in a if _predicted(r, 'raw') == 'changes' and _predicted(r, 'after support') != 'changes') or 'none'}."), ""]
    return out


def render_report(rows: list[dict], total: int, config: dict, stopped: str | None = None) -> str:
    profile = PROFILES[config.get("gate_profile", "tm60")]
    complete = len(rows) == total
    gate_note = ("Gate results on a partial prefix are provisional, never a landing decision." if profile["gated"]
                 else "Read per group, by the reading fixed before the run (see Groups)." if profile.get("groups")
                 else "No gate: this set is development data and its tables decide nothing.")
    out = [f"# {profile['title']}", "",
           (f"Status: {'complete' if complete else 'partial'}; {len(rows)}/{total} pairs; "
            f"stop reason: {stopped or 'none'}. {gate_note}"), "",
           f"Set SHA256: `{config['set_sha256']}`; vendor SHA: `{config['vendor_sha']}`.",
           (f"Old date: **{OLD_DATE}**; new date: **{NEW_DATE}** (local midnight, rendered by the library). "
            "These dates were fixed before the first model request."),
           (f"IDs are zero-based file positions p00–p{max(total - 1, 0):02d}. Chinese means either text contains "
            "U+3400–U+4DBF or U+4E00–U+9FFF; cross-kind means old.kind != new.kind. Texts are never copied into this report."),
           (f"Model: `{config['model']}`; base URL: `{config['base_url']}`. Requests are sequential, one library attempt, "
            f"{JUDGE_TIMEOUT_S:g} s timeout; no warmup, discovery, cache, embedding, gateway or retry loop in this runner."),
           ("Method A measures the provider as shipped: its internal switching retries remain enabled. HTTP request attempts, "
            "switching responses and actual follow-up requests after a switching response are counted per pair. "
            "Multi-request pairs are retained, never dropped or rerun. Request seconds include those waits/retries."),
           f"Prompt SHA256: `{config['prompt_sha256']}`; judge version: `{config['judge_version']}`.", ""]
    if profile.get("groups"):
        out += group_reading(rows, complete)
    for subset, selected in (("All", rows), ("Chinese", [r for r in rows if r["chinese"]]),
                             ("Cross-kind", [r for r in rows if r["cross_kind"]])):
        out += [f"## {subset}: {len(selected)} pairs", "",
                (f"HTTP requests: {sum(r['http_requests'] for r in selected)}; "
                 f"switching retries: {sum(r['switching_retries'] for r in selected)}; "
                 f"switching responses: {sum(r['switching_responses'] for r in selected)}. "
                 f"Pairs with >1 HTTP request: "
                 f"{', '.join(r['pair_id'] for r in selected if r['http_requests'] > 1) or 'none'}."), ""]
        for phase in ("raw", "after support"):
            out += [f"### {phase}", "", "| Expected \\ predicted | same | changes | other |", "|---|---:|---:|---:|"]
            for label in LABELS:
                counts = Counter(((r["raw_verdict"] or {"verdict": "other"})["verdict"] if phase == "raw"
                                  else r["verdict_after_support"]) for r in selected if r["expected"] == label)
                out.append(f"| {label} | {counts['same']} | {counts['changes']} | {counts['other']} |")
            out.append("")
    if profile["gated"]:
        out += ["## Frozen gates", "", "| Phase | Gate | Count | Limit | Result |", "|---|---|---:|---|---|"]
        verdicts = []
        for phase in ("raw", "after support"):
            values = gates(rows, phase)
            for key, value in values.items():
                limit = ("<= " if key.startswith("false_") else ">= ") + str(value["limit"])
                denominator = {"false_same": 45, "false_changes": 40, "same_recall": 15, "changes_recall": 20}[key]
                out.append(f"| {phase} | {key} | {value['count']}/{denominator} | {limit} | "
                           f"{'PASS' if value['pass'] else 'FAIL'} |")
            precision = all(values[k]["pass"] for k in ("false_same", "false_changes"))
            recall = all(values[k]["pass"] for k in ("same_recall", "changes_recall"))
            gated_recall = phase in profile["recall_phases"]
            verdicts.append((precision, recall or not gated_recall))
            status = "PENDING" if not complete else "PASS" if precision and (recall or not gated_recall) else "FAIL"
            recall_note = (f"recall {'passes' if recall else 'fails'}" if gated_recall
                           else f"recall ({'passes' if recall else 'fails'}) is not gated in this phase")
            out.append(f"\n{phase}: **{status}**. Precision {'passes' if precision else 'fails'}; {recall_note}. "
                       "Either precision failure blocks landing as built; "
                       "a recall failure means insufficient benefit, even if safe.\n")
        if profile["overall"]:
            precision = all(p for p, _ in verdicts)
            recall = all(r for _, r in verdicts)
            overall = ("PENDING" if not complete else "PASS" if precision and recall
                       else "FAIL (precision)" if not precision else "FAIL (recall only: safe, too little gain)")
            out.append(f"Overall: **{overall}**. Precision is read on raw and after support; "
                       "recall after support only.\n")
    failures = Counter(r["failure_type"] for r in rows if r["failure_type"])
    seconds = [r["seconds"] for r in rows]
    timing_summary = (
        f"Request seconds: n={len(seconds)}, total={sum(seconds):.6f}, mean={statistics.mean(seconds):.6f}, "
        f"median={statistics.median(seconds):.6f}, max={max(seconds):.6f}."
        if seconds else "Request seconds: n=0; mean, median and maximum not measured."
    )
    out += ["## Failures and request time", "",
            "Failures act as other in both confusion tables, and are retained, not retried or dropped.",
            f"Failure types/counts: `{json.dumps(dict(failures), sort_keys=True)}`.",
            f"Unsupported raw changes: {sum(r['supported'] is False for r in rows)}.",
            timing_summary,
            ("Elapsed seconds surround the full sequential library _ask call, including internal switching waits/retries, "
             "not setup or report/checkpoint writes; they are not divided by the number of HTTP attempts."), "",
            "| Pair | Expected | Raw | After support | Failure | HTTP requests | Switching retries | Switching responses | Seconds |",
            "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        mark = (f" ({row['group']}{', zh' if row['chinese'] else ''})" if "group" in row else "")
        out.append(f"| {row['pair_id']}{mark} | {row['expected']} | "
                   f"{row['raw_verdict']['verdict'] if row['raw_verdict'] else 'other (failed)'} | "
                   f"{row['verdict_after_support']} | {row['failure_type'] or '-'} | {row['http_requests']} | "
                   f"{row['switching_retries']} | {row['switching_responses']} | {row['seconds']:.6f} |")
    out += ["", "## Build environment", "", f"```text\n{config['sw_vers']}\n{config['xcodebuild_version']}\n```", "",
            (f"Python: {config['python']}. Raw verdict means the library reader's parsed verdict/item, not response prose. "
             "Only ID/label/boolean/numeric metadata is checkpointed. A completed run exits 0 regardless of gate outcome; "
             "exit 75 means resumable stop and exit 1 means setup/integrity failure."), ""]
    return "\n".join(out)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


async def run_pairs(pairs: list[dict], run_dir: Path, send, config: dict, *, deadline: float | None = None,
                    stop_requested=lambda: False, clock=time.time) -> int:
    run_dir = run_dir.resolve()
    config_path, rows_path = run_dir / "config.json", run_dir / "rows.jsonl"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("resume configuration changed")
    else:
        if run_dir.exists() and any(p.name != "STOP" for p in run_dir.iterdir()):
            raise ValueError("nonempty run directory has no config")
        atomic_write(config_path, json.dumps(config, indent=2) + "\n")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if not rows_path.exists():
        atomic_write(rows_path, "")
    if len(rows) > len(pairs):
        raise ValueError("too many checkpoint rows")
    for index, row in enumerate(rows):
        if row["pair_id"] != f"p{index:02d}" or row["expected"] != pairs[index]["label"]:
            raise ValueError("checkpoint is not an ordered prefix of this set")
    for index in range(len(rows), len(pairs)):
        reason = ("signal" if stop_requested() else "STOP" if (run_dir / "STOP").exists() else
                  "deadline" if deadline is not None and clock() >= deadline else None)
        if reason:
            atomic_write(run_dir / "report.md", render_report(rows, len(pairs), config, reason))
            print(f"tm60 stopped: {reason}; {len(rows)}/{len(pairs)} pairs done")
            return 75
        row = await evaluate_pair(pairs[index], f"p{index:02d}", send)
        rows.append(row)
        atomic_write(rows_path, "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows))
        atomic_write(run_dir / "report.md", render_report(rows, len(pairs), config))
        print(f"tm60 {row['pair_id']}: raw={row['raw_verdict']['verdict'] if row['raw_verdict'] else 'failed'} "
              f"after={row['verdict_after_support']} failure={row['failure_type'] or 'none'} seconds={row['seconds']:.6f}")
    atomic_write(run_dir / "report.md", render_report(rows, len(pairs), config))
    return 0


@contextmanager
def stop_signals():
    requested = False

    def receive(signum, frame):
        nonlocal requested
        requested = True

    previous = {sig: signal.signal(sig, receive) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield lambda: requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def provenance(base_url: str, model: str, gate: str, set_sha256: str, vendor_sha: str) -> dict:
    root = Path(hippo_judge.__file__).resolve().parents[2]

    def output(command):
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout.strip()

    sha = output(["git", "-C", str(root), "rev-parse", "HEAD"])
    if sha != vendor_sha or output(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("wrong or dirty vendor")
    profile = PROFILES[gate]
    config = {
        "protocol": profile["protocol"], "set_sha256": set_sha256, "vendor_sha": sha,
        "provider_switching_policy": "as_shipped_counted",
        "attempts": 1, "timeout_s": JUDGE_TIMEOUT_S,
        "gate": dict(GATE_LIMITS) if profile["gated"] else None,
        "model": model, "base_url": base_url, "old_date": OLD_DATE, "new_date": NEW_DATE,
        "chinese": "U+3400-U+4DBF or U+4E00-U+9FFF", "cross_kind": "old.kind != new.kind",
        "judge_version": hippo_judge.JUDGE_VERSION,
        "prompt_sha256": hashlib.sha256(hippo_judge.SYSTEM_PROMPT.encode()).hexdigest(),
        "sw_vers": output(["sw_vers"]), "xcodebuild_version": output(["xcodebuild", "-version"]),
        "python": sys.version.split()[0],
    }
    if gate != "tm60":  # TM60's config stays exactly as it was frozen
        config["gate_profile"] = gate
        config["recall_phases"] = list(profile["recall_phases"])
    if profile.get("groups"):
        config["reading"] = dict(TM68_READING)
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--set-sha256", required=True, help="SHA256 of the pair file, checked before parsing")
    parser.add_argument("--gate", choices=tuple(PROFILES), required=True)
    parser.add_argument("--vendor-sha", help="full build SHA; defaults to the gate's pinned build, required for none")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("NIGHT_URL", "http://127.0.0.1:8500/v1"))
    parser.add_argument("--model", default=os.environ.get("NIGHT_MODEL", "Qwen3.8-Flash-Next-IQ3XXS-splash"))
    args = parser.parse_args(argv)
    profile = PROFILES[args.gate]
    if not re.fullmatch(r"[0-9a-f]{64}", args.set_sha256):
        parser.error("--set-sha256 must be 64 lowercase hex digits")
    if profile["set_sha256"] is not None and args.set_sha256 != profile["set_sha256"]:
        parser.error(f"--set-sha256 is not the frozen {args.gate} set")
    if profile["vendor_sha"] is not None and args.vendor_sha not in (None, profile["vendor_sha"]):
        parser.error(f"--vendor-sha is not the pinned {args.gate} build")
    args.vendor_sha = args.vendor_sha or profile["vendor_sha"]
    if args.vendor_sha is None or not re.fullmatch(r"[0-9a-f]{40}", args.vendor_sha):
        parser.error("--gate none needs --vendor-sha (40 lowercase hex digits)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        frozen = FROZEN_COUNTS if PROFILES[args.gate]["gated"] else None
        # SHA before parsing, provenance, provider or request
        pairs = load_pairs(args.pairs, args.set_sha256, frozen, groups=bool(PROFILES[args.gate].get("groups")))
        deadline = float(os.environ["NIGHT_DEADLINE"]) if os.environ.get("NIGHT_DEADLINE") else None
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("nonfinite deadline")
        config = provenance(args.base_url, args.model, args.gate, args.set_sha256, args.vendor_sha)

        async def execute(requested):
            provider = make_provider(args.base_url, args.model)
            try:
                send = send_from_provider(hippo_encode._quiet(provider, "vllm"))
                return await run_pairs(pairs, args.run_dir, send, config, deadline=deadline, stop_requested=requested)
            finally:
                await provider.close()

        with stop_signals() as requested:
            return asyncio.run(execute(requested))
    except Exception as exc:  # noqa: BLE001 - any setup failure exits 1
        # API/config exceptions may carry the compared texts: type only.
        print(f"tm60 setup/integrity failure: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
