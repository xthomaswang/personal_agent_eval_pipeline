"""Run a real-web task set (offline replica pages plus their original captures) through a served model and score
it with the set's own scorer.

    python3 real.py --set <set dir> --captures <captures dir> --models <id>[,<id>] --out <run dir>
        [--arms replica:vision,replica:vision+ocr,replica:ax,capture:vision] [--samples 3] [--pages cu_001,cu_002]
        [--kinds click,fill,read] [--gold <corrected manifest>] [--exclude <file>]
    python3 real.py --set <set dir> --captures <captures dir> --out <run dir> --ideal     # no model: checks the pipeline
    python3 real.py --set <set dir> --captures <captures dir> --out <run dir> --score     # replay and score again

A set dir holds manifest.jsonl, pages/<id>.html and scripts/score.py (the contract is in the set's
scripts/SCORING.md): each page follows window.benchmark.reset(mode)/getState(), click targets carry
data-item-index and fill controls data-field-label. Each page is rendered in WebKit at its manifest size, once per
item type (webrun.swift), and OCR'd. An arm is a source and a condition: `replica` shows the rendered replica,
`capture` the original capture (scored on reads only: the replica reproduces the capture's layout page-wide but
not inline, so a click on the capture cannot be replayed on the replica), and
the condition is one of run.py's (vision, vision+ocr, ax). Every item is asked `samples` times per model with
run.py's prompts and sampling; click and fill answers are then replayed on the replica page and the page's own
getState() is the evidence. Each (model, arm, sample) gets an evidence file and the set's score.py report, plus a
lenient read check (run.py's normalisation). Records are appended as they arrive and a rerun skips what is there.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import importlib.util
import json
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import run

HERE = Path(__file__).resolve().parent
# Every value is literal text a form could take; notes go in the keys. (The first runs of 2026-10-27 gave the name as
# "Mira Chen (legal and preferred name)", and some fill replies typed that note into Name fields.)
PROFILE = {
    "legal name": "Mira Chen",
    "preferred name": "Mira Chen",
    "email": "mira.chen@example.com",
    "phone": "(415) 555-0172",
    "address": "1200 Harbor Way, Apt 4B, San Francisco, CA 94107, United States",
    "LinkedIn": "linkedin.com/in/mira-chen-example",
    "education": "M.S. Computer Science, Northbridge University, graduated May 2022",
    "authorized to work in the US": "Yes",
    "needs visa sponsorship": "No",
    "heard about the job": "company website",
    "birth date (only if the form requires one)": "1997-03-14",
}
FILL_RULE = (" Use only the information in the profile and leave every field it does not cover untouched."
             " Do not attach a file, save or submit.")


def load_rows(manifest: Path, pages: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not pages or Path(r["file"]).stem in pages]


def stem(row: dict) -> str:
    return Path(row["file"]).stem


def webrun(job: dict, path: Path) -> None:
    path.write_text(json.dumps(job), encoding="utf-8")
    subprocess.run(["swift", str(HERE / "webrun.swift"), str(path)], check=True, cwd=HERE, timeout=900)


def chinese(row: dict) -> bool:
    """Whether a page needs Chinese OCR: marked zh, or its items ask about Chinese text."""
    text = json.dumps(row["items"], ensure_ascii=False)
    return row["language"] == "zh" or row.get("source_language") == "zh" or re.search(r"[\u3400-\u9fff]", text) is not None


def ocr(png: Path, zh: bool) -> list[dict]:
    langs = ["zh-Hans,en-US"] if zh else []   # Vision reads English only by default
    out = subprocess.run(["swift", str(HERE / "ocr.swift"), str(png), "0", *langs], check=True, cwd=HERE,
                         capture_output=True, text=True).stdout
    return json.loads(out.splitlines()[0])["lines"] if out.strip() else []


def prepare(set_dir: Path, rows: list[dict], out: Path, need_ocr: bool) -> None:
    """Render every page once per item type it has, and OCR the renders (skips what is there)."""
    rdir = out / "render"
    rdir.mkdir(parents=True, exist_ok=True)

    def one(row: dict) -> None:
        try:
            render(row)
        except Exception as e:   # one bad page costs its own items, not the run
            print(f"prepare: {row['file']}: {e!r}", file=sys.stderr, flush=True)

    def render(row: dict) -> None:
        modes = sorted({it["type"] for it in row["items"]})
        todo = [m for m in modes if not (rdir / f"{stem(row)}@{m}.elements.json").exists()]
        if todo:
            webrun({"page": str(set_dir / row["page"]), "width": row["screen"]["width_px"],
                    "height": row["screen"]["height_px"],
                    "renders": [{"mode": m, "out": str(rdir / f"{stem(row)}@{m}")} for m in todo]},
                   rdir / f"{stem(row)}.render-job.json")
        if need_ocr:
            for m in modes:
                f = rdir / f"{stem(row)}@{m}.ocr.json"
                if not f.exists():
                    f.write_text(json.dumps(ocr(rdir / f"{stem(row)}@{m}.png", chinese(row)), ensure_ascii=False),
                                 encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(one, rows))


def asset(rdir: Path, captures: Path, row: dict, mode: str, source: str) -> dict:
    el = json.loads((rdir / f"{stem(row)}@{mode}.elements.json").read_text(encoding="utf-8"))
    ocr_file = rdir / f"{stem(row)}@{mode}.ocr.json"
    png = rdir / f"{stem(row)}@{mode}.png" if source == "replica" else captures / row["file"]
    return {"png": base64.b64encode(png.read_bytes()).decode(), "elements": el,
            "ocr": json.loads(ocr_file.read_text(encoding="utf-8")) if ocr_file.exists() else [],
            "tasks": {"language": row["language"]}}


def as_item(it: dict) -> dict:
    """A manifest item in run.py's form (the prompt side only; gold stays in the manifest)."""
    if it["type"] == "click":
        return {"type": "click", "instruction": it["instruction"]}
    if it["type"] == "fill":
        return {"type": "fill", "instruction": it["instruction"] + FILL_RULE, "profile": PROFILE}
    return {"type": "read", "question": it["question"]}


def record_key(r: dict) -> tuple:
    return (r["model"], r["arm"], r["file"], r["item_index"], r["sample"])


def read_records(path: Path) -> list[dict]:
    """records.jsonl, skipping a line cut off by a kill."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            print(f"skipping an unreadable record line: {line[:80]!r}", file=sys.stderr)
    return out


def ask_all(args, rows: list[dict], out: Path) -> None:
    rdir = out / "render"
    path = out / "records.jsonl"
    done = {record_key(r) for r in read_records(path) if "error" not in r}   # failed asks are asked again
    failures = 0
    arms = [a.split(":") for a in args.arms.split(",")]
    models = args.models.split(",")
    kinds = args.kinds.split(",")
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            for source, cond in arms:
                cache: dict[str, dict] = {}
                for index, it in enumerate(row["items"], 1):
                    if it["type"] not in kinds:
                        continue
                    if it["type"] not in cache:
                        cache[it["type"]] = asset(rdir, args.captures, row, it["type"], source)
                    a = cache[it["type"]]
                    text, with_image = run.build(as_item(it), cond, args.coords, a)
                    for s in range(args.samples):
                        for model in (models if s % 2 == 0 else models[::-1]):
                            key = (model, f"{source}:{cond}", row["file"], index, s)
                            if key in done:
                                continue
                            for attempt in range(3):
                                try:
                                    r = run.ask(args.url, model, text, a["png"] if with_image else None,
                                                run.MAX_TOKENS[it["type"]], args.timeout)
                                    failures = 0
                                    break
                                except Exception as e:   # retried; recorded as an error if it keeps failing
                                    r = {"reply": "", "error": repr(e)[:300]}
                                    time.sleep(5 * (attempt + 1))
                            if "error" in r:
                                failures += 1
                                if failures >= 5:
                                    sys.exit(f"ask: {failures} requests in a row failed, last {r['error']}; stopping")
                            rec = {"model": model, "arm": f"{source}:{cond}", "file": row["file"], "item_index": index,
                                   "type": it["type"], "sample": s, "coords": args.coords, **r}
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fh.flush()
                            print(f'{model[:20]:20} {source}:{cond:10} {row["file"]} {index:2} {it["type"]:5} s{s} '
                                  f'{r["reply"][:50]!r}' + (f' ERROR {r["error"]}' if "error" in r else ""), flush=True)


def where(a: Any, cond: str, coords: str, el: dict) -> tuple[float, float] | None:
    """Where an answer acts, in page pixels and inside the viewport. An ax answer acts on the visible part of its
    element; a point outside the screen does not act at all."""
    w, h = el["width"], el["height"]
    if cond == "ax":
        b = a[0] if isinstance(a, list) and a else a
        e = b.get("element") if isinstance(b, dict) else None
        if type(e) is not int and not (isinstance(e, str) and e.strip().isdigit()):
            return None
        if not 0 <= int(e) < len(el["ax"]):
            return None
        x0, y0, x1, y1 = el["ax"][int(e)]["box"]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        return ((x0 + x1) / 2, (y0 + y1) / 2) if x0 <= x1 and y0 <= y1 else None
    pt = run.point(a, cond, coords, el)
    return pt if pt and 0 <= pt[0] < w and 0 <= pt[1] < h else None


def actions(rec: dict, el: dict) -> list[dict]:
    """The click and fill actions of a reply, in page pixels. A fill reply may be a list of actions, one action, or
    a list under any key ({"actions": [...]}, {"steps": [...]}); an unknown verb counts as type when it carries
    text, select when it carries an option, and click otherwise."""
    cond = rec["arm"].split(":")[1]
    ans, _ = run.parse_json(rec.get("reply", ""))
    if rec["type"] == "click":
        pt = where(ans, cond, rec["coords"], el) if ans is not None else None
        return [{"kind": "click", "x": pt[0], "y": pt[1]}] if pt else []
    if isinstance(ans, dict):
        lists = [v for v in ans.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
        ans = ans["actions"] if isinstance(ans.get("actions"), list) else lists[0] if len(lists) == 1 else [ans]
    acts = []
    for a in ans if isinstance(ans, list) else []:
        if not isinstance(a, dict):
            continue
        pt = where(a, cond, rec["coords"], el)
        verb = str(a.get("action", "")).strip().lower()
        kind = verb if verb in ("click", "type", "select") else (
            "select" if a.get("option") else "type" if a.get("text") or a.get("value") else "click")
        if pt:
            acts.append({"kind": kind, "x": pt[0], "y": pt[1], "text": str(a.get("text", a.get("value", ""))),
                         "option": str(a.get("option", a.get("text", "")))})
    return acts


def replay_all(set_dir: Path, rows: list[dict], out: Path, records: list[dict]) -> dict[str, dict]:
    """Replay every click and fill record on its replica page; returns {record id: replay result}."""
    rdir, pdir = out / "render", out / "replay"
    pdir.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list] = defaultdict(list)
    for i, r in enumerate(records):
        if r["type"] in ("click", "fill"):
            by_file[r["file"]].append((i, r))
    rows_by_file = {r["file"]: r for r in rows}

    def one(file: str) -> dict[str, dict]:
        try:
            return replay_page(file)
        except Exception as e:   # that page's records count as replay failures
            print(f"replay: {file}: {e!r}", file=sys.stderr, flush=True)
            return {}

    def replay_page(file: str) -> dict[str, dict]:
        row = rows_by_file[file]
        els = {m: json.loads((rdir / f"{stem(row)}@{m}.elements.json").read_text(encoding="utf-8"))
               for m in ("click", "fill") if (rdir / f"{stem(row)}@{m}.elements.json").exists()}
        jobs = [{"key": str(i), "mode": r["type"], "actions": actions(r, els[r["type"]])} for i, r in by_file[file]]
        res = pdir / f"{stem(row)}.jsonl"
        webrun({"page": str(set_dir / row["page"]), "width": row["screen"]["width_px"],
                "height": row["screen"]["height_px"], "replays": jobs, "replay_out": str(res)},
               pdir / f"{stem(row)}.job.json")
        return {(d := json.loads(line))["key"]: d for line in res.read_text(encoding="utf-8").splitlines() if line.strip()}

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for part in pool.map(one, sorted(by_file)):
            results.update(part)
    return results


def load_scorer(set_dir: Path):
    spec = importlib.util.spec_from_file_location("set_score", set_dir / "scripts" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QUOTES = {'"': '"', "'": "'", "\u201c": "\u201d", "\u300c": "\u300d"}
SEP = re.compile(r"\s*[,\uff0c\u3001;\uff1b\n]\s*")


def answer_text(reply: str) -> str:
    """A read reply as an answer: trimmed, without one pair of wrapping quotes (the ax list quotes every name)."""
    a = reply.strip()
    if len(a) >= 2 and QUOTES.get(a[0]) == a[-1]:
        a = a[1:-1].strip()
    return a


TYPOGRAPHY = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-", "\u2212": "-",
                            "\u2032": "'", "\u2019": "'", "\u2018": "'", "\u2033": '"', "\u201c": '"', "\u201d": '"',
                            "\u3010": "[", "\u3011": "]", "\uff3b": "[", "\uff3d": "]", "#": ""})


def plain(t: Any) -> str:
    """Text without typography that does not change its meaning: dashes, quotes and primes, CJK brackets, a "#",
    full-width forms, and spaces in CJK text."""
    t = unicodedata.normalize("NFKC", str(t).translate(TYPOGRAPHY))
    return re.sub(r"\s+", "", t) if re.search(r"[\u3400-\u9fff]", t) else t


def lenient_read(item: dict, answer: str) -> bool:
    """run.py's normalised read check on plain() text, plus lists written with other separators (、，; new lines,
    or spaces between CJK items)."""
    if run.score_read(dict(item, answer=plain(item["answer"])), plain(answer))["ok"]:
        return True
    gold = str(item["answer"])
    if item["match"] == "number" or ", " not in gold or re.search(r"\d,\d{3}", gold):
        return False
    want = [run.norm_text(plain(x)) for x in gold.split(",") if x.strip()]
    parts = SEP.split(answer.strip())
    if len(parts) == 1 and not re.search(r"[A-Za-z]", answer):
        parts = answer.split()   # CJK items separated by spaces
    got = [run.norm_text(plain(x)) for x in parts if x.strip()]
    return got == want if item["match"] == "text" else set(got) == set(want)


def score_all(set_dir: Path, rows: list[dict], out: Path, exclude: set[tuple[str, int]]) -> dict:
    files = {r["file"] for r in rows}
    latest: dict[tuple, dict] = {}
    for r in read_records(out / "records.jsonl"):   # the last record of a key wins, unless it is an error
        if r["file"] in files and ("error" not in r or "error" in latest.get(record_key(r), {"error": 1})):
            latest[record_key(r)] = r
    records = list(latest.values())
    replays = replay_all(set_dir, rows, out, records)
    scorer = load_scorer(set_dir)
    gold = {(r["file"], i): it for r in rows for i, it in enumerate(r["items"], 1)}
    meta = {r["file"]: r for r in rows}
    groups: dict[tuple, list] = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r["model"], r["arm"], r["sample"])].append((i, r))
    edir = out / "evidence"
    edir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {}
    for (model, arm, sample), recs in sorted(groups.items()):
        evidence, lenient, problems = [], {}, {"errors": 0, "replay_failed": 0}
        for i, r in recs:
            ev = {"file": r["file"], "item_index": r["item_index"], "type": r["type"]}
            problems["errors"] += "error" in r
            if r["type"] == "read":
                ev["answer"] = answer_text(r.get("reply", ""))
                lenient[(r["file"], r["item_index"])] = lenient_read(gold[(r["file"], r["item_index"])], ev["answer"])
            else:
                rp = replays.get(str(i), {})
                if "error" in rp or rp.get("after") is None:
                    problems["replay_failed"] += 1
                after, before = rp.get("after") or {}, rp.get("before") or {}
                ev["state"] = ({"clicks": after.get("clicks", [])} if r["type"] == "click"
                               else {"fields": after.get("fields", {}), "before_fields": before.get("fields", {})})
            evidence.append(ev)
        tag = f"{model}__{arm.replace(':', '-')}__s{sample}"
        (edir / f"{tag}.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evidence),
                                           encoding="utf-8")
        kinds = {r["type"] for _, r in recs}
        sub_rows = [dict(row, items=row["items"]) for row in rows]
        report = scorer.score(sub_rows, evidence)
        (edir / f"{tag}.score.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        cell = summary.setdefault(f"{model} | {arm}", {"samples": 0, "by_type": defaultdict(lambda: [0, 0]),
                                                      "by_category": defaultdict(lambda: [0, 0]),
                                                      "by_language": defaultdict(lambda: [0, 0]),
                                                      "read_lenient": [0, 0], "missing": 0, "excluded": 0,
                                                      "errors": 0, "replay_failed": 0})
        cell["samples"] += 1
        cell["errors"] += problems["errors"]
        cell["replay_failed"] += problems["replay_failed"]
        for res in report["items"]:
            key = (res["file"], res["item_index"])
            if res["type"] not in kinds:
                continue
            # A capture arm scores reads only: its clicks are replayed on the replica, whose layout matches the
            # capture page-wide but not inline (icons and fonts differ, so a tab can sit 50 px away).
            capture_action = arm.startswith("capture:") and res["type"] != "read"
            if key in exclude or res["verdict"] == "excluded" or capture_action:
                cell["excluded"] += 1
                continue
            cell["missing"] += res["reason"] == "missing_evidence"   # counted as failures, as the set's contract says
            ok = res["verdict"] == "pass"
            for k, v in (("by_type", res["type"]), ("by_category", res["category"]),
                         ("by_language", meta[res["file"]]["language"])):
                cell[k][v][0] += ok
                cell[k][v][1] += 1
            if res["type"] == "read":
                cell["read_lenient"][0] += lenient.get(key, False)
                cell["read_lenient"][1] += 1
    summary = json.loads(json.dumps(summary))
    (out / "summary.json").write_text(json.dumps({
        "excluded": sorted(map(list, exclude)),
        "notes": ["read answers: trimmed and one pair of wrapping quotes removed before scoring",
                  "read_lenient: run.py's normalised check, ignoring typography (dashes, quotes, primes, CJK brackets, "
                  "'#'), plus lists written with other separators",
                  "capture arms score reads only (their clicks and fills land on the replica, whose inline layout "
                  "differs from the capture)"],
        "results": summary}, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in summary.items():
        types = "  ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(v["by_type"].items()))
        extra = "".join(f"  {n} {v[n]}" for n in ("missing", "errors", "replay_failed") if v[n])
        print(f"{k:55} {types}  read-lenient {v['read_lenient'][0]}/{v['read_lenient'][1]}{extra}")
    return summary


def ideal_records(set_dir: Path, rows: list[dict], out: Path) -> None:
    """Records with ideal answers (target centres, gold values, gold reads) for the replica:vision arm."""
    rdir = out / "render"
    lines = []
    for row in rows:
        for index, it in enumerate(row["items"], 1):
            el = json.loads((rdir / f"{stem(row)}@{it['type']}.elements.json").read_text(encoding="utf-8"))
            w, h = el["width"], el["height"]
            centre = lambda b: {"x": round((b[0] + b[2]) / 2 * 1000 / w), "y": round((b[1] + b[3]) / 2 * 1000 / h)}
            if it["type"] == "click":
                boxes = [b["box"] for b in el["items"].get(str(index), []) if b["visible"]]
                reply = json.dumps(centre(boxes[0])) if boxes else "none"
            elif it["type"] == "fill":
                acts = []
                for f in it["fields"]:
                    boxes = [b for b in el["fields"].get(f["label"], []) if b["visible"]]
                    options = [b for b in el.get("groups", {}).get(f["label"], []) if b["visible"] and b["value"] == f["value"]]
                    if options:
                        acts.append({"action": "click", **centre(options[0]["box"])})
                    if not boxes:
                        continue
                    b = boxes[0]
                    if b["type"] in ("checkbox", "radio"):
                        if f["value"] is True or f["value"] not in (False, "", None):
                            acts.append({"action": "click", **centre(b["box"])})
                    elif b["tag"] == "select":
                        acts.append({"action": "select", **centre(b["box"]), "option": f["value"]})
                    else:
                        acts.append({"action": "type", **centre(b["box"]), "text": f["value"]})
                reply = json.dumps(acts)
            else:
                reply = it["answer"]
            lines.append({"model": "ideal", "arm": "replica:vision", "file": row["file"], "item_index": index,
                          "type": it["type"], "sample": 0, "coords": "norm1000", "reply": reply})
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8510/v1/chat/completions")
    ap.add_argument("--arms", default="replica:vision,replica:vision+ocr,replica:ax,capture:vision")
    ap.add_argument("--kinds", default="click,fill,read")
    ap.add_argument("--pages", default="")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--coords", default="norm1000", choices=("pixel", "norm1000"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--gold", type=Path, help="a corrected manifest to score against (prompts still come from the set's)")
    ap.add_argument("--exclude", type=Path, help="JSON list of [file, item_index] left out of the summary")
    ap.add_argument("--ideal", action="store_true")
    ap.add_argument("--score", action="store_true", help="replay and score the existing records only")
    args = ap.parse_args()
    args.set, args.captures, args.out = args.set.resolve(), args.captures.resolve(), args.out.resolve()
    pages = args.pages.split(",") if args.pages else []
    rows = load_rows(args.set / "manifest.jsonl", pages)
    gold_rows = load_rows(args.gold.resolve(), pages) if args.gold else rows
    if [r["file"] for r in gold_rows] != [r["file"] for r in rows]:
        ap.error("--gold must list the same files as the set's manifest")
    exclude = {tuple(x) for x in json.loads(args.exclude.read_text())} if args.exclude else set()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ideal:
        prepare(args.set, rows, args.out, need_ocr=False)
        ideal_records(args.set, gold_rows, args.out)
    elif not args.score:
        if not args.models:
            ap.error("--models is required")
        prepare(args.set, rows, args.out, need_ocr="vision+ocr" in args.arms)
        ask_all(args, rows, args.out)
    score_all(args.set, gold_rows, args.out, exclude)
    return 0


if __name__ == "__main__":
    sys.exit(main())
