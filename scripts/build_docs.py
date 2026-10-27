"""Build the benchmark website from site/ and published docs/benchmark/**/summary.json.

    python3 scripts/build_docs.py           # generate all pages, assets, and data in docs/
    python3 scripts/build_docs.py --check   # exit 1 if any generated website file is out of date (for CI and pre-commit)

Standard library only. The format of summary.json is in docs/benchmark/README.md. The page is
deterministic: the same results give the same bytes, so --check can compare.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import hashlib
from pathlib import Path

from site_data import build_payload

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "docs" / "benchmark"
SOURCE = ROOT / "site"
OUTPUT = ROOT / "docs"
REPO_URL = "https://github.com/xthomaswang/personal_agent_eval_pipeline"   # a fork changes it here
SUITES = {"backend": "Backend", "agent_ability": "Agent ability", "memory": "Memory"}
REQUIRED = ("schema", "suite", "protocol", "title", "model", "family", "quant", "run_id", "date", "system", "metrics")
LEVELS = ("machine", "family", "quant", "suite", "run_id")


# ---- results -------------------------------------------------------------------------------------------

def well_typed(row: dict) -> bool:
    return (all(isinstance(row[k], str) for k in ("protocol", "title", "model", "family", "quant", "run_id", "date"))
            and isinstance(row["system"], dict) and isinstance(row.get("reference", False), bool)
            and isinstance(row.get("notes", ""), str) and isinstance(row.get("details", ""), str)
            and isinstance(row["metrics"], list)
            and all(isinstance(m, dict) and isinstance(m.get("name"), str) and isinstance(m.get("value"), str)
                    for m in row["metrics"]))


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        sys.exit(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")


def exists_exactly(rel: str) -> bool:
    """Whether a file exists with exactly this case. macOS volumes ignore case, GitHub links do not."""
    here = ROOT
    for part in rel.split("/"):
        if not here.is_dir() or part not in os.listdir(here):
            return False
        here = here / part
    return here.is_file()


def load() -> list[dict]:
    """Every row: the summary.json directly inside a <run-id>/ directory. A run's own files may hold
    more summary.json files deeper down; one that looks like a row, or any shallower one, is a path
    mistake."""
    for path in BENCH.rglob("summary.json"):
        depth = len(path.relative_to(BENCH).parts)
        misplaced = depth < len(LEVELS) + 1
        if depth > len(LEVELS) + 1:
            data = read_json(path)
            misplaced = isinstance(data, dict) and {"schema", "protocol", "suite"} <= data.keys()
        if misplaced:
            sys.exit(f"{path.relative_to(ROOT)}: a row belongs in <machine>/<family>/<quant>/<suite>/<run-id>/")
    rows = []
    for path in sorted(BENCH.glob("*/*/*/*/*/summary.json")):
        where = path.relative_to(ROOT)
        row = read_json(path)
        if not isinstance(row, dict):
            sys.exit(f"{where}: not a JSON object")
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            sys.exit(f"{where}: missing {', '.join(missing)}")
        if row["schema"] != 1 or not isinstance(row["suite"], str) or row["suite"] not in SUITES:
            sys.exit(f"{where}: schema must be 1 and suite one of {', '.join(SUITES)}")
        if not well_typed(row):
            sys.exit(f"{where}: a field has the wrong type (see docs/benchmark/README.md)")
        found = dict(zip(LEVELS, path.parent.relative_to(BENCH).parts))
        want = {**{k: row.get(k) for k in LEVELS}, "machine": row["system"].get("machine")}
        if found != want:
            sys.exit(f"{where}: the path does not match system.machine/family/quant/suite/run_id")
        row["_dir"] = path.parent.relative_to(ROOT).as_posix()
        if row.get("details"):
            target = posixpath.normpath(f"{row['_dir']}/{row['details']}")
            if not target.startswith("docs/benchmark/") or not exists_exactly(target):
                sys.exit(f"{where}: details {row['details']!r} is not a file under docs/benchmark")
            row["_details"] = target
        rows.append(row)
    return rows


def tables(rows: list[dict]) -> list[dict]:
    """One table per (machine, suite, protocol); every row of a table has the same title and metrics. Results
    from different machines never share a table: speed and timeouts depend on the machine."""
    out: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        machine = row["system"]["machine"]
        key = (machine, row["suite"], row["protocol"])
        names = [m["name"] for m in row["metrics"]]
        table = out.setdefault(key, {"machine": machine, "suite": row["suite"], "protocol": row["protocol"],
                                     "title": row["title"], "metrics": names, "rows": []})
        if row["title"] != table["title"] or names != table["metrics"]:
            sys.exit(f"{row['_dir']}: title or metric names differ from the other rows of protocol {row['protocol']}")
        table["rows"].append(row)
    for table in out.values():
        # The reference model's rows all carry reference: true; its newest run is the one compared with.
        refs = [r for r in table["rows"] if r.get("reference")]
        table["ref"] = max(refs, key=lambda r: (r["date"], r["run_id"]), default=None)
        table["rows"].sort(key=lambda r: (r is not table["ref"], r["family"], r["quant"], r["date"]))
    order = list(SUITES)
    return sorted(out.values(), key=lambda t: (t["machine"], order.index(t["suite"]), t["protocol"]))


PAGES = {
    "index": ("benchmark", "Personal Agent Bench"),
    "pipeline": ("pipeline", "Evaluation pipeline — Personal Agent Bench"),
    "methodology": ("methodology", "Methodology — Personal Agent Bench"),
    "reproduce": ("reproduce", "Reproduce — Personal Agent Bench"),
}


def output_files() -> dict[str, bytes]:
    """All managed outputs; no Git state, timestamps, or private run data enter the build."""
    rows = load()
    tables(rows)
    payload = build_payload(ROOT, rows, REPO_URL)
    data = "window.BENCHMARK_DATA = " + json.dumps(payload, indent=2, ensure_ascii=True) + ";\n"
    files = {
        "assets/site.css": (SOURCE / "styles.css").read_bytes(),
        "assets/app.js": (SOURCE / "app.js").read_bytes(),
        "assets/architecture.js": (SOURCE / "architecture.js").read_bytes(),
        "assets/data.js": data.encode("utf-8"),
        "assets/akita-avatar-chibi.png": (SOURCE / "assets/akita-avatar-chibi.png").read_bytes(),
    }
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.encode())
        digest.update(content)
    version = digest.hexdigest()[:12]
    template = (SOURCE / "template.html").read_text(encoding="utf-8")
    for name, (page, title) in PAGES.items():
        html = (template.replace("{{PAGE}}", page).replace("{{TITLE}}", title)
                .replace("{{REPO}}", REPO_URL).replace("{{VERSION}}", version))
        html = html.replace(f'data-nav="{page}"', f'data-nav="{page}" aria-current="page"')
        files[f"{name}.html"] = html.encode("utf-8")
    return files


MANAGED_DIRS = ("assets",)   # output directories the build owns entirely


def write_outputs(files: dict[str, bytes], output: Path, check: bool = False) -> list[str]:
    stale = []
    for name, content in files.items():
        path = output / name
        if path.is_file() and path.read_bytes() == content:
            continue
        stale.append(name)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    # A file left in a managed directory that the build no longer writes (a renamed or removed asset).
    for folder in MANAGED_DIRS:
        for path in sorted((output / folder).iterdir()) if (output / folder).is_dir() else []:
            name = f"{folder}/{path.name}"
            if path.is_file() and name not in files:
                stale.append(name)
                if not check:
                    path.unlink()
    return stale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="check every generated page, asset, and data file")
    ap.add_argument("--output", type=Path, default=OUTPUT, help="output directory; defaults to docs/")
    args = ap.parse_args()
    try:
        files = output_files()
    except (ValueError, OSError, KeyError) as exc:
        print(f"site build failed: {exc}", file=sys.stderr)
        return 1
    stale = write_outputs(files, args.output, args.check)
    if args.check:
        for name in stale:
            print(f"stale: {args.output / name} (run python3 scripts/build_docs.py)")
        return 1 if stale else 0
    print(f"{args.output}: {len(files)} files, {len(load())} published results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
