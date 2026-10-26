"""Per-task verdict matrix (verdict, wall seconds, model calls) for a run root. Usage: python3 -I scripts/task_matrix.py runs/<ts>"""

import json
import sys
from pathlib import Path

doc = json.loads((Path(sys.argv[1]) / "results.json").read_text())
harnesses = list(dict.fromkeys(r["harness"] for r in doc["records"]))
rows: dict[str, dict] = {}
for r in doc["records"]:
    if r["scored"]:
        rows.setdefault(r["task"], {})[r["harness"]] = r
print("| Task | " + " | ".join(harnesses) + " |")
print("|---|" + "---|" * len(harnesses))
for task, by in rows.items():
    cells = []
    for name in harnesses:
        r = by.get(name)
        cells.append("" if r is None else f"{r['verdict']} ({r['wall_s']:.0f} s, {r['n_llm']})")
    print(f"| {task} | " + " | ".join(cells) + " |")
