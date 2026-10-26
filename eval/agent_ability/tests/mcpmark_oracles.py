"""Scripted solutions for the ten MCPMark filesystem-easy tasks: the positive control.

Each oracle works out its answer from the workspace, the way an agent would, and writes what the
task asks for. A verifier that passes the untouched workspace, or fails its oracle, means the
workspace layout or the verifier is wrong, which a negative control alone would not show (a
workspace rooted one level too high fails both ways). For structure_analysis, papers_counting and
file_reorganize the verifier hard-codes the answer; their oracles still derive it, so a pass also
confirms the workspace has the expected layout.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path


def file_splitting(ws: Path) -> None:
    text = (ws / "large_file.txt").read_text()
    size = -(-len(text) // 3)
    (ws / "split").mkdir()
    for i in range(3):
        (ws / "split" / f"split_{i + 1:02d}.txt").write_text(text[i * size : (i + 1) * size])


def pattern_matching(ws: Path) -> None:
    reference = (ws / "large_file.txt").read_text()
    matches = []
    for i in range(1, 21):
        name = f"file_{i:02d}.txt"
        text = (ws / name).read_text()
        # A shared substring of 30 or more characters exists iff some 30-character window is shared
        if any(text[j : j + 30] in reference for j in range(len(text) - 29)):
            matches.append(name)
    (ws / "answer.txt").write_text("".join(f"{m}\n" for m in matches))


def uppercase(ws: Path) -> None:
    (ws / "uppercase").mkdir()
    for i in range(1, 6):
        name = f"file_{i:02d}.txt"
        (ws / "uppercase" / name).write_text((ws / name).read_text().upper())


def largest_rename(ws: Path) -> None:
    largest = max(ws.glob("*.jpg"), key=lambda p: p.stat().st_size)
    largest.rename(ws / "largest.jpg")


def txt_merging(ws: Path) -> None:
    parts = [p.read_text() for p in sorted(ws.glob("*.txt"))]
    (ws / "merge.txt").write_text("\n".join(parts))


def structure_analysis(ws: Path) -> None:
    count = sum(1 for p in ws.rglob("*.py") if p.is_file())
    (ws / "structure_analysis.txt").write_text(f"{count}\n")


def file_reorganize(ws: Path) -> None:
    legal = ws / "legal_files"

    def version(path: Path) -> int:
        found = re.search(r"_v(\d+)\.txt$", path.name)
        return int(found.group(1)) if found else -1

    final = max(legal.glob("Preferred_Stock_Purchase_Agreement_v*.txt"), key=version)
    (legal / "final_version").mkdir()
    (legal / "final_version" / final.name).touch()


def papers_counting(ws: Path) -> None:
    (ws / "count.txt").write_text(f"{len(list(ws.glob('*.html')))}\n")


def duplicate_name(ws: Path) -> None:
    ids_by_name: dict[str, list[str]] = defaultdict(list)
    for folder in sorted(p for p in ws.iterdir() if p.is_dir()):
        student_id, _, name = folder.name.partition("_")
        ids_by_name[name.replace("_", " ")].append(student_id)
    name, ids = next((n, i) for n, i in sorted(ids_by_name.items()) if len(i) > 1)
    (ws / "namesake.txt").write_text(f"name: {name}\ncount: {len(ids)}\nids: {', '.join(ids)}\n")


def recommender_name(ws: Path) -> None:
    (folder,) = ws.glob("*_Patricia_Jones")
    letter = (folder / "recommendation_letter.txt").read_text().splitlines()
    signer = letter[letter.index("Sincerely,") + 1].strip()
    (ws / "recommender.txt").write_text(f"{signer}\n")


ORACLES: dict[str, Callable[[Path], None]] = {
    "mcpmark/file_context/file_splitting": file_splitting,
    "mcpmark/file_context/pattern_matching": pattern_matching,
    "mcpmark/file_context/uppercase": uppercase,
    "mcpmark/file_property/largest_rename": largest_rename,
    "mcpmark/file_property/txt_merging": txt_merging,
    "mcpmark/folder_structure/structure_analysis": structure_analysis,
    "mcpmark/legal_document/file_reorganize": file_reorganize,
    "mcpmark/papers/papers_counting": papers_counting,
    "mcpmark/student_database/duplicate_name": duplicate_name,
    "mcpmark/student_database/recommender_name": recommender_name,
}
