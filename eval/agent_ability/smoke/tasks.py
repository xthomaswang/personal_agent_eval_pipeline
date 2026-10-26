"""The task registry, workspace builders and prompts.

Prompts follow the recipe in the smoke-test report, the same bytes for every harness:
- Aider: introduction.md + instructions.md + instructions.append.md, aider's own addendum
  (benchmark/prompts.py: instructions_addendum), a line adapted from aider's retry prompt
  (test_failures), then the closing sentence.
- MCPMark: description.md + the note MCPMark's task manager appends
  (src/base/task_manager.py: _format_task_instruction), then the closing sentence.
- The MCP arm keeps only the path in the closing sentence: it has no shell.
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path

from smoke import data, paths

MCPMARK_TASKS = (
    "file_context/file_splitting",
    "file_context/pattern_matching",
    "file_context/uppercase",
    "file_property/largest_rename",
    "file_property/txt_merging",
    "folder_structure/structure_analysis",
    "legal_document/file_reorganize",
    "papers/papers_counting",
    "student_database/duplicate_name",
    "student_database/recommender_name",
)
# The five smallest reference solutions (.meta/example.py), skipping book-store (hard optimisation)
AIDER_CORE = ("transpose", "proverb", "pig-latin", "list-ops", "variable-length-quantity")
AIDER_EXTENDED = ("robot-name", "grade-school", "affine-cipher", "scale-generator", "wordy", "phone-number", "react")
# Exercises whose stub or reference solution misbehaves under the checker (see tests/test_aider_checks.py)
AIDER_EXCLUDED: dict[str, str] = {}

# Verbatim from aider's benchmark/prompts.py (instructions_addendum)
AIDER_ADDENDUM = """
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
"""
AIDER_TESTS_LINE = "Unit tests are in {tests}; the tests are correct, don't change them.\n"
# Verbatim from MCPMark's src/base/task_manager.py (_format_task_instruction)
MCPMARK_NOTE = "\n\nNote: Based on your understanding, solve the task all at once by yourself, don't ask for my opinions on anything."
CLOSING = "Work in {ws}. You can run shell commands and read and write files there.\n"
CLOSING_MCP = "Work in {ws}.\n"
WARMUP_TEXT = "Create a file named warmup.txt that contains the single word: ready\n"


@dataclass(frozen=True)
class Task:
    id: str
    kind: str  # "mcpmark", "aider" or "warmup"
    scored: bool = True

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__")

    @property
    def name(self) -> str:
        """The part after the kind: `file_property/largest_rename`, `transpose`."""
        return self.id.split("/", 1)[1] if "/" in self.id else self.id

    @property
    def category(self) -> str:
        return self.name.split("/", 1)[0]


WARMUP = Task("warmup", "warmup", scored=False)


def mcpmark_task(name: str) -> Task:
    return Task(f"mcpmark/{name}", "mcpmark")


def aider_task(name: str) -> Task:
    return Task(f"aider/{name}", "aider")


def default_suite(extended: bool = False, mcp_arm: bool = False) -> list[Task]:
    tasks = [mcpmark_task(n) for n in MCPMARK_TASKS]
    if not mcp_arm:
        tasks += [aider_task(n) for n in AIDER_CORE]
        if extended:
            tasks += [aider_task(n) for n in AIDER_EXTENDED]
    return tasks


def known_tasks() -> list[Task]:
    """Every task --task can select: the ten MCPMark tasks and every usable Python exercise."""
    tasks = [mcpmark_task(n) for n in MCPMARK_TASKS]
    tasks += [aider_task(n) for n in data.exercise_names() if n not in AIDER_EXCLUDED]
    return tasks


def select(patterns: list[str]) -> list[Task]:
    """Tasks whose id matches any fnmatch pattern, in registry order; every pattern must match."""
    known = known_tasks()
    chosen: list[Task] = []
    for pattern in patterns:
        matches = [t for t in known if fnmatch.fnmatchcase(t.id, pattern)]
        if not matches:
            raise ValueError(f"--task {pattern!r} matches no task; --list-tasks lists them")
        chosen += [t for t in matches if t not in chosen]
    return [t for t in known if t in chosen]


def build_workspace(task: Task, ws: Path) -> None:
    """A fresh workspace: the stub and its tests (Aider), or the unzipped start folder (MCPMark)."""
    if ws.exists():
        shutil.rmtree(ws)
    if task.kind == "aider":
        ex = data.exercise(task.name)
        ws.mkdir(parents=True)
        for rel in ex.workspace_files:
            data.copy_regular_file(ex.directory / rel, ws / rel)
    elif task.kind == "mcpmark":
        data.extract_category(data.verified_zip(task.category), task.category, ws)
    elif task.kind == "warmup":
        ws.mkdir(parents=True)
    else:
        raise ValueError(f"unknown task kind {task.kind!r}")


def prompt(task: Task, ws: Path, mcp_arm: bool = False) -> str:
    closing = (CLOSING_MCP if mcp_arm else CLOSING).format(ws=ws)
    if task.kind == "aider":
        ex = data.exercise(task.name)
        text = ex.instructions() + AIDER_ADDENDUM.format(file_list=" ".join(Path(f).name for f in ex.solution))
        text += "\n" + AIDER_TESTS_LINE.format(tests=", ".join(ex.test))
        return text + "\n" + closing
    if task.kind == "mcpmark":
        description = (paths.MCPMARK_EASY / task.name / "description.md").read_text()
        return description + MCPMARK_NOTE + "\n\n" + closing
    if task.kind == "warmup":
        return WARMUP_TEXT + "\n" + closing
    raise ValueError(f"unknown task kind {task.kind!r}")
