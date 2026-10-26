"""Write call records to <run_dir>/calls.jsonl and bodies to <run_dir>/bodies/."""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path
from typing import Any

CALLS_FILE = "calls.jsonl"
BODIES_DIR = "bodies"


class CallRecorder:
    def __init__(self, run_dir: Path, log_bodies: bool) -> None:
        self.run_dir = run_dir
        self.calls_path = run_dir / CALLS_FILE
        self.bodies_dir = run_dir / BODIES_DIR
        self.log_bodies = log_bodies
        run_dir.mkdir(parents=True, exist_ok=True)
        if log_bodies:
            self.bodies_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._counter = itertools.count()

    def next_index(self) -> int:
        """Call indexes follow arrival order; records are written on completion,
        so concurrent calls can appear out of index order in the file."""
        return next(self._counter)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock, self.calls_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_body(self, call_index: int, name: str, ext: str, data: bytes) -> str:
        """Store raw bytes; return the path relative to the run directory."""
        filename = f"{call_index:06d}-{name}.{ext}"
        (self.bodies_dir / filename).write_bytes(data)
        return f"{BODIES_DIR}/{filename}"
