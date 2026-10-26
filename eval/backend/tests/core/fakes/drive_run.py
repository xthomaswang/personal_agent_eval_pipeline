"""Run the orchestrator in a separate process with the test sandbox (for the signal-cleanup test).

Usage: python -B drive_run.py <config.json>. The config names the patched `paths` attributes, the fake
speed module, the fake capability interpreter, the registry tables and the run options. Monkeypatching
does not cross a process boundary, so the same patches as the in-process tests are applied here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from modeleval import paths, quiet, registry, run


def main() -> int:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    for name, value in cfg["paths"].items():
        setattr(paths, name, value if name in ("PORT", "BASE_URL") else Path(value))
    quiet.MONITOR_INTERVAL = 0.2
    quiet.MARGIN = 0.3
    quiet.battery = lambda: (95, True)
    spec = importlib.util.spec_from_file_location("fake_speed", cfg["fake_speed"])
    fake = importlib.util.module_from_spec(spec)
    sys.modules["fake_speed"] = fake
    spec.loader.exec_module(fake)
    run.load_speed = lambda: fake
    run.load_allowed_env = lambda: (lambda env, extra: {**env, **extra})
    run.ensure_epoch = lambda: "test-epoch"
    run.CAP_PYTHON = Path(cfg["cap_python"])
    run.CAFFEINATE = False
    run.SSD_CACHE_BYTES = 0
    run.machine_info = lambda: {}
    models = registry.parse_models(cfg["models"])
    return run.run(run.Options(**cfg["options"]), models)


if __name__ == "__main__":
    sys.exit(main())
