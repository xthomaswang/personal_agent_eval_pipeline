#!/usr/bin/env bash
# Pre-flight for a timed unit (DESIGN.md section 5.4): is the machine quiet enough to measure?
# Prints a provenance block and exits 1 when it is not quiet.
# Usage: quiet_check.sh [--allow-pid PID]... [--settle SECONDS] [--max-gpu-gb GB] [--min-battery PCT] [--allow NAMES]
#                       [--major-only]
#   --allow-pid  exempts PID and all its descendants (server, engine workers, orchestrator, harness).
#   --settle     re-checks soft failures (load, busy processes, build tools, thermal) every 5 s
#                for up to SECONDS; lock, foreign server ports (8500, 8600-8699) and power fail at once.
#   --max-gpu-gb before a launch: fails while more than GB of GPU memory is in use (a model resident).
#   --allow      accepts an interference load's own side effects by name (see quietlib.py).
#   --major-only the pre-unit check: fails only on major conditions; minor ones (a busy process,
#                foreign CPU below 300 %, build tools, the load) are printed and pass (DESIGN 5.4).
# The checks live in quietlib.py, which modeleval.quiet also uses to monitor the whole launch.
exec /usr/bin/python3 -I -B "$(dirname "$0")/quietlib.py" check "$@"
