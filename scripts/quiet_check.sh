#!/usr/bin/env bash
# Pre-flight for a timed segment: is the machine quiet enough for publishable numbers?
# Prints a provenance block (each raw file's header stores it) and exits 1 when not quiet.
# Usage: quiet_check.sh [--allow-pid PID]... [--settle SECONDS] [--max-gpu-gb GB] [--min-battery PCT] [--allow NAMES]
#   --allow-pid  exempts PID and all its descendants (server, engine workers, harness, an M8 load).
#   --allow      M8 only: accepts an interference load's own side effects by name (load, gpu, thermal,
#                lowpower; see quietlib.py) and prints them, so the raw header records them.
#   --settle     re-checks soft failures (load, busy processes, build tools, thermal) every 5 s
#                for up to SECONDS; lock, port 8500 and power failures fail at once.
# The checks live in quietlib.py, which run_framework.sh also uses to monitor whole runs.
exec /usr/bin/python3 -I "$(dirname "$0")/quietlib.py" check "$@"
