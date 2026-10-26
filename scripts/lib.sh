# Shared helpers for gpu_lock.sh (quiet_check.sh needs none: it runs quietlib.py directly).
# Source it; do not run it.
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no mapfile, no ${x,,}.

GPU_LOCK="${GPU_LOCK_PATH:-$HOME/Developer/mlsys/experiments/.gpu-lock}"   # GPU_LOCK_PATH: tests only
LOCK_HOLDER="local_ai-benchmark session"
PEER_PORT=8500                 # the agent-benchmark session's oMLX
PEER_RANGE="8600-8699"         # the agent benchmark's other ports
OUR_PORT=8703                  # the oMLX this pipeline launches (modeleval.paths.PORT)

# Exit codes shared by the scripts.
EX_USAGE=2 EX_REFUSED=5

say() { printf '%s\n' "$*"; }
# A script's usage text: its leading comment block, without the shebang.
print_header() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$1"; }
warn() { printf '%s: %s\n' "${0##*/}" "$*" >&2; }
die() { local code=$1; shift; warn "$*"; exit "$code"; }
# A path as printed: the home directory as ~ (the shell twin of modeleval.paths.tilde).
tilde() { case "$1" in "$HOME"|"$HOME"/*) printf '~%s' "${1#"$HOME"}" ;; *) printf '%s' "$1" ;; esac; }

# TCP listeners on a port spec in lsof syntax (8703, or 8500,8600-8699): "port pid" per line.
listeners() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fpn 2>/dev/null \
    | awk '/^p/ { pid = substr($0, 2) } /^n/ { n = $0; sub(/.*:/, "", n); print n, pid }' | sort -u || true
}
our_listeners() { listeners "$OUR_PORT"; }
pid_cmd() { ps -ww -o command= -p "$1" 2>/dev/null || true; }
# Describe the listeners of a port spec: "port pid command" per line, or nothing.
describe_listeners() {
  local port pid
  listeners "$1" | while read -r port pid; do printf '%s %s %s\n' "$port" "$pid" "$(pid_cmd "$pid" | cut -c1-160)"; done
}

now_s() { /bin/date +%s; }
iso_time() { /bin/date -r "$1" '+%Y-%m-%dT%H:%M:%S%z'; }
iso_to_s() { /bin/date -j -f '%Y-%m-%dT%H:%M:%S%z' "$1" +%s 2>/dev/null; }

lock_exists() { [ -e "$GPU_LOCK" ] || [ -L "$GPU_LOCK" ]; }
lock_is_ours() { lock_exists && [ "$(grep -m1 '^holder:' "$GPU_LOCK" 2>/dev/null || true)" = "holder: $LOCK_HOLDER" ]; }
# The lock's run id ("run: <id>" line), empty without one.
lock_run() { sed -n 's/^run: //p' "$GPU_LOCK" 2>/dev/null | head -1; }
# With MODELEVAL_RUN_ID set, the lock must carry that run id: a run never releases or rewrites
# a lock it did not take. Without it (a person at the terminal), any lock of this session goes.
run_matches() { [ -z "${MODELEVAL_RUN_ID:-}" ] || [ "$(lock_run)" = "$MODELEVAL_RUN_ID" ]; }
