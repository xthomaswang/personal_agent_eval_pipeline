# Shared helpers for gpu_lock.sh, serve.sh, wait_ready.sh, stop_server.sh, omlx_configure.sh
# and make_procfix.sh. Source it; do not run it.
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no mapfile, no ${x,,}.
# Empty arrays are expanded as ${a[@]+"${a[@]}"} so that `set -u` holds on bash < 4.4.

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BENCH_DIR/scripts"
LOG_DIR="$BENCH_DIR/logs"
MLSYS="${MLSYS_DIR:-$HOME/Developer/mlsys}"
SERVING="$MLSYS/serving"
HF_HOME="${HF_HOME:-$HOME/Developer/ml_source/huggingface}"
HF_HUB="${HF_HUB_CACHE:-$HF_HOME/hub}"
GPU_LOCK="${GPU_LOCK_PATH:-$MLSYS/experiments/.gpu-lock}"   # GPU_LOCK_PATH: tests only
LOCK_HOLDER="local_ai-benchmark session"
PEER_PORT=8500                       # the agent-benchmark session's oMLX
OUR_PORTS="8700 8701 8702 8703 8704 8705 8706"
FRAMEWORKS="mlx-lm llama.cpp omlx lithos-metal vllm-metal mlc-llm coreai"
PY="${BENCH_PYTHON:-/usr/bin/python3}"   # stdlib-only helpers; always run with -I

# Exit codes shared by the scripts (see scripts/README.md).
EX_USAGE=2 EX_NOT_READY=4 EX_REFUSED=5 EX_UNSUPPORTED=6

say() { printf '%s\n' "$*"; }
# A script's usage text: its leading comment block, without the shebang.
print_header() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$1"; }
warn() { printf '%s: %s\n' "${0##*/}" "$*" >&2; }
die() { local code=$1; shift; warn "$*"; exit "$code"; }

# Shell-quote one word so that the printed command can be pasted back unchanged.
q() {
  case "$1" in
    '') printf "''" ;;
    *[!A-Za-z0-9_./:=,@%+-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}
quote_cmd() { local out='' w; for w in "$@"; do out="$out${out:+ }$(q "$w")"; done; printf '%s' "$out"; }

fw_port() {
  case "$1" in
    mlx-lm) echo 8701 ;; llama.cpp) echo 8702 ;; omlx) echo 8703 ;;
    lithos-metal) echo 8704 ;; vllm-metal) echo 8705 ;; mlc-llm) echo 8700 ;; coreai) echo 8706 ;;
    *) return 1 ;;
  esac
}
port_fw() {
  case "$1" in
    8701) echo mlx-lm ;; 8702) echo llama.cpp ;; 8703) echo omlx ;;
    8704) echo lithos-metal ;; 8705) echo vllm-metal ;; 8700) echo mlc-llm ;; 8706) echo coreai ;;
    *) return 1 ;;
  esac
}
fw_srcdir() {
  case "$1" in
    mlx-lm) echo "$SERVING/mlx-lm" ;; llama.cpp) echo "$SERVING/llama.cpp" ;;
    omlx) echo "$SERVING/omlx_0.7" ;; lithos-metal) echo "$SERVING/lithos-metal" ;;
    vllm-metal) echo "$SERVING/vllm-metal" ;; mlc-llm) echo "$SERVING/mlc-llm_bench" ;;
    coreai) echo "$SERVING/coreai-models" ;; *) return 1 ;;
  esac
}
# The executable our serve.sh launches; its path appears in the server's command line.
fw_exe() {
  case "$1" in
    mlx-lm) echo "$SERVING/mlx-lm/.venv/bin/mlx_lm.server" ;;
    llama.cpp) echo "$SERVING/llama.cpp/build/bin/llama-server" ;;
    omlx) echo "$SERVING/omlx_0.7/.venv/bin/omlx" ;;
    lithos-metal) echo "$SERVING/lithos-metal/.venv/bin/lithos-metal" ;;
    vllm-metal) echo "$SERVING/vllm-metal/.venv/bin/vllm" ;;
    mlc-llm) echo "$SERVING/mlc-llm_bench/.venv/bin/python" ;;
    coreai) echo "$SERVING/coreai-models/.build/out/Products/Release/llm-server" ;; *) return 1 ;;
  esac
}
is_framework() { case " $FRAMEWORKS " in *" $1 "*) return 0 ;; esac; return 1; }
pidfile_of() { echo "$LOG_DIR/$1.pid"; }

# PIDs with a TCP LISTEN socket on a port (one per line; empty if none).
listen_pids() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fp 2>/dev/null | sed -n 's/^p//p' | sort -u || true; }
# Ports a PID listens on (one per line).
pid_ports() { lsof -nP -a -p "$1" -iTCP -sTCP:LISTEN -Fn 2>/dev/null | sed -n 's/^n.*:\([0-9][0-9]*\)$/\1/p' | sort -u || true; }
pid_cmd() { ps -ww -o command= -p "$1" 2>/dev/null || true; }
pid_alive() { kill -0 "$1" 2>/dev/null; }
# Describe the listeners on a port: "PID command" per line, or nothing.
describe_port() {
  local p
  for p in $(listen_pids "$1"); do printf '%s %s\n' "$p" "$(pid_cmd "$p" | cut -c1-160)"; done
}

# Seconds since a process started (from `ps -o etime=`, format [[dd-]hh:]mm:ss).
pid_age_s() {
  local e a b c d=0 h=0 m=0 s=0
  e=$(ps -o etime= -p "$1" 2>/dev/null | tr -d ' ') || return 1
  [ -n "$e" ] || return 1
  case "$e" in *-*) d=${e%%-*}; e=${e#*-} ;; esac
  IFS=: read -r a b c <<<"$e"
  if [ -n "${c:-}" ]; then h=$a; m=$b; s=$c; else m=$a; s=$b; fi
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
file_mtime() { /usr/bin/stat -f %m "$1"; }
now_s() { /bin/date +%s; }
iso_time() { /bin/date -r "$1" '+%Y-%m-%dT%H:%M:%S%z'; }
iso_to_s() { /bin/date -j -f '%Y-%m-%dT%H:%M:%S%z' "$1" +%s 2>/dev/null; }

lock_exists() { [ -e "$GPU_LOCK" ] || [ -L "$GPU_LOCK" ]; }
lock_is_ours() { lock_exists && [ "$(grep -m1 '^holder:' "$GPU_LOCK" 2>/dev/null || true)" = "holder: $LOCK_HOLDER" ]; }

# Our servers that are still up: "port pid" per line, for 8700-8706 listeners.
our_listeners() { local p x; for p in $OUR_PORTS; do for x in $(listen_pids "$p"); do echo "$p $x"; done; done; }

# Decide whether a PID from our pidfile really is the server serve.sh started.
# Sets OWN_REASON. Returns 0 if owned.
pid_owned() {
  local fw=$1 pid=$2 pidfile=$3 age mt now cmd port p
  OWN_REASON=''
  case "$pid" in ''|*[!0-9]*) OWN_REASON="not a PID: '$pid'"; return 1 ;; esac
  pid_alive "$pid" || { OWN_REASON="PID $pid is not running"; return 1; }
  for p in $(pid_ports "$pid"); do
    if [ "$p" = "$PEER_PORT" ] || { [ "$p" -ge 8600 ] && [ "$p" -le 8699 ]; }; then
      OWN_REASON="PID $pid listens on $p (agent-benchmark port)"; return 1
    fi
  done
  age=$(pid_age_s "$pid") || { OWN_REASON="cannot read the start time of PID $pid"; return 1; }
  mt=$(file_mtime "$pidfile"); now=$(now_s)
  # serve.sh writes the pidfile after its own process started, then execs the server in place,
  # so the real server started before the pidfile was written. A recycled PID started later.
  if [ $(( now - age )) -gt $(( mt + 5 )) ]; then
    OWN_REASON="PID $pid started after $pidfile was written (recycled PID)"; return 1
  fi
  cmd=$(pid_cmd "$pid"); port=$(fw_port "$fw")
  case "$cmd" in *"$(fw_exe "$fw")"*) return 0 ;; esac
  # Some servers retitle themselves (oMLX shows up as "omlx-server"). serve.sh starts the
  # server inside the framework checkout, so accept that cwd, or a listener on our port.
  [ "$(pid_cwd "$pid")" = "$(fw_srcdir "$fw")" ] && return 0
  for p in $(pid_ports "$pid"); do [ "$p" = "$port" ] && return 0; done
  OWN_REASON="PID $pid does not look like our $fw server: $cmd"; return 1
}
pid_cwd() { lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1 || true; }

# Resolve a HF cache snapshot and check its files (see hf_snapshot.py).
# Usage: check_snapshot <org/name> <download-hint> [--safetensors] [--file REL]...
# Prints the snapshot dir on stdout when refs/main resolves; status lines on stderr. Returns 0 or 4.
check_snapshot() {
  local repo=$1 hint=$2; shift 2
  "$PY" -I "$SCRIPTS_DIR/hf_snapshot.py" --hint "$hint" "$repo" "$@"
}

# "tag (sha)" plus a dirty marker, read without taking git's optional locks.
git_desc() {
  local d=$1 desc sha dirty
  [ -d "$d/.git" ] || [ -f "$d/.git" ] || { echo "(not a git checkout)"; return 0; }
  desc=$(GIT_OPTIONAL_LOCKS=0 git -C "$d" describe --tags --always 2>/dev/null || echo '?')
  sha=$(GIT_OPTIONAL_LOCKS=0 git -C "$d" rev-parse --short=8 HEAD 2>/dev/null || echo '?')
  dirty=$(GIT_OPTIONAL_LOCKS=0 git -C "$d" status --porcelain --untracked-files=no 2>/dev/null | awk '{print $2}' | tr '\n' ' ')
  if [ -n "$dirty" ]; then echo "$desc ($sha), modified: $dirty"; else echo "$desc ($sha), clean"; fi
}

# Installed versions from dist-info directory names (no interpreter start).
dist_versions() {
  local venv=$1 name d out='' v; shift
  for name in "$@"; do
    v=''
    for d in "$venv"/lib/python3.*/site-packages/"$name"-*.dist-info; do
      [ -d "$d" ] || continue
      v=${d##*/"$name"-}; v=${v%.dist-info}
    done
    out="$out${out:+ }$name=${v:-?}"
  done
  echo "$out"
}
