#!/usr/bin/env bash
# Shared GPU lock with the agent-benchmark session (METHODOLOGY section 3, resume.md
# "Shared with the other session"). The lock is ~/Developer/mlsys/experiments/.gpu-lock.
#
#   gpu_lock.sh acquire <purpose> <eta-minutes>   create the lock (exit 2 if held, 3 if port 8500 is up)
#   gpu_lock.sh release [--force]                 delete it, only if we are the holder
#   gpu_lock.sh status                            show the lock, its age/ETA, and the server ports
#
# acquire keeps the lock when port 8500 is up (exit 3): ask the agent-benchmark session to
# stop its oMLX, then start timing. release refuses while one of our servers still listens on
# 8700-8706 (stop it first, or pass --force). The holder line names the session, not an agent,
# so release deletes any lock of this session: only whoever acquired it should release it.
# GPU_LOCK_PATH overrides the path; tests must use it and never touch the real lock.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() { print_header "${BASH_SOURCE[0]}"; }

show_lock() {
  local start eta s e now
  sed 's/^/  /' "$GPU_LOCK" 2>/dev/null || true
  [ -s "$GPU_LOCK" ] || { say "  (empty file)"; return 0; }
  now=$(now_s)
  start=$(sed -n 's/^start: //p' "$GPU_LOCK" | head -1)
  eta=$(sed -n 's/^eta: //p' "$GPU_LOCK" | head -1)
  # Ours: ISO start and ETA. The other side's: "start: 2026-10-18 00:55:14 EDT", "eta: 15 min".
  s=$(iso_to_s "$start" || /bin/date -j -f '%Y-%m-%d %H:%M:%S %Z' "$start" +%s 2>/dev/null || true)
  [ -z "$s" ] || say "  age: $(( (now - s) / 60 )) min"
  e=$(iso_to_s "${eta%% *}" || true)
  case "$eta" in [0-9]*' min') [ -z "$s" ] || e=$(( s + ${eta%% *} * 60 )) ;; esac
  if [ -n "$e" ]; then
    if [ "$now" -le "$e" ]; then say "  ETA in $(( (e - now) / 60 )) min"; else say "  ETA passed $(( (now - e) / 60 )) min ago"; fi
  fi
}

acquire() {
  local purpose=${1:-} eta=${2:-} now out
  [ -n "$purpose" ] && [ -n "$eta" ] || { usage >&2; exit "$EX_USAGE"; }
  case "$eta" in ''|*[!0-9]*) die "$EX_USAGE" "eta-minutes must be a whole number of minutes, got '$eta'" ;; esac
  [ "$eta" -gt 0 ] || die "$EX_USAGE" "eta-minutes must be > 0"
  case "$purpose" in *$'\n'*) die "$EX_USAGE" "purpose must be one line" ;; esac
  now=$(now_s)
  # noclobber makes the redirection an O_EXCL create: exactly one side can win.
  if ! ( set -o noclobber
         printf 'holder: %s\npurpose: %s\nstart: %s\neta: %s (%s min)\n' \
           "$LOCK_HOLDER" "$purpose" "$(iso_time "$now")" "$(iso_time $(( now + eta * 60 )))" "$eta" \
           > "$GPU_LOCK" ) 2>/dev/null; then
    if lock_exists; then
      say "GPU lock is already held ($GPU_LOCK):"
      show_lock
      exit 2
    fi
    die 1 "cannot create $GPU_LOCK"
  fi
  say "acquired $GPU_LOCK:"
  sed 's/^/  /' "$GPU_LOCK"
  out=$(lsof -nP -iTCP:"$PEER_PORT" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$out" ]; then
    say "$out"
    say "port 8500 is up — ask the agent-benchmark session to stop it"
    exit 3
  fi
  out=$(our_listeners)
  [ -z "$out" ] || say "note: one of our ports is already in use (port pid): $(echo $out)"
  exit 0
}

release() {
  local force=0 up
  [ "${1:-}" = "--force" ] && force=1
  if ! lock_exists; then say "no GPU lock at $GPU_LOCK; nothing to release"; exit 0; fi
  if ! lock_is_ours; then
    say "GPU lock is held by someone else; not releasing it:"
    show_lock
    exit 2
  fi
  up=$(our_listeners)
  if [ -n "$up" ] && [ "$force" -eq 0 ]; then
    say "our server is still listening (port pid): $(echo $up)"
    say "stop it first (scripts/stop_server.sh <framework>), or run: gpu_lock.sh release --force"
    exit "$EX_REFUSED"
  fi
  rm -f "$GPU_LOCK"
  say "released $GPU_LOCK"
}

status() {
  local p d ports=''
  if lock_exists; then
    if lock_is_ours; then say "GPU lock: held by us ($GPU_LOCK)"; else say "GPU lock: held by the other side ($GPU_LOCK)"; fi
    show_lock
  else
    say "GPU lock: free ($GPU_LOCK)"
  fi
  d=$(describe_port "$PEER_PORT")
  if [ -n "$d" ]; then say "port $PEER_PORT (agent-benchmark oMLX): LISTEN"; say "$d" | sed 's/^/  /'; else say "port $PEER_PORT (agent-benchmark oMLX): free"; fi
  ports=$(lsof -nP -iTCP:8600-8699 -sTCP:LISTEN -Fn 2>/dev/null | sed -n 's/^n.*:\([0-9][0-9]*\)$/\1/p' | sort -un | tr '\n' ' ' || true)
  say "ports 8600-8699 (agent-benchmark): ${ports:-none listening}"
  for p in $OUR_PORTS; do
    d=$(describe_port "$p")
    if [ -n "$d" ]; then say "port $p ($(port_fw "$p")): LISTEN"; say "$d" | sed 's/^/  /'; fi
  done
  for d in $FRAMEWORKS; do
    p=$(pidfile_of "$d")
    [ -f "$p" ] || continue
    if pid_alive "$(head -1 "$p")"; then say "pidfile ${p#$BENCH_DIR/}: $(head -1 "$p") (running)"; else say "pidfile ${p#$BENCH_DIR/}: $(head -1 "$p") (not running; stale)"; fi
  done
}

case "${1:-}" in
  acquire) shift; acquire "$@" ;;
  release) shift; release "$@" ;;
  status) status ;;
  -h|--help) usage ;;
  *) usage >&2; exit "$EX_USAGE" ;;
esac
