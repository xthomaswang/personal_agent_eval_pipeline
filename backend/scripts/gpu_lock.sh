#!/usr/bin/env bash
# Shared GPU lock with the agent-benchmark session (DESIGN.md sections 5 and 6.7). The lock is
# ~/Developer/mlsys/experiments/.gpu-lock.
#
#   gpu_lock.sh acquire <purpose> <eta-minutes>         create the lock (exit 2 if held, 3 if port
#                                                       8500 is up)
#   gpu_lock.sh update-purpose <purpose> <eta-minutes>  rewrite purpose and ETA of our lock in place
#   gpu_lock.sh release [--force]                       delete it, only if we are the holder
#   gpu_lock.sh status                                  show the lock, its age/ETA, and the ports
#
# The holder line names the session ("holder: local_ai-benchmark session"), not an agent.
# With MODELEVAL_RUN_ID set, acquire adds a "run: <id>" line, and release and update-purpose
# refuse (exit 2) unless the lock carries that run id, so one run never drops another's lock.
# Without it, release deletes any lock of this session: only whoever acquired it should.
#
# acquire keeps the lock when port 8500 is up (exit 3): ask the agent-benchmark session to stop
# its oMLX, then start timing; listeners on 8600-8699 only get a note (quietlib's pre-unit check
# fails timed units on them, untimed work may run). release refuses while our oMLX still listens
# on 8703 (exit 5; stop it first, or pass --force). update-purpose (exit 2 unless the lock is
# ours) writes a temporary file next to the lock and renames it over the lock, so the lock
# never disappears while the peers read it; the run's QUIET-WINDOW-END uses it to announce
# "accuracy only (untimed)".
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

check_purpose_eta() {
  local purpose=$1 eta=$2
  [ -n "$purpose" ] && [ -n "$eta" ] || { usage >&2; exit "$EX_USAGE"; }
  case "$eta" in *[!0-9]*) die "$EX_USAGE" "eta-minutes must be a whole number of minutes, got '$eta'" ;; esac
  [ "$eta" -gt 0 ] || die "$EX_USAGE" "eta-minutes must be > 0"
  case "$purpose" in *$'\n'*|*$'\r'*) die "$EX_USAGE" "purpose must be one line" ;; esac
}

# Refuse (exit 2) when MODELEVAL_RUN_ID is set and the lock carries another run id, or none.
refuse_other_run() {
  run_matches && return 0
  say "GPU lock belongs to run '$(lock_run)', not '$MODELEVAL_RUN_ID'; not ${1} it:"
  show_lock
  exit 2
}

acquire() {
  local purpose=${1:-} eta=${2:-} run=${MODELEVAL_RUN_ID:-} now out
  check_purpose_eta "$purpose" "$eta"
  case "$run" in *[!A-Za-z0-9._:-]*) die "$EX_USAGE" "MODELEVAL_RUN_ID may only contain A-Z a-z 0-9 . _ : -, got '$run'" ;; esac
  now=$(now_s)
  # noclobber makes the redirection an O_EXCL create: exactly one side can win.
  if ! ( set -o noclobber
         { printf 'holder: %s\npurpose: %s\nstart: %s\neta: %s (%s min)\n' \
             "$LOCK_HOLDER" "$purpose" "$(iso_time "$now")" "$(iso_time $(( now + eta * 60 )))" "$eta"
           [ -z "$run" ] || printf 'run: %s\n' "$run"; } > "$GPU_LOCK" ) 2>/dev/null; then
    if lock_exists; then
      say "GPU lock is already held ($(tilde "$GPU_LOCK")):"
      show_lock
      exit 2
    fi
    die 1 "cannot create $(tilde "$GPU_LOCK")"
  fi
  say "acquired $(tilde "$GPU_LOCK"):"
  sed 's/^/  /' "$GPU_LOCK"
  out=$(listeners "$PEER_PORT")
  if [ -n "$out" ]; then
    say "port $PEER_PORT is up (port pid): $(echo $out)"
    say "ask the agent-benchmark session to stop its oMLX before timing"
    exit 3
  fi
  # The peer's other ports do not block a launch here: quietlib fails a timed unit on them.
  out=$(listeners "$PEER_RANGE")
  [ -z "$out" ] || say "note: agent-benchmark port(s) in $PEER_RANGE are listening (port pid): $(echo $out)"
  out=$(our_listeners)
  [ -z "$out" ] || say "note: port $OUR_PORT is already in use (port pid): $(echo $out)"
  exit 0
}

update_purpose() {
  local purpose=${1:-} eta=${2:-} now eta_line line tmp p_done=0 e_done=0
  check_purpose_eta "$purpose" "$eta"
  if ! lock_exists; then say "no GPU lock at $(tilde "$GPU_LOCK"); nothing to update"; exit 2; fi
  if ! lock_is_ours; then
    say "GPU lock is held by someone else; not updating it:"
    show_lock
    exit 2
  fi
  refuse_other_run updating
  now=$(now_s)
  eta_line="eta: $(iso_time $(( now + eta * 60 ))) ($eta min)"
  tmp=$(mktemp "$GPU_LOCK.tmp.XXXXXX") || die 1 "cannot create a temporary file next to $(tilde "$GPU_LOCK")"
  trap 'rm -f "$tmp"' EXIT
  # Same lines in the same order; only purpose and eta change (holder, start, run stay).
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      purpose:*) [ "$p_done" -eq 1 ] || printf 'purpose: %s\n' "$purpose"; p_done=1 ;;
      eta:*) [ "$e_done" -eq 1 ] || printf '%s\n' "$eta_line"; e_done=1 ;;
      *) printf '%s\n' "$line" ;;
    esac
  done < "$GPU_LOCK" > "$tmp"
  [ "$p_done" -eq 1 ] || printf 'purpose: %s\n' "$purpose" >> "$tmp"
  [ "$e_done" -eq 1 ] || printf '%s\n' "$eta_line" >> "$tmp"
  chmod 644 "$tmp"
  mv -f "$tmp" "$GPU_LOCK"   # rename(2): readers see the old file or the new one, never none
  trap - EXIT
  say "updated $(tilde "$GPU_LOCK"):"
  sed 's/^/  /' "$GPU_LOCK"
}

release() {
  local force=0 up
  case "${1:-}" in --force) force=1 ;; '') ;; *) usage >&2; exit "$EX_USAGE" ;; esac
  if ! lock_exists; then say "no GPU lock at $(tilde "$GPU_LOCK"); nothing to release"; exit 0; fi
  if ! lock_is_ours; then
    say "GPU lock is held by someone else; not releasing it:"
    show_lock
    exit 2
  fi
  refuse_other_run releasing
  up=$(our_listeners)
  if [ -n "$up" ] && [ "$force" -eq 0 ]; then
    say "our server is still listening (port pid): $(echo $up)"
    say "stop it first, or run: gpu_lock.sh release --force"
    exit "$EX_REFUSED"
  fi
  rm -f "$GPU_LOCK"
  say "released $(tilde "$GPU_LOCK")"
}

status() {
  local d
  if lock_exists; then
    if lock_is_ours; then say "GPU lock: held by us ($(tilde "$GPU_LOCK"))"; else say "GPU lock: held by the other side ($(tilde "$GPU_LOCK"))"; fi
    show_lock
  else
    say "GPU lock: free ($(tilde "$GPU_LOCK"))"
  fi
  d=$(describe_listeners "$PEER_PORT")
  if [ -n "$d" ]; then say "port $PEER_PORT (agent-benchmark oMLX): LISTEN"; say "$d" | sed 's/^/  /'; else say "port $PEER_PORT (agent-benchmark oMLX): free"; fi
  d=$(describe_listeners "$PEER_RANGE")
  if [ -n "$d" ]; then say "ports $PEER_RANGE (agent benchmark): LISTEN"; say "$d" | sed 's/^/  /'; else say "ports $PEER_RANGE (agent benchmark): none listening"; fi
  d=$(describe_listeners "$OUR_PORT")
  if [ -n "$d" ]; then say "port $OUR_PORT (oMLX under test): LISTEN"; say "$d" | sed 's/^/  /'; else say "port $OUR_PORT (oMLX under test): free"; fi
}

case "${1:-}" in
  acquire) shift; acquire "$@" ;;
  update-purpose) shift; update_purpose "$@" ;;
  release) shift; release "$@" ;;
  status) status ;;
  -h|--help) usage ;;
  *) usage >&2; exit "$EX_USAGE" ;;
esac
