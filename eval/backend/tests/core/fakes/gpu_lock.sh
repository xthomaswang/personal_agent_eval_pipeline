#!/bin/bash
# Fake scripts/gpu_lock.sh: the real one's exit codes, on $GPU_LOCK_PATH only (never the real lock).
#   acquire <purpose> <eta-min>   exit 2 if held; writes holder/purpose/run lines (run from MODELEVAL_RUN_ID)
#   update-purpose <purpose> <eta-min>   exit 2 unless the lock is ours
#   release                       exit 2 if held by someone else
# Every call is appended to $FAKE_LOCK_LOG. FAKE_LOCK_ACQUIRE_RC forces acquire's exit code once.
set -u
LOCK=${GPU_LOCK_PATH:?tests must set GPU_LOCK_PATH}
echo "$* run=${MODELEVAL_RUN_ID:-}" >> "${FAKE_LOCK_LOG:-/dev/null}"
case "${1:-}" in
  acquire)
    if [ -n "${FAKE_LOCK_ACQUIRE_RC:-}" ] && [ -f "$FAKE_LOCK_ACQUIRE_RC" ]; then
      rc=$(cat "$FAKE_LOCK_ACQUIRE_RC"); rm -f "$FAKE_LOCK_ACQUIRE_RC"
      [ "$rc" = 3 ] && printf 'holder: local_ai-benchmark session\npurpose: %s\nrun: %s\n' "$2" "${MODELEVAL_RUN_ID:-}" > "$LOCK"
      exit "$rc"
    fi
    [ -e "$LOCK" ] && { echo "held"; exit 2; }
    printf 'holder: local_ai-benchmark session\npurpose: %s\neta: %s min\nrun: %s\n' "$2" "$3" "${MODELEVAL_RUN_ID:-}" > "$LOCK"
    echo "acquired"; exit 0 ;;
  update-purpose)
    grep -q "^run: ${MODELEVAL_RUN_ID:-}\$" "$LOCK" 2>/dev/null || exit 2
    printf 'holder: local_ai-benchmark session\npurpose: %s\neta: %s min\nrun: %s\n' "$2" "$3" "${MODELEVAL_RUN_ID:-}" > "$LOCK.tmp"
    mv -f "$LOCK.tmp" "$LOCK"; exit 0 ;;
  release)
    [ -e "$LOCK" ] || exit 0
    grep -q "^run: ${MODELEVAL_RUN_ID:-}\$" "$LOCK" || { echo "not ours"; exit 2; }
    rm -f "$LOCK"; echo "released"; exit 0 ;;
  status) cat "$LOCK" 2>/dev/null; exit 0 ;;
  *) exit 2 ;;
esac
