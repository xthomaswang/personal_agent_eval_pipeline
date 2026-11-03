#!/bin/zsh
# Validity check on the dev suite (MPS): base, closed-form post-maps, a read pass, and direction-removal edits.
# 1) time base at --dev-lme 20 and pick N, 2) base at N, 3) post-maps (free), 4) one acts pass,
# 5) edits from edits/example_remove_direction.py, 6) paired comparisons.
#
# Environment: WSP_ORIGINAL, WSP_LME (required), WSP_EDITS, WSP_DIR (default: the package dir, one level above this
# script). Booking the GPU is optional: set WSP_GPU_OWNER and a `gpu` booking tool, or just WSP_ALLOW_MPS=1.
set -u
WSP_DIR=${WSP_DIR:-${0:A:h:h}}
cd $WSP_DIR || exit 1
: ${WSP_ORIGINAL:?set WSP_ORIGINAL to the Qwen3-Embedding-0.6B folder}
: ${WSP_LME:?set WSP_LME to longmemeval_s_cleaned.json}
EDITS=${WSP_EDITS:-$WSP_DIR/models/edits}
W="uv run wsp"
L=work/lead
mkdir -p $L
log() { print -r -- "[$(date +%H:%M:%S)] $*" | tee -a $L/validity.log; }
NOTE="embedding weight-edit validity check (dev suite): base, closed-form post-maps, direction-removal edits on MPS"

# Optional booking step: only when a `gpu` tool exists and WSP_GPU_OWNER names the booking owner.
if [[ -n ${WSP_GPU_OWNER:-} ]] && command -v gpu >/dev/null; then
  until gpu 2>/dev/null | head -1 | grep -q "now: bench by $WSP_GPU_OWNER"; do
    gpu lock --as $WSP_GPU_OWNER --for 99m --kind bench "$NOTE" >/dev/null 2>&1
    gpu 2>/dev/null | head -1 | grep -q "now: bench by $WSP_GPU_OWNER" && break
    sleep 30
  done
  log "slot started: $(gpu | head -1)"
else
  export WSP_ALLOW_MPS=${WSP_ALLOW_MPS:-1}
fi

t0=$(date +%s)
$W eval --arm base --suite dev --dev-lme 20 >> $L/validity.log 2>&1 || { log "base lme20 failed"; exit 1; }
t20=$(( $(date +%s) - t0 ))
if (( t20 <= 150 )); then N=100; elif (( t20 <= 300 )); then N=40; else N=20; fi
log "base dev lme20 took ${t20}s -> N=$N"
print $N > $L/N
if (( N != 20 )); then
  $W eval --arm base --suite dev --dev-lme $N >> $L/validity.log 2>&1 || { log "base lme$N failed"; exit 1; }
  log "base dev lme$N done"
fi

# Post-maps on cached vectors (no model run).
for k in 1 2 4 8 16; do
  $W postmap fit --kind pca-remove --k $k --arm base --dev-lme $N --out state/postmaps/pca$k.lme$N.npz >> $L/validity.log 2>&1 &&
  $W eval --arm base --suite dev --dev-lme $N --post state/postmaps/pca$k.lme$N.npz --label base+pca$k >> $L/validity.log 2>&1
done
for s in 0.1 0.3 0.6; do
  $W postmap fit --kind whiten --shrink $s --arm base --dev-lme $N --out state/postmaps/wh$s.lme$N.npz >> $L/validity.log 2>&1 &&
  $W eval --arm base --suite dev --dev-lme $N --post state/postmaps/wh$s.lme$N.npz --label base+wh$s >> $L/validity.log 2>&1
done
log "post-maps done"
for lab in base+pca1 base+pca2 base+pca4 base+pca8 base+pca16 base+wh0.1 base+wh0.3 base+wh0.6; do
  print "== compare base $lab" >> $L/compare.log
  $W compare base $lab --suite dev --dev-lme $N >> $L/compare.log 2>&1
done

# One read pass, then edits: W' = W - a u (u^T W) on o_proj/down_proj of blocks >= layer.
$W acts --arm base --suite dev --layers 20,24,26,27,28 --limit 512 --dev-lme $N >> $L/validity.log 2>&1 || log "acts failed"
log "acts done"
typeset -a arms
arms=("e26c1a10:layer=26 a=1.0 center=1" "e24c1a05:layer=24 a=0.5 center=1" "e20c0a05:layer=20 a=0.5 center=0" "e28c1a10:layer=27 a=1.0 center=1")
for spec in $arms; do
  arm=v-${spec%%:*}; args=(${=spec#*:})
  [[ -d $EDITS/$arm ]] || $W copy $arm >> $L/validity.log 2>&1
  eargs=(); for x in $args; do eargs+=(--arg $x); done
  rep=(); [[ -f $EDITS/$arm/edit_manifest.json ]] && rep=(--replace)
  $W edit apply $arm edits/example_remove_direction.py $eargs $rep >> $L/validity.log 2>&1 || { log "edit $arm failed"; continue; }
  $W eval --arm $arm --suite dev --dev-lme $N >> $L/validity.log 2>&1 || { log "eval $arm failed"; continue; }
  print "== compare base $arm" >> $L/compare.log
  $W compare base $arm --suite dev --dev-lme $N >> $L/compare.log 2>&1
  log "arm $arm done"
done
$W verify-original >> $L/validity.log 2>&1
log "all done"
