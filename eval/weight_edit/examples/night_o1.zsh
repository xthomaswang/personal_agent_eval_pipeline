#!/bin/zsh
# One budgeted editing run by an editing agent, then the held-out test suite for base and the chosen arm if the chosen
# arm beat base on dev. Run examples/validity.zsh first (it writes work/lead/N and the base dev results).
#
# Environment: WSP_ORIGINAL, WSP_LME, WSP_LOCOMO (required), WSP_EDITS, WSP_DIR (default: the package dir), and
# NIGHT_DEADLINE (epoch seconds; bounds the run, optional) or WSP_ALLOW_MPS=1 for the MPS gate.
#
# EDITOR_CMD is the hook that runs your editing agent: "run your editing agent with examples/editor_brief.md, a pass
# budget and a USD cap". It is run with WSP_AGENT_MODE=1, WSP_BUDGET_ACCOUNT=opus1, WSP_DEV_LME=$N, CL_WALL_MIN (the
# wall-clock limit in minutes) and the harness folder as the working directory. It must leave work/opus1/final.md,
# whose first "opus1-<name>" token names the chosen arm.
# Example: EDITOR_CMD='my_agent_runner --brief examples/editor_brief.md --model claude-opus-5-5 --max-usd 18'
set -u
WSP_DIR=${WSP_DIR:-${0:A:h:h}}
cd $WSP_DIR || exit 1
: ${WSP_ORIGINAL:?set WSP_ORIGINAL to the Qwen3-Embedding-0.6B folder}
: ${WSP_LME:?set WSP_LME to longmemeval_s_cleaned.json}
: ${WSP_LOCOMO:?set WSP_LOCOMO to locomo10.json}
: ${EDITOR_CMD:?set EDITOR_CMD to the command that runs your editing agent}
export WSP_ALLOW_MPS=${WSP_ALLOW_MPS:-1}
W="uv run wsp"
L=work/lead
mkdir -p $L work/opus1
log() { print -r -- "[$(date +%H:%M:%S)] $*" | tee -a $L/night_o1.log; }
N=$(cat $L/N 2>/dev/null || print 20)
S=".lme$N"; [[ $N == 100 ]] && S=""   # result file suffix (none at the default 100)
[[ -f results/base/dev$S.json ]] || { log "no base dev result at N=$N; running it"; $W eval --arm base --suite dev --dev-lme $N >> $L/night_o1.log 2>&1; }

# Editing run: shared budget account, agent mode, long Bash timeouts for passes.
if [[ ! -f work/opus1/final.md ]]; then
  [[ -f state/budget/opus1.json ]] || $W budget set opus1 15 >> $L/night_o1.log 2>&1
  left=$(( (${NIGHT_DEADLINE:-$(( $(date +%s) + 10800 ))} - $(date +%s)) / 60 - 40 ))
  (( left > 200 )) && left=200
  (( left < 30 )) && { log "only $left min left; not starting the editing run"; exit 75; }
  log "editing run starts: N=$N, wall limit ${left} min"
  WSP_AGENT_MODE=1 WSP_BUDGET_ACCOUNT=opus1 WSP_DEV_LME=$N BASH_DEFAULT_TIMEOUT_MS=1800000 BASH_MAX_TIMEOUT_MS=1800000 \
    CL_WALL_MIN=$left WORKDIR=$WSP_DIR \
    zsh -c "$EDITOR_CMD" >> $L/night_o1.log 2>&1
  log "editing run ended"
fi

# Held-out test suite, only if the chosen arm beat base on dev LongMemEval recall_any@5.
arm=$(grep -m1 -oE 'opus1-[A-Za-z0-9_.-]+' work/opus1/final.md 2>/dev/null)
if [[ -n $arm && -f results/$arm/dev$S.json ]]; then
  better=$(python3 -I - results/base/dev$S.json results/$arm/dev$S.json <<'EOF'
import json, sys
b, e = (json.load(open(p))["tasks"] for p in sys.argv[1:3])
lme = e["lme_dev"]["metrics"]["recall_any@5"] - b["lme_dev"]["metrics"]["recall_any@5"]
sf = e["scifact_train"]["metrics"]["ndcg@10"] - b["scifact_train"]["metrics"]["ndcg@10"]
print(1 if lme > 0 and sf > -0.005 else 0)
EOF
)
  log "chosen arm $arm, dev better: $better"
  if [[ $better == 1 ]]; then
    $W eval --arm base --suite test >> $L/night_o1.log 2>&1
    $W eval --arm $arm --suite test >> $L/night_o1.log 2>&1
    $W compare base $arm --suite test > $L/test_compare_$arm.log 2>&1
    log "test suite done: $L/test_compare_$arm.log"
  fi
else
  log "no chosen arm with a dev result; no test run"
fi
$W verify-original >> $L/night_o1.log 2>&1
log "night job done"
exit 0
