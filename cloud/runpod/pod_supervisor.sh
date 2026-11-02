#!/usr/bin/env bash
# Runs ON the RunPod GPU pod, unattended. Installs everything, keeps the CLM servers alive, works through
# the dataset queue with --resume (restarting after any failure), and deletes the pod when it is finished.
#
# Safety:
#   - MAX_HOURS: hard limit; the pod deletes itself after this many hours whatever happens.
#   - When the queue is done it waits for /workspace/PULLED (created by the Mac after its final sync), or
#     GRACE_HOURS at most, then deletes itself. The account's prepaid credit is the last backstop.
# Usage (started by cloud/runpod/launch.sh): nohup bash pod_supervisor.sh > /workspace/supervisor.log 2>&1 &
set -uo pipefail
# Processes started over ssh do not inherit the pod env (API key, pod id, limits, HF_TOKEN); RunPod writes it here.
[ -f /etc/rp_environment ] && . /etc/rp_environment
[ -f /workspace/override.env ] && . /workspace/override.env  # set from the Mac to change limits/shard on a running pod
MAX_HOURS=${MAX_HOURS:-120}
GRACE_HOURS=${GRACE_HOURS:-12}
BRANCH=${BRANCH:-isaac_cwb}
W=/workspace
mkdir -p $W/logs
log() { echo "$(date -u +%FT%TZ) $*"; }
delete_pod() {
  log "deleting pod: $1"; sync
  curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
    || runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
}
export HF_HOME=$W/hf  # model downloads persist on the volume across restarts

# Hard time limit, independent of everything else.
( sleep $((MAX_HOURS * 3600)); delete_pod "MAX_HOURS=$MAX_HOURS reached" ) &

# --- install (idempotent) ---------------------------------------------------------------------------------
if [ ! -x $W/venv/bin/classify ]; then
  log "installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 $W/venv
  uv pip install --python $W/venv/bin/python vllm==0.32.0 contrastive-lm==0.1.0
  [ -d $W/fin-decider ] || git clone -q -b $BRANCH https://github.com/bansalr/fin-decider.git $W/fin-decider
  uv pip install --python $W/venv/bin/python -e $W/fin-decider
fi
export PATH=$W/venv/bin:$HOME/.local/bin:$PATH
cd $W/fin-decider
git pull -q --ff-only || true
git rev-parse HEAD > $W/logs/commit.txt; uv pip freeze --python $W/venv/bin/python > $W/logs/pip_freeze.txt 2>/dev/null; nvidia-smi > $W/logs/gpu.txt

# --- model servers ----------------------------------------------------------------------------------------
up() { curl -sf -o /dev/null "$1"; }
start_servers() {
  pkill -f "vllm serve" ; pkill -f clm-serve ; sleep 5
  log "starting encoder (vLLM)"
  nohup vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --served-model-name qwen3-8b \
    --runner pooling --max-model-len 2048 --dtype bfloat16 --port 8090 >> $W/logs/vllm.log 2>&1 &
  for i in $(seq 1 180); do up http://127.0.0.1:8090/health && break; sleep 10; done
  log "starting clm-serve"
  nohup clm-serve >> $W/logs/clm_server.log 2>&1 &
  for i in $(seq 1 60); do up http://127.0.0.1:8700/health && break; sleep 5; done
  up http://127.0.0.1:8700/health && log "servers up" || log "servers NOT up"
}
ensure_servers() {
  up http://127.0.0.1:8090/health && up http://127.0.0.1:8700/health && return
  if pgrep -f "vllm serve" >/dev/null; then  # still loading (e.g. after a supervisor restart): wait before restarting
    for i in $(seq 1 180); do up http://127.0.0.1:8090/health && break; sleep 10; done
    up http://127.0.0.1:8700/health || { pgrep -f clm-serve >/dev/null || nohup clm-serve >> $W/logs/clm_server.log 2>&1 & }
    for i in $(seq 1 60); do up http://127.0.0.1:8700/health && break; sleep 5; done
    up http://127.0.0.1:8090/health && up http://127.0.0.1:8700/health && { log "servers up"; return; }
  fi
  start_servers
}

# --- queue (name|config|extra flags) ----------------------------------------------------------------------
G="--set models.clm.enabled=true --set models.clm.quantization=none --set models.clm.concurrency=32"
QUEUE=(
  "reuters_v1|configs/native_reuters_top10_local.yaml|--no-dev-eval"
  "reuters_v2_1|configs/native_reuters_top10_v2_1_local.yaml|--no-dev-eval"
  "reuters_v3_1|configs/native_reuters_top10_v3_1_local.yaml|--no-dev-eval"
  "fomc|configs/bench_fomc.yaml|$G --no-dev-eval"
  "edgar|configs/bench_edgar.yaml|$G --no-dev-eval"
  "credit_agreements|configs/bench_credit_agreements.yaml|$G --no-dev-eval"
  "wiki_industry|configs/bench_wiki_industry.yaml|$G --no-dev-eval"
  "usnews|configs/bench_usnews.yaml|$G --no-dev-eval"
  "tradetheevent|configs/bench_tradetheevent.yaml|$G --no-dev-eval"
  "reuters_gsib|configs/bench_reuters.yaml|$G --no-dev-eval"
  "fincausal|configs/fincausal.yaml|$G --no-dev-eval"
  "bloomberg_full|configs/classification.yaml|$G${BLOOMBERG_SHARD:+ --shard $BLOOMBERG_SHARD}"
)
# ONLY=name[,name]: run just those queue entries (shard pods use ONLY=bloomberg_full BLOOMBERG_SHARD=i/N)
if [ -n "${ONLY:-}" ]; then
  keep=(); for item in "${QUEUE[@]}"; do [[ ",$ONLY," == *",${item%%|*},"* ]] && keep+=("$item"); done; QUEUE=("${keep[@]}")
fi
log "queue: ${QUEUE[*]%%|*} (shard ${BLOOMBERG_SHARD:-none}, max ${MAX_HOURS}h, grace ${GRACE_HOURS}h)"
model_flag() {  # pin the server's reported model name, or record that it reports none
  name=$(curl -s -X POST http://127.0.0.1:8700/v1/systemone -H 'content-type: application/json' \
    -d '{"state":"Bank lends money.","questions":{"q1":{"type":"noul","instructions":"Is this about lending?"}}}' \
    | python -c "import json,sys;print(json.load(sys.stdin).get('model') or '')" 2>/dev/null)
  [ -n "$name" ] && echo "--set models.clm.expected_model_returned=$name" || echo "--set models.clm.allow_returned_model_mismatch=true"
}

for pass in $(seq 1 50); do  # outer passes: anything that failed gets retried on the next pass
  remaining=0
  for item in "${QUEUE[@]}"; do
    IFS='|' read -r name cfg extra <<< "$item"
    [ -f $W/logs/$name.done ] && continue
    if [ "$name" = credit_agreements ] && [ -z "${HF_TOKEN:-}" ]; then log "skip $name (no HF_TOKEN)"; continue; fi
    remaining=$((remaining + 1))
    if [ "$name" = bloomberg_full ] && [ ! -e corpus ]; then
      # Use the Mac's prepared corpus (uploaded to $W/corpus, manifest last) so every shard shares one corpus hash.
      for i in $(seq 1 90); do [ -f $W/corpus/manifest.json ] && break; [ $i = 1 ] && log "waiting for uploaded corpus"; sleep 20; done
      [ -f $W/corpus/manifest.json ] && ln -sfn $W/corpus corpus && log "using uploaded corpus"
    fi
    dir=$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['corpus']['directory'])")
    if [ ! -f "$dir/manifest.json" ]; then
      log "prepare $name"; classify prepare -c $cfg >> $W/logs/$name.log 2>&1 || { log "prepare $name failed"; continue; }
    fi
    ensure_servers
    log "run $name"
    classify run -c $cfg -m clm --resume $extra $(model_flag) >> $W/logs/$name.log 2>&1
    if grep "session end" $W/logs/$name.log | tail -1 | grep -q "status COMPLETE"; then
      touch $W/logs/$name.done; log "done $name"
    else
      log "incomplete $name (will retry)"; sleep 30
    fi
  done
  [ $remaining -eq 0 ] && break
done

touch $W/ALL_DONE; log "queue finished; waiting for the Mac to confirm its final sync (max ${GRACE_HOURS}h)"
for i in $(seq 1 $((GRACE_HOURS * 60))); do [ -f $W/PULLED ] && break; sleep 60; done
delete_pod "finished (pulled=$([ -f $W/PULLED ] && echo yes || echo no))"
