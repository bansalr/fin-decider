#!/usr/bin/env bash
# Mac-side watcher for the RunPod CLM pods listed in state/pods ("name id ip port"). Every INTERVAL seconds, per pod:
# pull new results into the repo's results folders and its logs into logs/runpod_clm/<name>/, restart its supervisor
# if it has died, and, once it says ALL_DONE, do a final pull and tell it it may delete itself. Logs credit each round.
#   cloud/runpod/watch.sh            (runs until every pod is gone)
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
INTERVAL=${INTERVAL:-1800}
MIN_BALANCE=${MIN_BALANCE:-6}  # below this, pull everything and STOP (not delete) all pods so volumes survive for a resume
KEY=~/.ssh/fin_decider_runpod
STATE=cloud/runpod/state
R=/workspace/fin-decider

pull() {  # name ip port
  local e="ssh -i $KEY -p $3 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
  for d in classification_results classification_results_native reports; do
    mkdir -p $d
    rsync -az -e "$e" --include='clm-*/***' --include='clm-*' --exclude='*' root@$2:$R/$d/ $d/ 2>/dev/null
  done
  mkdir -p logs/runpod_clm/$1
  rsync -az -e "$e" root@$2:/workspace/logs/ root@$2:/workspace/supervisor.log logs/runpod_clm/$1/ 2>/dev/null
}

while true; do
  balance=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'content-type: application/json' \
    -d '{"query":"{ myself { clientBalance currentSpendPerHr } }"}' \
    | python3 -c "import json,sys;m=json.load(sys.stdin)['data']['myself'];print(f\"\${m['clientBalance']:.2f} left, \${m['currentSpendPerHr']:.2f}/h\")" 2>/dev/null || echo "balance unknown")
  echo "$(date) $balance"
  bal=$(echo "$balance" | sed -nE 's/^\$([0-9.]+) left.*/\1/p')
  if [ -n "$bal" ] && python3 -c "import sys;sys.exit(0 if float('$bal') < $MIN_BALANCE else 1)"; then
    echo "$(date) balance below \$$MIN_BALANCE: final pull and stopping all pods"
    while read -r name id ip port <&3; do
      [ -f $STATE/finished_$name ] && continue
      pull $name $ip $port
      curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods/$id/stop >/dev/null && echo "  stopped $name ($id)"
    done 3< $STATE/pods
    exit 0
  fi
  alive=0
  while read -r name id ip port <&3; do
    [ -f $STATE/finished_$name ] && continue
    status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods/$id \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('desiredStatus') or d.get('error') or 'gone')" 2>/dev/null || echo unknown)
    if [ "$status" != RUNNING ]; then
      echo "  $name: pod status $status"; [ "$status" = gone ] || alive=$((alive+1)); continue; fi
    alive=$((alive+1))
    SSH=(ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$ip)
    pull $name $ip $port
    "${SSH[@]}" "pgrep -f pod_supervisor.sh >/dev/null || (cd /workspace && nohup bash pod_supervisor.sh >> supervisor.log 2>&1 &)" < /dev/null 2>/dev/null
    done_list=$(ls logs/runpod_clm/$name/*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/.done//' | tr '\n' ' ')
    echo "  $name: done ${done_list:-none} | $(tail -1 logs/runpod_clm/$name/supervisor.log 2>/dev/null)"
    if "${SSH[@]}" test -f /workspace/ALL_DONE < /dev/null 2>/dev/null; then
      pull $name $ip $port; "${SSH[@]}" touch /workspace/PULLED < /dev/null; touch $STATE/finished_$name
      echo "  $name: ALL_DONE, final pull complete; the pod will now delete itself"
    fi
  done 3< $STATE/pods
  [ $alive -eq 0 ] && { echo "$(date) no pods left"; exit 0; }
  sleep $INTERVAL
done
