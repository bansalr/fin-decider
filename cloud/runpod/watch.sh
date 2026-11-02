#!/usr/bin/env bash
# Mac-side watcher for the RunPod CLM run. Every INTERVAL seconds: pull new results into the repo's results
# folders, restart the pod's supervisor if it has died, report progress and credit, and, once the pod says
# ALL_DONE, do a final pull and tell the pod it may delete itself.
#   cloud/runpod/watch.sh            (runs until the pod is gone)
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
INTERVAL=${INTERVAL:-1800}
KEY=~/.ssh/fin_decider_runpod
STATE=cloud/runpod/state
id=$(cat $STATE/pod_id)
read -r ip port < $STATE/ssh
SSH=(ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$ip)
R=/workspace/fin-decider

pull() {
  for d in classification_results classification_results_native reports; do
    mkdir -p $d
    rsync -az -e "ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new" \
      --include='clm-*/***' --include='clm-*' --exclude='*' root@$ip:$R/$d/ $d/ 2>/dev/null
  done
  mkdir -p logs/runpod_clm
  rsync -az -e "ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new" root@$ip:/workspace/logs/ logs/runpod_clm/ 2>/dev/null
  rsync -az -e "ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new" root@$ip:/workspace/supervisor.log logs/runpod_clm/ 2>/dev/null
}

while true; do
  status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods/$id \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('desiredStatus') or d.get('error') or 'gone')" 2>/dev/null || echo unknown)
  balance=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'content-type: application/json' \
    -d '{"query":"{ myself { clientBalance currentSpendPerHr } }"}' \
    | python3 -c "import json,sys;m=json.load(sys.stdin)['data']['myself'];print(f\"\${m['clientBalance']:.2f} left, \${m['currentSpendPerHr']:.2f}/h\")" 2>/dev/null || echo "balance unknown")
  if [ "$status" != RUNNING ]; then echo "$(date) pod status: $status; $balance"; [ "$status" = gone ] && exit 0; sleep $INTERVAL; continue; fi
  pull
  "${SSH[@]}" "pgrep -f pod_supervisor.sh >/dev/null || (cd /workspace && nohup bash pod_supervisor.sh >> supervisor.log 2>&1 &)" 2>/dev/null
  done_list=$(ls logs/runpod_clm/*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/.done//' | tr '\n' ' ')
  echo "$(date) $balance | done: ${done_list:-none} | $(tail -1 logs/runpod_clm/supervisor.log 2>/dev/null)"
  if "${SSH[@]}" test -f /workspace/ALL_DONE 2>/dev/null; then
    pull; "${SSH[@]}" touch /workspace/PULLED; echo "$(date) ALL_DONE: final pull complete; the pod will now delete itself"; exit 0
  fi
  sleep $INTERVAL
done
