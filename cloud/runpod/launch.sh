#!/usr/bin/env bash
# Create the RunPod GPU pod for the CLM queue and start pod_supervisor.sh on it. Run from the Mac.
#   cloud/runpod/launch.sh            # one NVIDIA L4 (default)
#   GPU="NVIDIA A100-SXM4-80GB" cloud/runpod/launch.sh
# Needs RUNPOD_API_KEY (and optionally HF_TOKEN) in fin-decider/.env, and the key ~/.ssh/fin_decider_runpod.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
: "${RUNPOD_API_KEY:?add RUNPOD_API_KEY to .env}"
GPU=${GPU:-NVIDIA L4}
MAX_HOURS=${MAX_HOURS:-120}
KEY=~/.ssh/fin_decider_runpod
STATE=cloud/runpod/state
mkdir -p $STATE
API=https://rest.runpod.io/v1
auth=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json")

if [ -f $STATE/pod_id ]; then echo "pod already recorded: $(cat $STATE/pod_id) (delete $STATE/pod_id to launch another)"; exit 1; fi

body=$(python3 - "$GPU" "$MAX_HOURS" <<'EOF'
import json, os, sys
gpu, max_hours = sys.argv[1], sys.argv[2]
env = {"PUBLIC_KEY": open(os.path.expanduser("~/.ssh/fin_decider_runpod.pub")).read().strip(),
       "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"], "MAX_HOURS": max_hours, "GRACE_HOURS": "12",
       "HF_HOME": "/workspace/hf"}
if os.environ.get("HF_TOKEN"):
    env["HF_TOKEN"] = os.environ["HF_TOKEN"]
print(json.dumps({"name": "fin-decider-clm", "gpuTypeIds": [gpu], "gpuCount": 1, "cloudType": "SECURE",
                  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                  "containerDiskInGb": 60, "volumeInGb": 150, "volumeMountPath": "/workspace",
                  "ports": ["22/tcp"], "env": env, "interruptible": False}))
EOF
)
resp=$(curl -s -X POST "${auth[@]}" -d "$body" $API/pods)
id=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)
[ -n "$id" ] || { echo "create failed: $(echo "$resp" | head -c 600)"; exit 1; }
echo "$id" > $STATE/pod_id; echo "pod $id created ($GPU); waiting for SSH..."

for i in $(seq 1 90); do
  info=$(curl -s "${auth[@]}" $API/pods/$id)
  read -r ip port <<< "$(echo "$info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('publicIp') or '', (d.get('portMappings') or {}).get('22') or '')")"
  [ -n "$ip" ] && [ -n "$port" ] && break
  sleep 10
done
[ -n "${ip:-}" ] || { echo "no public IP after 15 minutes; pod $id left running, check the RunPod console"; exit 1; }
echo "$ip $port" > $STATE/ssh
SSH=(ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 root@$ip)
for i in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
scp -i $KEY -P $port -o StrictHostKeyChecking=accept-new cloud/runpod/pod_supervisor.sh root@$ip:/workspace/
"${SSH[@]}" "cd /workspace && nohup bash pod_supervisor.sh >> /workspace/supervisor.log 2>&1 &"
echo "supervisor started on $ip:$port"
