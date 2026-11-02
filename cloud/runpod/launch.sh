#!/usr/bin/env bash
# Create the RunPod GPU pod for the CLM queue and start pod_supervisor.sh on it. Run from the Mac.
#   cloud/runpod/launch.sh            # first available of GPUS (L40S preferred), community cloud
#   GPUS="NVIDIA L4" CLOUD=SECURE cloud/runpod/launch.sh
#   NAME=s3 ONLY=bloomberg_full BLOOMBERG_SHARD=3/8 cloud/runpod/launch.sh   # extra shard pod
# Every pod is recorded as "name id ip port" in state/pods (read by watch.sh and alarm.sh).
# Needs RUNPOD_API_KEY (and optionally HF_TOKEN) in fin-decider/.env, and the key ~/.ssh/fin_decider_runpod.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
: "${RUNPOD_API_KEY:?add RUNPOD_API_KEY to .env}"
GPUS=${GPUS:-NVIDIA L40S,NVIDIA GeForce RTX 5090,NVIDIA A100 80GB PCIe,NVIDIA L4,NVIDIA RTX PRO 4500 Blackwell}
CLOUD=${CLOUD:-COMMUNITY}
MAX_HOURS=${MAX_HOURS:-14}
KEY=~/.ssh/fin_decider_runpod
STATE=cloud/runpod/state
mkdir -p $STATE
API=https://rest.runpod.io/v1
auth=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json")

NAME=${NAME:-main}
if grep -q "^$NAME " $STATE/pods 2>/dev/null; then echo "pod $NAME already recorded in $STATE/pods"; exit 1; fi

body=$(python3 - "$GPUS" "$MAX_HOURS" "$CLOUD" <<'EOF'
import json, os, sys
gpus, max_hours, cloud = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
env = {"PUBLIC_KEY": open(os.path.expanduser("~/.ssh/fin_decider_runpod.pub")).read().strip(),
       "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"], "MAX_HOURS": max_hours, "GRACE_HOURS": os.environ.get("GRACE_HOURS", "1"),
       "HF_HOME": "/workspace/hf"}
for k in ("HF_TOKEN", "ONLY", "BLOOMBERG_SHARD"):
    if os.environ.get(k):
        env[k] = os.environ[k]
print(json.dumps({"name": "fin-decider-clm-" + os.environ.get("NAME", "main"), "gpuTypeIds": gpus, "gpuTypePriority": "custom", "gpuCount": 1, "cloudType": cloud, "supportPublicIp": True,
                  "imageName": "runpod/base:1.4.1-ubuntu2204",
                  "containerDiskInGb": 60, "volumeInGb": 150, "volumeMountPath": "/workspace",
                  "ports": ["22/tcp"], "env": env, "interruptible": False}))
EOF
)
resp=$(curl -s -X POST "${auth[@]}" -d "$body" $API/pods)
id=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)
[ -n "$id" ] || { echo "create failed: $(echo "$resp" | head -c 600)"; exit 1; }
echo "pod $NAME = $id created; waiting for SSH..."

for i in $(seq 1 90); do
  info=$(curl -s "${auth[@]}" $API/pods/$id)
  read -r ip port <<< "$(echo "$info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('publicIp') or '', (d.get('portMappings') or {}).get('22') or '')")"
  [ -n "$ip" ] && [ -n "$port" ] && break
  sleep 10
done
[ -n "${ip:-}" ] || { echo "no public IP after 15 minutes; deleting pod $id"; curl -s -X DELETE "${auth[@]}" $API/pods/$id; exit 1; }
echo "$NAME $id $ip $port" >> $STATE/pods
SSH=(ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 root@$ip)
for i in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
scp -i $KEY -P $port -o StrictHostKeyChecking=accept-new cloud/runpod/pod_supervisor.sh root@$ip:/workspace/
"${SSH[@]}" -n "cd /workspace && (setsid nohup bash pod_supervisor.sh >> /workspace/supervisor.log 2>&1 < /dev/null &)"  # detach fully or ssh never returns
echo "supervisor started on $ip:$port"
