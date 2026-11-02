#!/usr/bin/env bash
# Launch Bloomberg shard pods s<FROM>..s<TO> of N and upload the prepared corpus to each (manifest last).
#   N=6 FROM=1 TO=5 cloud/runpod/launch_shards.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
KEY=~/.ssh/fin_decider_runpod
: "${N:?}" "${FROM:?}" "${TO:?}"
export GPUS=${GPUS:-NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA GeForce RTX 5090,NVIDIA H100 80GB HBM3,NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA RTX PRO 6000 Blackwell Server Edition,NVIDIA L40}
export CLOUD=${CLOUD:-SECURE}
for i in $(seq $FROM $TO); do
  NAME=s$i ONLY=bloomberg_full BLOOMBERG_SHARD=$i/$N cloud/runpod/launch.sh || { echo "launch s$i failed"; continue; }
  read -r _ id ip port <<< "$(grep "^s$i " cloud/runpod/state/pods)"
  ( e="ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new"
    for t in 1 2 3; do rsync -rt --no-o --no-g -e "$e" --exclude manifest.json corpus/ root@$ip:/workspace/corpus/ && \
      rsync -rt --no-o --no-g -e "$e" corpus/manifest.json root@$ip:/workspace/corpus/ && { echo "s$i corpus uploaded $(date -u)"; break; }; sleep 30; done ) &
done
wait
