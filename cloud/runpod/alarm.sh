#!/bin/bash
# Exits (waking the operator) on the first anomaly on any pod listed in state/pods (lines: name id ip port), else state/ssh.
# Anomalies: ssh unreachable 3x in a row; no write to any pod log for STALL_MIN; error/failed/NOT up/Traceback lines; ALL_DONE.
cd "$(dirname "$0")"
STALL_MIN=${STALL_MIN:-20}
declare -A miss seen
pods() { if [ -s state/pods ]; then cat state/pods; else read ip port < state/ssh; echo main x $ip $port; fi; }
while true; do
  while read name id ip port <&3; do
    out=$(ssh -i ~/.ssh/fin_decider_runpod -p $port -o ConnectTimeout=20 -o BatchMode=yes root@$ip '
      now=$(date +%s); newest=$(stat -c %Y /workspace/supervisor.log /workspace/logs/* /workspace/fin-decider/classification_results/*/*/* 2>/dev/null | sort -n | tail -1)
      echo AGE $(( (now-newest)/60 ))
      grep -nE "NOT up|failed|incomplete|Traceback|deleting pod|ALL_DONE|watchdog" /workspace/supervisor.log | tail -3' 2>/dev/null)
    if [ -z "$out" ]; then miss[$name]=$(( ${miss[$name]:-0}+1 ))
      [ ${miss[$name]} -ge 3 ] && { echo "ALARM $name unreachable 3x $(date -u)"; exit 1; }
      continue; fi
    miss[$name]=0
    age=$(echo "$out" | awk '/^AGE/{print $2}')
    [ "${age:-0}" -ge $STALL_MIN ] && { echo "ALARM $name no log writes for ${age} min $(date -u)"; exit 1; }
    bad=$(echo "$out" | grep -v '^AGE')
    if [ -n "$bad" ] && [ "$bad" != "$(cat state/alarm_seen_$name 2>/dev/null)" ]; then echo "$bad" > state/alarm_seen_$name; echo "ALARM $name: $bad $(date -u)"; exit 1; fi
  done 3< <(pods)
  sleep 120
done
