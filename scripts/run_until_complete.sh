#!/usr/bin/env bash
# Run one model over the full corpus, resuming after crashes until its manifest
# says COMPLETE. Keeps the Mac awake while running (caffeinate).
#   scripts/run_until_complete.sh laya [extra classify args...]
set -u
model="$1"; shift
cd "$(dirname "$0")/.."
max_attempts=${MAX_ATTEMPTS:-30}
pause=${RETRY_PAUSE_S:-300}
for attempt in $(seq 1 "$max_attempts"); do
  echo "$(date -u +%FT%TZ) attempt $attempt/$max_attempts: classify run -m $model --resume $*"
  caffeinate -i -s .venv/bin/classify run -m "$model" --resume --progress-every 5000 "$@"
  code=$?
  status=$(.venv/bin/python - "$model" <<'PY'
import json, sys, glob
model = sys.argv[1]
done = []
for mf in glob.glob("classification_results/*/manifest.json"):
    m = json.load(open(mf))
    if m.get("scope", {}).get("kind") == "full_corpus" and model in m.get("models", {}) and not m.get("dry_run"):
        done.append(m["models"][model].get("status"))
print("COMPLETE" if "COMPLETE" in done else (done[0] if done else "NONE"))
PY
)
  echo "$(date -u +%FT%TZ) exit $code, manifest status $status"
  if [ "$status" = "COMPLETE" ]; then echo "done"; exit 0; fi
  if [ "$code" = "2" ]; then echo "fatal provider error (auth/identity); not retrying"; exit 2; fi
  sleep "$pause"
done
echo "gave up after $max_attempts attempts"; exit 1
