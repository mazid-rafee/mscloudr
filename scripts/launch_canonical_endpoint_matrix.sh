#!/usr/bin/env bash
# Launch one detached GPU-6 CNN worker and one detached GPU-7 NAF worker.
# Each worker runs rho=0,0.10,0.25,0.50 sequentially.
# Usage: bash scripts/launch_canonical_endpoint_matrix.sh smoke|pilot
set -euo pipefail
if [[ "$#" != 1 || ( "$1" != "smoke" && "$1" != "pilot" ) ]]; then
  echo "Usage: $0 smoke|pilot" >&2
  exit 2
fi

MODE="$1"
cd "$(dirname "$0")/.."
OUT="../mscloudr/outputs"
mkdir -p "$OUT"

for BACKBONE in cnn naf; do
  if [[ "$BACKBONE" == "cnn" ]]; then GPU=6; else GPU=7; fi
  LOG=$(printf '%s/CanonicalBridge_%s_EndpointMatrix_%s.log' "$OUT" "$BACKBONE" "$MODE")
  nohup env BACKBONE="$BACKBONE" GPU="$GPU" MODE="$MODE" bash -c '
    set -euo pipefail
    for RHO in 0.00 0.10 0.25 0.50; do
      bash scripts/run_canonical_endpoint.sh "$BACKBONE" "$RHO" "$GPU" "$MODE"
    done
    echo "ALL FINISHED: $BACKBONE $MODE"
  ' > "$LOG" 2>&1 < /dev/null &
  PID="$!"
  disown "$PID"
  printf 'Started %-3s matrix (%s) on GPU %s; PID=%s; log=%s\n' \
    "$BACKBONE" "$MODE" "$GPU" "$PID" "$LOG"
done

echo "These are detached SSH-independent jobs (subject to cluster/job policies)."
echo "For full training, explicitly select validation-winning configurations."
