#!/usr/bin/env bash
# Four detached GPU workers. Each GPU is assigned one rho and two backbones.
#
# GPU 4: CNN+NAF rho=0.00
# GPU 5: CNN+NAF rho=0.10
# GPU 6: CNN+NAF rho=0.25
# GPU 7: CNN+NAF rho=0.50
#
# Usage: bash scripts/launch_canonical_endpoint_matrix.sh smoke|pilot [sequential|parallel]
#
# Default sequential = 4 active training processes, one per GPU.
# Optional parallel = 8 active training processes, TWO per GPU;
# confirm sufficient free memory with nvidia-smi first.
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 smoke|pilot [sequential|parallel]" >&2
  exit 2
fi
MODE="$1"
PAIR_MODE="\${2:-sequential}"
case "$MODE" in
  smoke|pilot) ;;
  *) echo "Expected smoke or pilot, received: $MODE" >&2; exit 2 ;;
esac
case "$PAIR_MODE" in
  sequential|parallel) ;;
  *) echo "Expected sequential or parallel, received: $PAIR_MODE" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
OUT="../mscloudr/outputs"
mkdir -p "$OUT"

for SPEC in "4:0.00:0" "5:0.10:10" "6:0.25:25" "7:0.50:50"; do
  IFS=: read -r GPU RHO PERCENT <<< "$SPEC"
  LOG=$(printf '%s/CanonicalBridge_Endpoint%s_%s_GPU%s_%s.log' \
    "$OUT" "$PERCENT" "$MODE" "$GPU" "$PAIR_MODE")

  nohup env GPU="$GPU" RHO="$RHO" MODE="$MODE" PAIR_MODE="$PAIR_MODE" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
    bash -c '
      set -euo pipefail
      echo "GPU $GPU: rho=$RHO; mode=$MODE; pair_mode=$PAIR_MODE"
      if [[ "$PAIR_MODE" == "sequential" ]]; then
        for BACKBONE in cnn naf; do
          echo "GPU $GPU: starting $BACKBONE rho=$RHO"
          bash scripts/run_canonical_endpoint.sh "$BACKBONE" "$RHO" "$GPU" "$MODE"
        done
      else
        # Parallel sharing is opt-in: both models compete for GPU memory.
        bash scripts/run_canonical_endpoint.sh cnn "$RHO" "$GPU" "$MODE" &
        CNN_PID=$!
        bash scripts/run_canonical_endpoint.sh naf "$RHO" "$GPU" "$MODE" &
        NAF_PID=$!
        set +e
        wait "$CNN_PID"; CNN_STATUS=$?
        wait "$NAF_PID"; NAF_STATUS=$?
        set -e
        if (( CNN_STATUS != 0 || NAF_STATUS != 0 )); then
          echo "GPU $GPU: failure CNN=$CNN_STATUS NAF=$NAF_STATUS" >&2
          exit 1
        fi
      fi
      echo "ALL FINISHED GPU $GPU: both CNN and NAF rho=$RHO mode=$MODE"
    ' > "$LOG" 2>&1 < /dev/null &

  PID="$!"
  disown "$PID"
  printf 'Started GPU %s rho=%s (%s, %s): PID=%s log=%s\n' \
    "$GPU" "$RHO" "$MODE" "$PAIR_MODE" "$PID" "$LOG"
done

echo "Four detached GPU workers launched; eight jobs total."
echo "Do not relaunch with an existing mode/rho output directory."
echo "For full 50-epoch runs, select configurations using validation results."
