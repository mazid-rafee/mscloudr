#!/usr/bin/env bash
# Launch four GPU workers, each running two model variants at the same rho.
# Per-model logs are saved in ../mscloudr/outputs/<run-name>.log.
# Usage: bash scripts/launch_canonical_endpoint_matrix.sh smoke|pilot|full [sequential|parallel]
#
# Default sequential: 4 active training jobs, 1/GPU.
# Explicit parallel: 8 active training jobs, 2/GPU (requires enough VRAM).
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 smoke|pilot|full [sequential|parallel]" >&2
  exit 2
fi
MODE="$1"
PAIR_MODE="sequential"
if [[ "$#" -eq 2 ]]; then PAIR_MODE="$2"; fi
case "$MODE" in
  smoke|pilot|full) ;;
  *) echo "Expected smoke, pilot or full; got: $MODE" >&2; exit 2 ;;
esac
case "$PAIR_MODE" in
  sequential|parallel) ;;
  *) echo "Expected sequential or parallel; got: $PAIR_MODE" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
OUT="../mscloudr/outputs"
mkdir -p "$OUT"

for SPEC in "4:0.00:0" "5:0.10:10" "6:0.25:25" "7:0.50:50"; do
  IFS=: read -r GPU RHO PERCENT <<< "$SPEC"
  LOG=$(printf '%s/CanonicalBridge_Endpoint%s_%s_GPU%s_%s.log' \
    "$OUT" "$PERCENT" "$MODE" "$GPU" "$PAIR_MODE")

  nohup env GPU="$GPU" RHO="$RHO" PERCENT="$PERCENT" MODE="$MODE" \
    PAIR_MODE="$PAIR_MODE" OUT="$OUT" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
    bash -c '
      set -euo pipefail
      echo "GPU $GPU: rho=$RHO; mode=$MODE; pair_mode=$PAIR_MODE"
      cnn_log=$(printf "%s/CanonicalBridge_CNN_Endpoint%s_%s_seed42.log" "$OUT" "$PERCENT" "$MODE")
      naf_log=$(printf "%s/CanonicalBridge_NAF_Endpoint%s_%s_seed42.log" "$OUT" "$PERCENT" "$MODE")

      if [[ -e "$cnn_log" || -e "$naf_log" ]]; then
        echo "Refusing to overwrite existing per-model log file(s) on GPU $GPU" >&2
        exit 1
      fi

      if [[ "$PAIR_MODE" == "sequential" ]]; then
        echo "GPU $GPU: CNN started, log=$cnn_log"
        bash scripts/run_canonical_endpoint.sh cnn "$RHO" "$GPU" "$MODE" > "$cnn_log" 2>&1
        echo "GPU $GPU: CNN completed"
        echo "GPU $GPU: NAF started, log=$naf_log"
        bash scripts/run_canonical_endpoint.sh naf "$RHO" "$GPU" "$MODE" > "$naf_log" 2>&1
        echo "GPU $GPU: NAF completed"
      else
        echo "GPU $GPU: CNN and NAF started, logs=$cnn_log , $naf_log"
        bash scripts/run_canonical_endpoint.sh cnn "$RHO" "$GPU" "$MODE" > "$cnn_log" 2>&1 &
        CNN_PID=$!
        bash scripts/run_canonical_endpoint.sh naf "$RHO" "$GPU" "$MODE" > "$naf_log" 2>&1 &
        NAF_PID=$!
        set +e
        wait "$CNN_PID"; CNN_STATUS=$?
        wait "$NAF_PID"; NAF_STATUS=$?
        set -e
        echo "GPU $GPU: CNN exit=$CNN_STATUS, NAF exit=$NAF_STATUS"
        if (( CNN_STATUS != 0 || NAF_STATUS != 0 )); then
          echo "At least one model failed on GPU $GPU; inspect model-specific logs" >&2
          exit 1
        fi
      fi
      echo "ALL FINISHED GPU $GPU: both CNN and NAF rho=$RHO mode=$MODE"
    ' > "$LOG" 2>&1 < /dev/null &

  PID="$!"
  disown "$PID"
  printf 'Started GPU %s rho=%s mode=%s (%s): PID=%s log=%s\n' \
    "$GPU" "$RHO" "$MODE" "$PAIR_MODE" "$PID" "$LOG"
done

echo "Four detached GPU workers launched; eight experiments total."
echo "Each model now gets a separate log for NEW launches."
echo "Already-running jobs do not change when you pull these updates."
