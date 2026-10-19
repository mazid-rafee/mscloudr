#!/usr/bin/env bash
# Usage: bash scripts/run_canonical_endpoint.sh cnn|naf 0.00|0.10|0.25|0.50 GPU smoke|pilot|full
# FULL runs automatically evaluate the held-out test using the best endpoint checkpoint.
set -euo pipefail

if [[ "$#" != 4 ]]; then
  echo "Usage: $0 <cnn|naf> <0.00|0.10|0.25|0.50> <gpu-index> <smoke|pilot|full>" >&2
  exit 2
fi

BACKBONE="$1"
RHO="$2"
GPU="$3"
MODE="$4"

case "$BACKBONE" in cnn|naf) ;; *) echo "Bad backbone: $BACKBONE" >&2; exit 2 ;; esac
case "$RHO" in 0.00|0.10|0.25|0.50) ;; *) echo "Bad probability: $RHO" >&2; exit 2 ;; esac
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then echo "GPU must be an integer" >&2; exit 2; fi
case "$MODE" in
  smoke) EPOCHS=1 ;;
  pilot) EPOCHS=3 ;;
  full) EPOCHS=50 ;;
  *) echo "Bad mode: $MODE" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
OUT="../mscloudr/outputs"
mkdir -p "$OUT"

if [[ "$BACKBONE" == "cnn" ]]; then NAME="CNN"; else NAME="NAF"; fi
case "$RHO" in
  0.00) PERCENT=0 ;;
  0.10) PERCENT=10 ;;
  0.25) PERCENT=25 ;;
  0.50) PERCENT=50 ;;
esac
RUN=$(printf 'CanonicalBridge_%s_Endpoint%s_%s_seed42' "$NAME" "$PERCENT" "$MODE")

echo "START $RUN on GPU $GPU ($(date -Iseconds))"
echo "Bridge=canonical_alpha; beta_a=1; conditioning=mr_inverse; rho=$RHO; epochs=$EPOCHS"

set -- \
  --run-name "$RUN" \
  --output-root "$OUT" \
  --widths 32 64 128 256 \
  --backbone "$BACKBONE" \
  --endpoint-probability "$RHO" \
  --beta-a 1.0 \
  --conditioning-coordinate mr_inverse \
  --train-seed 42 \
  --sampler-seed 42 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 5e-5 \
  --total-steps 1000 \
  --epochs "$EPOCHS" \
  --device "cuda:$GPU" \
  --progress-every 200

if [[ "$MODE" == "smoke" ]]; then
  set -- "$@" --smoke-run --smoke-train-batches 2 --smoke-val-batches 2
fi
python -m mscloudr.cli.train_canonical_bridge "$@"

if [[ "$MODE" == "full" ]]; then
  echo "TRAIN COMPLETE: $RUN; starting full held-out test evaluation"
  python -m mscloudr.cli.eval_canonical_bridge \
    --checkpoint "$OUT/$RUN/checkpoints/best_endpoint.pt" \
    --device "cuda:$GPU" \
    --batch-size 4 \
    --num-workers 4 \
    --progress-every 100
  echo "TRAIN AND TEST COMPLETE: $RUN ($(date -Iseconds))"
else
  echo "FINISHED $MODE: $RUN ($(date -Iseconds))"
fi
