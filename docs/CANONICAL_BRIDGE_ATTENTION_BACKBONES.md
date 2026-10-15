# CanonicalBridge Swin and HDiT-inspired controlled backbones

Two opt-in variants are added to the original CanonicalBridgeNet.
Neither uses CAM, ASPP, PSP, nor pretrained segmentation weights.

## Architecture

Common to all models:
- Four optical stages and a separate four-stage CNN SAR encoder.
- Original gated SAR fusion, FiLM conditioning, residual CNN decoder,
  multiscale skip additions, and 13-band direct x0 output.
- Original physical bridge x_alpha=(1-alpha)*x0+alpha*cloudy.
- Uniform alpha sampling, inverse-MR(r=3) conditioning, L1 loss,
  best validation endpoint checkpoint, NFE=1 held-out test evaluation.
- Original widths 32,64,128,256 for controlled experiments.

Backbone settings:
- cnn (default): original residual-convolutional CanonicalBridge unchanged.
- swin: encoder stages 3 and 4 each have two regular/shifted-window
  attention blocks (window 8, shift 4), and bottleneck has two alternating
  windowed attention blocks. High-resolution stages 1 and 2 stay CNN.
- hdit_global: encoder stages 1–3 stay CNN; encoder stage 4 and the
  bottleneck each have two full spatial-attention blocks. SDPA uses
  optimized attention kernels when available.
- The Swin option is Swin-inspired and the HDiT option is HDiT-inspired:
  neither is a pretrained/off-the-shelf full Transformer implementation.
- Checkpoints store backbone_variant and eval reconstructs automatically.
  Old CNN checkpoints continue to load as cnn.

## Pull on Ruby into a separate worktree

~~~bash
cd "$HOME/Desktop/Source Code (Research)/mscloudr"
git fetch origin
git worktree add -b exp/canonical-bridge-attention-backbones \
  "../mscloudr-canonical-attention" \
  origin/exp/canonical-bridge-attention-backbones
cd "../mscloudr-canonical-attention"
conda activate pylrt

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_canonical_bridge_attention_backbones.py \
  tests/test_canonical_bridge_capacity.py \
  tests/test_canonical_bridge_full_eval.py \
  tests/test_canonical_bridge_full_controls.py
~~~

## GPU 6 smoke tests (run after pytest succeeds)

~~~bash
for BACKBONE in swin hdit_global; do
  PYTHONPATH="$PWD/src" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m mscloudr.cli.train_canonical_bridge \
    --run-name "CanonicalBridge_$BACKBONE_MRCond_smoke_seed42" \
    --output-root "../mscloudr/outputs" \
    --widths 32 64 128 256 \
    --backbone "$BACKBONE" --beta-a 1.0 \
    --conditioning-coordinate mr_inverse \
    --train-seed 42 --sampler-seed 42 \
    --batch-size 4 --epochs 1 \
    --smoke-run --smoke-train-batches 2 --smoke-val-batches 2 \
    --device cuda:6
done
~~~

Note: use an explicit RUN name if running the above as written in shell;
the sequence "$BACKBONE_MRCond" would otherwise be treated as one shell
variable. Prefer the corrected smoke loop below:

~~~bash
for BACKBONE in swin hdit_global; do
  RUN=$(printf 'CanonicalBridge_%s_MRCond_smoke_seed42' "$BACKBONE")
  PYTHONPATH="$PWD/src" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m mscloudr.cli.train_canonical_bridge \
    --run-name "$RUN" --output-root "../mscloudr/outputs" \
    --widths 32 64 128 256 --backbone "$BACKBONE" \
    --beta-a 1.0 --conditioning-coordinate mr_inverse \
    --train-seed 42 --sampler-seed 42 \
    --batch-size 4 --epochs 1 \
    --smoke-run --smoke-train-batches 2 --smoke-val-batches 2 \
    --device cuda:6
done
~~~

## Detached 50-epoch training + automatic full NFE=1 test evaluation

Run from mscloudr-canonical-attention after smoke tests.
Launch one backbone at a time on GPU 6. Set either:
- BACKBONE=swin; RUN=CanonicalBridge_Swin_MRCond_full_seed42
- BACKBONE=hdit_global; RUN=CanonicalBridge_HDiTGlobal_MRCond_full_seed42

~~~bash
BACKBONE=swin
RUN=CanonicalBridge_Swin_MRCond_full_seed42

nohup env BACKBONE="$BACKBONE" RUN="$RUN" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD/src" \
  bash -c '
    set -euo pipefail
    OUT="../mscloudr/outputs"
    CFG="../mscloudr/outputs/CanonicalBridge_Uniform_MRCond_full_seed42/run_config.json"
    read -r LR BS NW TSTEPS < <(
      python -c "import json,sys; c=json.load(open(sys.argv[1])); print(c[\"learning_rate\"],c[\"batch_size\"],c[\"num_workers\"],c[\"total_steps\"])" "$CFG"
    )
    python -m mscloudr.cli.train_canonical_bridge \
      --run-name "$RUN" --output-root "$OUT" \
      --widths 32 64 128 256 --backbone "$BACKBONE" \
      --beta-a 1.0 --conditioning-coordinate mr_inverse \
      --train-seed 42 --sampler-seed 42 \
      --lr "$LR" --batch-size "$BS" --num-workers "$NW" \
      --total-steps "$TSTEPS" --epochs 50 \
      --device cuda:6 --progress-every 200
    python -m mscloudr.cli.eval_canonical_bridge \
      --checkpoint "$OUT/$RUN/checkpoints/best_endpoint.pt" \
      --device cuda:6 --batch-size 4 --num-workers 4 \
      --progress-every 100
    echo "TRAIN AND EVALUATION COMPLETE: $RUN"
  ' > "../mscloudr/outputs/$RUN.log" 2>&1 < /dev/null &
echo "PID: $!"
disown
~~~

The output is under the ORIGINAL repository's mscloudr/outputs.
Monitor with: tail -f "../mscloudr/outputs/$RUN.log"

Launch HDiT-inspired after Swin completes, changing BACKBONE and RUN.
Both model architectures are untested experimentally until these are run.

## Measurements

Compare held-out test L1/MAE, PSNR, SSIM, SAM, RMSE and model
parameter counts plus GPU memory and latency against CNN baseline.
