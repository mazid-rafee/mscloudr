# CanonicalBridge capacity: paired fixed-alpha validation diagnostic

Compare original 7.65M CanonicalBridge and CanonicalBridge-M (16.95M)
at identical bridge positions using identical validation patches.

- Frozen ROI-disjoint SEN12MS-CR **validation split only** (7,176 samples).
- At each alpha: x_alpha = (1-alpha)*target + alpha*cloudy.
- MR-inverse r=3 conditioning via mscloudr.training.model_conditioning_coordinate.
- Identical x_alpha and SAR input for both checkpoints.
- Per-image reference metrics: MAE/L1, RMSE, PSNR, SAM, SSIM. No clipping.
- Output: both model metrics and signed M-minus-baseline differences.
- Alpha 1 matches the existing deterministic NFE=1 endpoint calculation.
- At alpha less than 1 the clean target is used to build the bridge: this
  is an **oracle diagnostic**, NOT deployable cloud-removal inference.
- No training or checkpoint modifications.

## Pull the code on Ruby

From the sibling worktree:

~~~bash
cd "$HOME/Desktop/Source Code (Research)/mscloudr-canonical-m"
git pull --ff-only origin exp/canonical-bridge-m-capacity
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_canonical_bridge_alpha_grid.py
~~~

## Pilot: 50 validation batches (200 samples), GPU 6

~~~bash
PYTHONPATH="$PWD/src" python -m mscloudr.cli.eval_canonical_bridge_alpha_grid \
  --checkpoint-a "../mscloudr/outputs/CanonicalBridge_Uniform_MRCond_full_seed42/checkpoints/best_endpoint.pt" \
  --checkpoint-b "../mscloudr/outputs/CanonicalBridgeM_Uniform_MRCond_full_seed42/checkpoints/best_endpoint.pt" \
  --alphas 0.25 0.50 0.75 0.90 0.95 1.00 \
  --split val --device cuda:6 --batch-size 4 --num-workers 4 \
  --max-batches 50 --progress-every 10 \
  --output "../mscloudr/outputs/CanonicalBridge_capacity_alpha_grid_pilot.json"
~~~

## Full validation: detached on GPU 6

Only after pilot and tests succeed:

~~~bash
nohup env PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 \
  python -m mscloudr.cli.eval_canonical_bridge_alpha_grid \
  --checkpoint-a "../mscloudr/outputs/CanonicalBridge_Uniform_MRCond_full_seed42/checkpoints/best_endpoint.pt" \
  --checkpoint-b "../mscloudr/outputs/CanonicalBridgeM_Uniform_MRCond_full_seed42/checkpoints/best_endpoint.pt" \
  --alphas 0.25 0.50 0.75 0.90 0.95 1.00 \
  --split val --device cuda:6 --batch-size 4 --num-workers 4 \
  --progress-every 50 \
  --output "../mscloudr/outputs/CanonicalBridge_capacity_alpha_grid_full.json" \
  > "../mscloudr/outputs/CanonicalBridge_capacity_alpha_grid_full.log" 2>&1 \
  < /dev/null &
echo "PID: $!"
disown
~~~

Monitor:

~~~bash
tail -f "../mscloudr/outputs/CanonicalBridge_capacity_alpha_grid_full.log"
~~~

For L1/MAE, RMSE, SAM, positive delta_b_minus_a is worse for the M model.
For PSNR/SSIM, negative delta_b_minus_a is worse.

All outputs are separate from training runs and existing endpoint metrics.
