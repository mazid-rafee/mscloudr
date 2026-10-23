# CanonicalBridge independently supervised components (Direction B)

**Experimental hypothesis, not yet a validated improvement.** The experiment
keeps the original CNN CanonicalBridge backbone, uniform physical alpha,
MR-inverse r=3 conditioning, original straight geometry, and NFE=1 direct-x0
endpoint evaluation. It changes only the last output head and training loss.

## Method

Split the clean optical target into orthogonal 2x2-block subspaces:

- P_L(x): 2x2 spatial block average, repeated back to full spatial resolution.
- P_H(x) = x - P_L(x).
- P_L^2=P_L, P_H^2=P_H, P_L P_H=0, P_L+P_H=I.

The network's shared decoder produces two independent output tensors, with
learned separate output kernels: raw_low and raw_high.

    x0_hat = P_L(raw_low) + P_H(raw_high)

Training minimizes

    L = L1(x0_hat, x0) + lambda/2 * (
          L1(P_L(raw_low), P_L(x0)) + L1(P_H(raw_high), P_H(x0))).

The independent targets are identifiable orthogonal subspace components; this
does not create independent information or prove novel bridge dynamics.
The shared trunk can still couple the two optimization signals. The additive
component loss is auxiliary regularization. Any performance improvement should
be compared against *both* the original CanonicalBridge checkpoint and a
lambda=0 two-head architecture control to separate head capacity from loss.

**Important:** The historical runner's `train_l1` field stores the composite
training objective for this experiment, not plain training L1; compare only
`val_endpoint_l1`, `val_random_t_l1`, and paper-grade test metrics.

## Procedure: no git switch

From the original `mscloudr` repo worktree (not the separate A worktree):

```bash
git fetch origin
git worktree add --detach ../mscloudr-components origin/exp/canonical-component-supervision
cd ../mscloudr-components
conda activate pylrt
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_canonical_components.py
```

Use CUBLAS_WORKSPACE_CONFIG before Python if deterministic CUDA is enabled.
Run smoke test on an idle GPU before a full run:

```bash
export OUT="/aul/homes/mmazi007/Desktop/Source Code (Research)/mscloudr/outputs"
export DATA="/a/bear.cs.fiu.edu./disk/bear-b/users/mmazi007/Desktop/Source Code (Research)/Cloud Removal/data/SEN12MS-CR"
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH="$PWD/src" python -m mscloudr.cli.train_canonical_components \
  --data-root "$DATA" --output-root "$OUT" \
  --run-name CanonicalComp_mrInverse_seed42_smoke \
  --epochs 1 --smoke-run --smoke-train-batches 2 --smoke-val-batches 2 \
  --batch-size 4 --train-seed 42 --sampler-seed 42 \
  --device cuda:0 --component-weight 0.5
```

GPU 3 is just an example and may be occupied. This smoke test cannot be used
as a paper-grade performance result. Inspect checkpoint and history afterward.

Only after passing the unit tests and smoke run, a 50-epoch full run may use

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=GPU_ID \
  python -m mscloudr.cli.train_canonical_components \
  --data-root "$DATA" --output-root "$OUT" \
  --run-name CanonicalComp_mrInverse_seed42_lambda05 \
  --epochs 50 --batch-size 4 --train-seed 42 --sampler-seed 42 \
  --device cuda:0 --component-weight 0.5
```

Replace GPU_ID with an available physical GPU index (do not type GPU_ID literally).
After the successful run, evaluate the best endpoint checkpoint:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=GPU_ID \
  PYTHONPATH="$PWD/src" python -m mscloudr.cli.eval_canonical_components \
  --checkpoint "$OUT/CanonicalComp_mrInverse_seed42_lambda05/checkpoints/best_endpoint.pt" \
  --data-root "$DATA" --device cuda:0 \
  --output "$OUT/CanonicalComp_mrInverse_seed42_lambda05/eval_metrics.json"
```

Do not launch a full run while GPUs are oversubscribed, or assume that unit
tests imply end-to-end dataset training works. Test a small pilot first and
check loss, gradient flow, and checkpoint reload.
