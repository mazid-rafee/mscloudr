# CanonicalBridgeNet: endpoint gradient alignment

This branch migrates the endpoint-gradient-alignment diagnostic to the
**original 7,653,037-parameter CanonicalBridgeNet CNN backbone**. It does not
modify SourceGuidedDBCRNet, LegacyDBCRNet, or existing training/evaluation
runners.

## Frozen experimental identity

- Backbone: `CanonicalBridgeNet(backbone_variant="cnn", widths=(32,64,128,256))`
- Training alpha measure: uniform discrete `alpha=t/T` (beta-a=1)
- Conditioning: `physical_alpha` (network receives `T*alpha`)
- No explicit endpoint mixture
- Geometry: straight bridge `x_alpha=(1-alpha)*x0+alpha*y`
- Endpoint: `T`; prediction: direct x0; NFE: 1
- Split: full verified UnCRtainTS ROI-disjoint SEN12MS-CR train, never val/test
- Model weights frozen; exactly the checkpoint's architecture and tensors
- Purpose: measure alignment of intermediate-t gradients with NFE=1 endpoint
  gradients. It is a *diagnostic*, not a training method.

## Local execution

```bash
git fetch origin
git switch exp/canonical-endpoint-alignment
python -m pytest -q tests/test_endpoint_alignment.py tests/test_canonical_endpoint_alignment.py

python -m mscloudr.cli.diagnose_canonical_alignment \
  --checkpoint outputs/CANONICAL_UNIFORM_FULL/checkpoints/best_endpoint.pt \
  --data-root /path/to/SEN12MS-CR \
  --device cuda:0 \
  --batch-size 1 --max-batches 4 --num-bins 8 \
  --sampler-seed 42 \
  --output-dir outputs/canonical_endpoint_alignment/uniform_seed42
```

Use an *actual* available checkpoint path. Use a GPU not occupied by training.
If the checkpoint came from a different repository or lacked the stored split
metadata, migrate/check its provenance rather than disabling the audit.
Checkpoint selection remains by `val_endpoint_l1`.

Output: `alignment.json`. Per-bin statistics include gradient cosine,
dot product, endpoint projection, gradient norms, losses and alpha/timestep
means and standard deviations across sampled training batches. Timesteps
0..T-1 are binned; endpoint T is explicitly separate.

## Interpretation

Positive cosine indicates a locally aligned endpoint-improvement gradient
under first-order SGD; it does **not** establish a long-run improvement under
Adam. Magnitude is separately reported because cosine ignores gradient size.
The 4-batch default is a smoke diagnostic: increase samples and repeat with
multiple seeds or checkpoints before designing learned weights.

Do not launch a new 50-epoch method merely because a bin has a positive
cosine. First define a bounded weighting rule, compare with the existing
uniform physical-alpha control, and test with a matched pilot. Direction B's
component-decomposition model also requires its own non-redundancy checks.
