# Endpoint-gradient alignment diagnostic

This is a **read-only TRAIN-split diagnostic** for the source-guided DB-CR
backbone and original sine bridge. It is not a new training method.

From a frozen checkpoint, for a fixed training batch, compare the gradient of
the endpoint direct-x0 L1 loss at `t=T` against intermediate-t loss gradients.
Eight disjoint bins cover timesteps 0..T-1; the endpoint is evaluated separately.
Each bin samples one timestep per image in the batch.

Metrics: full-parameter gradient cosine, gradient dot product, endpoint
projection (dot / endpoint gradient squared norm), norms, L1 and alpha.
Results are batch means/std; 4 batches is exploratory and *not* statistical
validation. Loss gradients are never used for an optimizer update.

The diagnostic verifies the frozen ROI-disjoint membership and checkpoint split
fingerprint. No validation/test samples are passed to the profiler. It never
changes the source-guided training entry point.

Example (choose an idle GPU, *not* GPU 7 if occupied):

```bash
python -m mscloudr.cli.diagnose_endpoint_alignment \
  --checkpoint outputs/SOURCE_RUN/checkpoints/latest.pt \
  --data-root /path/to/SEN12MS-CR \
  --device cuda:0 --max-batches 4 --num-bins 8 \
  --output-dir outputs/endpoint_alignment/source_epochN
```

The command writes `alignment.json`. This code does not launch full training.
A negative gradient cosine is local evidence of gradient conflict, not proof
that removing a timestep improves long-run endpoint accuracy. The checkpoint
must not be modified. To avoid racing a running writer, analyze a copied
checkpoint or a checkpoint from a completed epoch.

Full runs require independent smoke tests and a separately justified
timestep-weighting strategy.
