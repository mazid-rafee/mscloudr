# CanonicalBridge-M: capacity-only scaling experiment

This branch makes encoder widths configurable without changing the backbone or any bridge mathematics.

| Setting | Baseline | CanonicalBridge-M |
|---|---|---|
| Encoder widths | 32, 64, 128, 256 | 48, 96, 192, 384 |
| Trainable parameters | 7,653,037 | 16,947,773 |

Keep bridge geometry straight, the physical alpha distribution uniform (`--beta-a 1.0`), inverse mean-reverting conditioning (`--conditioning-coordinate mr_inverse`), seed 42, multiscale gated SAR fusion, direct x0 prediction, and L1 loss.

The full training protocol uses the frozen SEN12MS-CR ROI split and validation endpoint L1 checkpoint selection.

## Test

```bash
pytest -q tests/test_canonical_bridge_capacity.py tests/test_canonical_bridge_full_controls.py tests/test_canonical_bridge_full_eval.py
```

## Smoke run (after setting actual dataset root)

```bash
python -m mscloudr.cli.train_canonical_bridge \
  --data-root /path/to/SEN12MS-CR \
  --run-name CanonicalBridgeM_MRInverse_seed42_smoke \
  --widths 48 96 192 384 \
  --beta-a 1.0 --conditioning-coordinate mr_inverse \
  --train-seed 42 --sampler-seed 42 \
  --epochs 1 --smoke-run --smoke-train-batches 2 --smoke-val-batches 2
```

## Full run

Match the **baseline's actual batch size and learning rate** before execution (supply `--batch-size` and `--lr` with the exact control values).

```bash
python -m mscloudr.cli.train_canonical_bridge \
  --data-root /path/to/SEN12MS-CR \
  --run-name CanonicalBridgeM_MRInverse_seed42_50epoch \
  --widths 48 96 192 384 \
  --beta-a 1.0 --conditioning-coordinate mr_inverse \
  --train-seed 42 --sampler-seed 42 --epochs 50
```

## Evaluation

```bash
python -m mscloudr.cli.eval_canonical_bridge \
  --data-root /path/to/SEN12MS-CR \
  --checkpoint outputs/CanonicalBridgeM_MRInverse_seed42_50epoch/checkpoints/best_endpoint.pt
```

Width metadata is saved in the run config and checkpoint and used to reconstruct the model during evaluation. Missing width metadata in older checkpoints maps to the baseline widths, while resume rejects width mismatches. No full training was launched from this branch.
