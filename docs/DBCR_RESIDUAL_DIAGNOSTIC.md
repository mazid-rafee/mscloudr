# Read-only Source-Guided DB-CR residual diagnostic

This diagnostic evaluates a saved checkpoint from the source-guided DB-CR
architecture. It does not change model weights, optimizer, model source,
or training; it uses validation data and not the test set.

For each patent time-conditioned NAFBlock:

    x = block input
    z = x + beta * MBConv(x, alpha)
    y = z + gamma * FFN(z, alpha)

The hooks intercept x (pre-block), z (pre-norm2), and y (post-block).

    MBConv relative L2 = ||z-x||_2 / (||x||_2 + 1e-12)
    FFN relative L2    = ||y-z||_2 / (||z||_2 + 1e-12)
    Total relative L2  = ||y-x||_2 / (||x||_2 + 1e-12)

Norms are computed across C,H,W per sample. Per-sample ratios are
averaged; per-stage figures average constituent blocks' means.
These are activation contribution magnitudes, not accuracy or causal impact.

## Run on an unused GPU, not GPU 7 where training is running

From the mscloudr-dbcr-source-guided worktree:

    git pull --ff-only origin exp/dbcr-source-guided-backbone
    python -m pytest tests/test_dbcr_residual_diagnostics.py -q
    nvidia-smi -i 6

If GPU 6 is available:

    python -m mscloudr.cli.diagnose_dbcr_residuals \
      --checkpoint ../mscloudr/outputs/DBCR_SourceGuided_perblock_full_seed42/checkpoints/latest.pt \
      --device cuda:6 \
      --num-samples 20 \
      --batch-size 2 \
      --num-workers 2

The diagnostic:
- Checks checkpoint identity, model parameter count and frozen ROI split.
- Chooses deterministic evenly-spaced positions in the 7,176-sample
  validation split; records exact indices and sample IDs in the JSON.
- Performs endpoint-only inference (x_T=cloudy, t=T) with no gradients.
- Prints all stage averages and saves per-block detail in a JSON file
  below the run's diagnostics directory with the checkpoint epoch suffix.

The saved latest.pt file may be atomically replaced while training continues;
the diagnostic loads a consistent checkpoint snapshot, records its epoch, and
never modifies the checkpoint. Use a fixed epoch checkpoint copy when exact
epoch stability is necessary. Results refuse overwrite by default.

If stage residual ratios are orders of magnitude smaller than others,
investigate branch activations and gradients next. Small beta/gamma values
or small update norms alone are insufficient to prove an architectural bug.
