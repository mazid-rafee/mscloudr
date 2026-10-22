# Source-guided DB-CR SAR sensitivity diagnostic

This read-only CLI compares one saved checkpoint with identical cloudy optical inputs,
clean targets, and endpoint time T on fixed ROI-disjoint **validation** samples:

- NORMAL: correctly paired normalized SAR
- SHUFFLED: deterministic cyclic permutation of SAR across all 20 selected samples
- ZERO: SAR replaced with zeros in normalized input space

Cross-sample SAR pairing is done globally rather than within small batches.

## Launch

From the mscloudr-dbcr-source-guided worktree:

    git pull --ff-only origin exp/dbcr-source-guided-backbone
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python -m pytest tests/test_dbcr_sar_sensitivity.py -q
    nvidia-smi -i 0

With free GPU 0:

    python -m mscloudr.cli.diagnose_dbcr_sar_sensitivity \
      --checkpoint ../mscloudr/outputs/DBCR_SourceGuided_perblock_full_seed42/checkpoints/latest.pt \
      --device cuda:0 \
      --num-samples 20 \
      --batch-size 2 \
      --num-workers 2

Output: diagnostics/sar_sensitivity_epochNNN.json under the training run.

## Measurements

- Mean validation reconstruction L1 for NORMAL, SHUFFLED and ZERO
- Mean change in reconstruction L1 vs NORMAL for SHUFFLED and ZERO
- Absolute L1, RMS and relative L2 change of final optical prediction
- Absolute L1, RMS and relative L2 output change of each of the four SFBlocks
- Exact sample IDs, shuffled SAR source IDs, and per-sample reconstruction errors

These are diagnostic validation measurements, not held-out test-set metrics.
Substantial sensitivity does not establish that SAR guidance is beneficial;
consider change in accuracy against the unchanged clean targets.
The ZERO-SAR condition can be out-of-distribution.

The checkpoint is loaded independently; no training/model/optimizer/checkpoint
files are changed. The CLI refuses to overwrite existing results by default.
