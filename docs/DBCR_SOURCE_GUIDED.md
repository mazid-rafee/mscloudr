# DB-CR source-guided architecture v1 (NOT authors' official code)

Branch: exp/dbcr-source-guided-backbone

## What is implemented

New isolated model identity: dbcr_source_guided_v1.
Never substitute for legacy_dbcr or reuse legacy checkpoint state dictionaries.

Topology follows DB-CR paper:
- Optical input 13 channels; SAR input 2 channels.
- Optical U-Net and dedicated SAR encoder, both 4 scales.
- Published widths [22,44,88,176].
- Encoder NAF blocks [1,1,1,28] on BOTH optical and SAR branches.
- Decoder/bottleneck NAF blocks [1,1,1,1].
- SFBlock heads [1,1,2,4], using the existing audited fusion module.
- Direct x0 prediction, straight bridge
  x_t=(1-alpha_t)*x0+alpha_t*y, alpha_t=sin(pi*t/(2*T)).
- Frozen UnCRtainTS ROI-disjoint data/evaluation protocol.

The existing PatentTimeEmbeddedNAFBlock is applied to EVERY NAF block
in optical encoder, SAR encoder, bottleneck and optical decoder.
Its time module consumes alpha(t), uses three Linear layers and two
SimpleGates to produce four scale/shift vectors for convolution/FFN.
The normalized-feature modulation follows the patent disclosure.

The SFBlock is from mscloudr.models.blocks.SFBlock,
not historical LegacySFBlock.

## Unresolved settings: explicit ASSUMPTIONS, not recovered facts

- Positional time-embedding dimension: default 128, not source-established.
- Time MLP hidden dimension: default 128, not source-established.
- Time MLP sharing: default independent per NAF block; optional per
  feature width using --time-sharing per_width. Not source-established.
- NAF feature-map norm reduction follows the patent's whole-feature-map
  description, with per-channel affine weights (assumption).
- SF attention-head/operator details follow our paper audit, not authors' code.
- Separate modality downsampling, transpose-conv upsampling, one
  bottleneck NAF block and 3x3 output head are audited assumptions.
- Matching reported 18.06M parameters is NOT used to pick dimensions/sharing.

This is a source-guided reproduction candidate, NOT a verified
exact/paper-faithful implementation.

Source material:
- DB-CR paper (project attachment DB-CR.pdf)
- US patent US20260289738A1
- docs/MODEL_AUDIT.md
- docs/PATENT_ARCHITECTURE.md
- docs/TIME_EMBEDDING_SOURCE_AUDIT.md

## Step 1 — Architecture audit and tests (NO training)

From mscloudr worktree on the new branch:

    python -m mscloudr.cli.audit_source_guided_dbcr --smoke-forward
    python -m mscloudr.cli.audit_source_guided_dbcr --time-sharing per_width
    python -m pytest tests/test_source_guided_dbcr.py tests/test_patent_blocks.py tests/test_dbcr_network.py -q

Record the actual trainable parameter count in both modes. Do not modify
hidden dimensions merely to force a match with 18.06M. Do NOT start a
full 50-epoch experiment before these commands pass on Ruby.

## Step 2 — Short smoke training, after Step 1 passes

    python -m mscloudr.cli.train_source_guided_dbcr \
      --run-name DBCR_SourceGuided_perblock_smoke_seed42 \
      --output-root ../mscloudr/outputs \
      --smoke-run --epochs 1 \
      --device cuda:4 --batch-size 4 --num-workers 4

This uses the audited ROI-disjoint manifest but only two training
and two validation batches. Its checkpoint is non-paper-grade.

## Step 3 — Full training and test (DO NOT start automatically)

    python -m mscloudr.cli.train_source_guided_dbcr \
      --run-name DBCR_SourceGuided_perblock_full_seed42 \
      --output-root ../mscloudr/outputs \
      --epochs 50 \
      --device cuda:4 --batch-size 4 --num-workers 4

    python -m mscloudr.cli.eval_source_guided_dbcr \
      --checkpoint ../mscloudr/outputs/DBCR_SourceGuided_perblock_full_seed42/checkpoints/best_endpoint.pt \
      --device cuda:4 --batch-size 4 --num-workers 4

Training is NOT detached by default. Use nohup/disown if needed AFTER
verification. The trainer selects best_endpoint.pt by val_endpoint_l1.
Evaluator checks model identity, config, split audit, parameter count
and strict state-dict compatibility before full 7,899-sample NFE=1
evaluation. It writes eval_metrics.json without prediction clipping.

## Interpretation

Compare v1 with LegacyDBCRNet (29.939 dB on our ROI-disjoint protocol)
with the exact same manifest and metrics. Architecture and conditioning
change together, so one comparison does not causally isolate either.
Follow with controlled ablations. More parameters do not guarantee a gain.
