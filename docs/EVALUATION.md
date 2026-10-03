# Evaluation command

The paper baseline has a dedicated held-out test evaluator for trained
`legacy_dbcr` checkpoints.

Evaluation uses deterministic **NFE=1 endpoint inference**:

    x_T = cloudy
    t = T
    prediction = model(x_T, t, SAR)

No bridge schedule is evaluated at test time for NFE=1. The checkpoint's
`original` or `mr_r3` schedule identity is retained in the output only as
training provenance. This prevents an evaluation-time schedule argument from
silently changing the experimental identity.

## Before evaluation

From the repository root, point the code at the audited SEN12MS-CR install:

    export MSCLOUDR_DATA_ROOT="/path/to/SEN12MS-CR"

The evaluator re-discovers the dataset, excludes the documented invalid sample,
verifies the frozen 122,217-sample membership, reconstructs the fixed
UnCRtainTS ROI split, and requires that the resulting split audit exactly match
the checkpoint provenance.

## Canonical evaluation

Use `best_endpoint.pt`, because training selects the canonical NFE=1 checkpoint
only by `val_endpoint_l1`.

Original example:

    CUDA_VISIBLE_DEVICES=0 python -m mscloudr.cli.eval \
      --checkpoint outputs/DBCR_Original_official_seed42/checkpoints/best_endpoint.pt \
      --device cuda \
      --batch-size 4 \
      --num-workers 4

MR_r3 example:

    CUDA_VISIBLE_DEVICES=1 python -m mscloudr.cli.eval \
      --checkpoint outputs/DBCR_MR_r3_official_seed42/checkpoints/best_endpoint.pt \
      --device cuda \
      --batch-size 4 \
      --num-workers 4

The installed console-script equivalent is:

    mscloudr-eval --checkpoint <checkpoint> --device cuda

## Output

For a checkpoint under `<run-dir>/checkpoints/`, the default output is:

    <run-dir>/eval_metrics.json

The JSON records:

- checkpoint path and epoch;
- checkpoint-selection metadata;
- model and training schedule identity;
- exact test split audit;
- NFE=1 endpoint inference identity;
- test sample and batch counts;
- metric implementation provenance;
- per-image arithmetic-mean metrics.

The current paper-facing reference metrics are:

    L1 / MAE
    RMSE
    PSNR
    SAM (degrees)
    SSIM

`L1` is an alias of the per-image MAE mean so the output remains easy to compare
with historical DB-CR tables. The underlying definitions follow the repository's
UnCRtainTS-compatible metric implementation. Predictions are not clipped before
metric computation.

LPIPS and FID are not part of this reference evaluator yet; they should be added
as a separate perceptual-metric layer rather than mixed silently into the
benchmark-compatible reconstruction metrics.

## Progress

The evaluator prints running L1 and PSNR every 100 test batches by default:

    eval batch=100/1975 samples=400 L1=0.012345 PSNR=33.9876

Change the interval with `--progress-every N`, or disable progress with
`--progress-every 0`.

## Diagnostic truncation

For a quick plumbing check only:

    python -m mscloudr.cli.eval \
      --checkpoint <checkpoint> \
      --device cuda \
      --max-batches 2 \
      --output /tmp/mscloudr_eval_smoke.json

Any use of `--max-batches` marks the output `paper_grade=false`.

Smoke-training checkpoints are rejected by default. They can be evaluated only
with `--allow-smoke-checkpoint`, and remain non-paper-grade.

Existing output files are not overwritten unless `--overwrite` is supplied.
