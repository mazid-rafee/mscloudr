# Training command

The paper baseline now has one command-line entry point that wires together the
audited components without reintroducing historical implicit behavior.

The command supports exactly two bridge identities at this stage:

    original
    mr_r3

Both use the frozen 13,588,641-parameter legacy_dbcr backbone and the fixed
UnCRtainTS ROI split.

## Before training

From the repository root, point the code at the external dataset:

    export MSCLOUDR_DATA_ROOT="/path/to/SEN12MS-CR"

The command excludes the documented invalid SAR triplet using
manifests/known_invalid_samples.txt and then requires the discovered installation
to match the frozen 122,217-sample fingerprint. A mismatch aborts before model
optimization starts.

## Original sine control

Example on one visible GPU:

    CUDA_VISIBLE_DEVICES=0 python -m mscloudr.cli.train \
      --run-name DBCR_Original_official_seed42 \
      --schedule original \
      --device cuda \
      --epochs 50 \
      --batch-size 4 \
      --num-workers 4 \
      --train-seed 42 \
      --sampler-seed 42

## MR_r3

Use the same settings and change only the run name and schedule:

    CUDA_VISIBLE_DEVICES=0 python -m mscloudr.cli.train \
      --run-name DBCR_MR_r3_official_seed42 \
      --schedule mr_r3 \
      --device cuda \
      --epochs 50 \
      --batch-size 4 \
      --num-workers 4 \
      --train-seed 42 \
      --sampler-seed 42

The mr_r3 command fixes the mean-reversion rate at 3.0. The CLI does not expose
a rate sweep, preventing the canonical identity from becoming ambiguous.

## Outputs

Each run writes under outputs/<run-name>/:

    run_config.json
    history.json
    checkpoints/latest.pt
    checkpoints/best_endpoint.pt

The command prints one compact metric line after each epoch.

best_endpoint.pt is selected only by val_endpoint_l1. val_random_t_l1 remains a
diagnostic quantity and cannot select the canonical checkpoint.

No test-set evaluation is run during training.

## Resume

Resume only into the same run directory. Example:

    CUDA_VISIBLE_DEVICES=0 python -m mscloudr.cli.train \
      --run-name DBCR_Original_official_seed42 \
      --schedule original \
      --device cuda \
      --epochs 50 \
      --batch-size 4 \
      --num-workers 4 \
      --train-seed 42 \
      --sampler-seed 42 \
      --resume outputs/DBCR_Original_official_seed42/checkpoints/latest.pt

The target epoch and runtime device may change on resume. Experiment-defining
settings such as schedule, batch size, learning rate, seeds, split fingerprint,
and model identity must remain unchanged.

## Split seed

There is intentionally no split seed for this protocol. Membership is fixed by
the reference ROI lists. run_config.json records split_seed as null rather than
pretending a random split was used.
