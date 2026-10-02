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


## Real-data smoke run

Before launching a full 50-epoch experiment, use smoke mode to exercise the
actual TIFF loader, verified split, GPU model, optimizer, validation path, and
checkpoint writer on only a few batches:

    CUDA_VISIBLE_DEVICES=0 python -m mscloudr.cli.train \
      --run-name DBCR_Original_smoke_seed42 \
      --schedule original \
      --device cuda \
      --epochs 1 \
      --batch-size 4 \
      --num-workers 4 \
      --train-seed 42 \
      --sampler-seed 42 \
      --smoke-run

By default smoke mode uses 2 training batches and 2 validation batches. These
limits can be changed with --smoke-train-batches and --smoke-val-batches.

Smoke mode requires exactly one epoch and does not support resume. Its
run_config.json and checkpoint metadata explicitly record:

    run_kind = "smoke"
    paper_grade = false
    max_train_batches = 2
    max_val_batches = 2

A smoke checkpoint is only a plumbing artifact. It must not be used in a paper
table or as initialization for the controlled full-run comparison.


## Within-epoch progress logging

Full runs print lightweight progress every 500 processed batches by default.
The final batch of each train/validation phase is always printed as well.

Examples:

    epoch=3 phase=train batch=500/26786 train_l1_running=0.021314
    epoch=3 phase=val batch=500/1794 val_random_t_l1_running=0.018402 val_endpoint_l1_running=0.024116
    epoch=3 train_l1=0.020901 val_random_t_l1=0.017980 val_endpoint_l1=0.023802

Change the interval with:

    --progress-every 250

or disable within-epoch progress while retaining epoch summaries with:

    --progress-every 0

For the two parallel controlled runs, one terminal can monitor both logs:

    watch -n 2 '
    for x in \
      "GPU2 Original |outputs/DBCR_Original_official_seed42.log" \
      "GPU3 MR_r3    |outputs/DBCR_MR_r3_official_seed42.log"
    do
      name="${x%%|*}"
      file="${x#*|}"
      printf "%-14s " "$name"
      if [ -f "$file" ]; then
        tail -c 16384 "$file" \
          | tr "\r" "\n" \
          | grep -E "^epoch=" \
          | tail -1
      else
        echo "log not found"
      fi
    done
    '

The progress values are running means for operational monitoring only. Final
epoch metrics in history.json remain the recorded experimental quantities.
