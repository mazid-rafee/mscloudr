# Epoch runner and checkpoints

The clean runner now wraps the already-tested bridge batch computations with
optimizer, epoch aggregation, validation, and epoch-boundary checkpointing.

## Optimizer

The default optimizer is Adam with learning rate 5e-5, matching the historical
ms-cloudR training setup.

## Checkpoint selection

For NFE=1 experiments:

    canonical selection metric: val_endpoint_l1
    diagnostic metric:          val_random_t_l1

The runner writes:

    checkpoints/latest.pt
    checkpoints/best_endpoint.pt

There is intentionally no best_random.pt in the canonical path.

best_endpoint.pt is replaced only when val_endpoint_l1 strictly improves.

## Resume state

An epoch-boundary checkpoint stores:

- model state;
- Adam/optimizer state;
- completed epoch;
- current validation metrics;
- best val_endpoint_l1 so far;
- run metadata;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- CUDA RNG states when available;
- explicit bridge sampler generator state;
- train/validation DataLoader generator states.

This allows a resumed run to continue the same shuffle and bridge-timestep
streams at the next epoch boundary.

Mid-epoch resume is not supported.

## Metadata

The runner always records the following protocol fields:

- model_identity;
- schedule_name;
- total_steps;
- target_epochs;
- optimizer class;
- optimizer learning rates;
- checkpoint_selection_metric;
- diagnostic_random_t_metric.

The future command-line training entry point must additionally provide the
experiment seed configuration, split fingerprint/protocol, dataset identity,
and other run-specific settings through run_metadata.

## History

history.json contains one entry per completed epoch:

    epoch
    train_l1
    val_random_t_l1
    val_endpoint_l1

When resuming in the same output directory, existing history is retained and
the checkpoint epoch must agree with the final recorded history epoch.
