# Identity-centered scale/shift 3-epoch pilot (experimental, NOT the patent DB-CR)

## Hypothesis

Epoch-10 probes show encoder scale values around 0.04–0.05 and
correspondingly small encoder MBConv and FFN activations, whereas the
bottleneck and decoder scales exceed 1. This pilot tests whether
identity-centering the scale alleviates weak encoder branches.

One intended change in **every** time-conditioned NAFBlock (including
both encoders, bottleneck and decoder):

- CONTROL / source-guided patent equation: h * scale(alpha) + shift(alpha)
- EXPERIMENT: h * (1 + scale(alpha)) + shift(alpha)

All trainable parameter names, shapes, counts, initialization RNG,
time-MLP outputs, SFBlocks, bridge schedule and architecture depth stay
identical. Therefore this experiment isolates the *equation*, not just
the encoder: it changes the bottleneck/decoder modulation too.

Do not interpret the experiment as a fix to the source-authentic patent
implementation. The default continues to use the patent equation.

## Step 1: create an isolated worktree (do not switch the running branch)

From the existing mscloudr-dbcr-source-guided worktree:

    git fetch origin exp/dbcr-identity-centered-sas-pilot
    git worktree add ../mscloudr-dbcr-identity-sas exp/dbcr-identity-centered-sas-pilot
    cd ../mscloudr-dbcr-identity-sas
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python -m pytest tests/test_dbcr_identity_centered_sas.py -q

Check unused GPU 0 first:

    nvidia-smi -i 0

## Step 2: independent 3 FULL-epoch pilot

    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -m mscloudr.cli.train_source_guided_dbcr \
      --run-name DBCR_IdentityCenteredSaS_seed42_3epoch_pilot \
      --output-root ../mscloudr/outputs \
      --scale-shift-mode identity_centered \
      --time-sharing per_block \
      --epochs 3 \
      --device cuda:0 \
      --batch-size 4 \
      --num-workers 4 \
      --train-seed 42 \
      --sampler-seed 42 \
      --progress-every 500

NOT a --smoke-run. All train and validation batches run for three
epochs. Training only; no held-out test evaluation.

Independent run outputs:

    ../mscloudr/outputs/DBCR_IdentityCenteredSaS_seed42_3epoch_pilot/
    history.json
    run_config.json
    checkpoints/latest.pt
    checkpoints/best_endpoint.pt

The 50-epoch source-guided control remains untouched in
DBCR_SourceGuided_perblock_full_seed42. Same seed ensures identical
initial model state in tests; GPU kernel determinism is not guaranteed.

## Step 3: what to compare

At epochs 1, 2, 3 compare the new pilot to the ORIGINAL source-guided
run (same 19.54M parameter model; the old data is already logged):

| Epoch | Old source-guided train L1 | Old source-guided val endpoint L1 |
| --- | ---: | ---: |
| 1 | 0.031838 | 0.038223 |
| 2 | 0.025659 | 0.035474 |
| 3 | 0.023534 | 0.034116 |

The decisive metric is validation endpoint L1 (NFE=1). Also compare
validation random-t L1 and training L1 to identify a training-only gain.
Early (3 epoch) results are a pilot, not a final paper result.
Do not run the test set while deciding whether to pursue the variant.

Additional control: run the existing internal_path diagnostic on the
new variant's epoch-3 checkpoint using its saved scale_shift_mode
configuration, and compare absolute residual norms to the original.

Do not copy or resume the ongoing epoch-10 original checkpoint into
the new variant: that would confound the from-scratch comparison.

## Metadata and checkpoint safety

Train parser flag: --scale-shift-mode {patent,identity_centered}
Config field: scale_shift_mode, serialized in architecture_config.
Run metadata also records experiment_variant and conditioning_identity.
Evaluation and diagnostic model reconstruction read the saved mode;
older checkpoints without the key are interpreted as patent.
Existing 50-epoch process is in a different worktree and is not restarted.

Tests inspect the exact pre-convolution and pre-FFN tensors, verify
identical seeded state_dict and parameter counts, and require that the
equation switch changes output once beta and gamma are nonzero.
