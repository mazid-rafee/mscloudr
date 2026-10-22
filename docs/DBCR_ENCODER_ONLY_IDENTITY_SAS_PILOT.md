# Encoder-only identity-centered SaS — controlled 3-epoch pilot

## Question
Does removing near-zero time modulation ONLY inside optical/SAR encoders
improve early cloud removal? The earlier all-block variant also changed
bottleneck and decoder SaS, so this isolates the encoder intervention.

| Modules | Original patent | All-block variant | Encoder-only variant |
|---|---|---|---|
| Optical encoder (31 NAFBlocks) | x*s+b | x*(1+s)+b | **x*(1+s)+b** |
| SAR encoder (31 NAFBlocks) | x*s+b | x*(1+s)+b | **x*(1+s)+b** |
| Bottleneck (1) | x*s+b | x*(1+s)+b | **x*s+b** |
| Decoder (3) | x*s+b | x*(1+s)+b | **x*s+b** |

Same source-guided architecture, original sine bridge, time embeddings,
frozen ROI-disjoint SEN12MS-CR dataset, full validation, and seed 42.
All runs have 19,542,305 parameters. Checkpoints encode the new
architecture_config.scale_shift_mode and per-component equation audit.
The default remains patent SaS; old checkpoints without the key remain
compatible. The new option is encoder_only_identity_centered.

## Step 1: create an isolated Git worktree
From mscloudr-dbcr-identity-sas (do not switch or disrupt other worktrees):

    git fetch origin
    git worktree add -b exp/dbcr-encoder-only-identity-sas-pilot \
      ../mscloudr-dbcr-encoder-only-sas origin/exp/dbcr-encoder-only-identity-sas-pilot
    cd ../mscloudr-dbcr-encoder-only-sas
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

    python -m pytest \
      tests/test_dbcr_encoder_only_sas.py \
      tests/test_dbcr_identity_centered_sas.py \
      tests/test_source_guided_dbcr.py \
      tests/test_patent_blocks.py -q

## Step 2: three FULL epochs on a free GPU (not training GPU 7)

    nvidia-smi -i 0
    mkdir -p ../mscloudr/outputs
    nohup bash -c '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -m mscloudr.cli.train_source_guided_dbcr \
      --run-name DBCR_EncoderOnlyIdentitySaS_seed42_3epoch_pilot \
      --output-root ../mscloudr/outputs \
      --scale-shift-mode encoder_only_identity_centered \
      --time-sharing per_block \
      --epochs 3 \
      --device cuda:0 \
      --batch-size 4 \
      --num-workers 4 \
      --train-seed 42 \
      --sampler-seed 42 \
      --progress-every 500
    ' > ../mscloudr/outputs/DBCR_EncoderOnlyIdentitySaS_seed42_3epoch_pilot.log 2>&1 < /dev/null &
    disown

To watch progress:

    tail -f ../mscloudr/outputs/DBCR_EncoderOnlyIdentitySaS_seed42_3epoch_pilot.log

## Comparison

| Epoch | Original source-guided endpoint L1 | All-block identity endpoint L1 | Encoder-only endpoint L1 |
|---|---:|---:|---:|
| 1 | 0.038223 | 0.037707 | TBD |
| 2 | 0.035474 | 0.034670 | TBD |
| 3 | 0.034116 | 0.033429 | TBD |

Primary metric is validation endpoint L1 (NFE=1). Training and random-t
L1 are secondary. Do not run on held-out test data for pilot selection.
A 3-epoch one-seed pilot is not definitive evidence of generalization.

## Optional post-training internal-path diagnostic

    python -m mscloudr.cli.diagnose_dbcr_internal_paths \
      --checkpoint ../mscloudr/outputs/DBCR_EncoderOnlyIdentitySaS_seed42_3epoch_pilot/checkpoints/latest.pt \
      --device cuda:0 --num-samples 4 --batch-size 1 --num-workers 2

Compare encoder activity and bottleneck/decoder gradient magnitudes.
Do not resume from another modulation variant; each run is independent.
