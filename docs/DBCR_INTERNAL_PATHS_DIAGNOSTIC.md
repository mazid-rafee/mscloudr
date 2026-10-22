# DB-CR internal time conditioning / channel attention / gradient-flow diagnostic

This is a **checkpoint-only** diagnostic. The model remains in evaluation mode;
a short endpoint L1 backward pass is performed to examine gradients, but the
script creates no optimizer and executes **zero weight updates**.

## Step 1: pull and run data-free tests

From the mscloudr-dbcr-source-guided worktree:

    git pull --ff-only origin exp/dbcr-source-guided-backbone
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python -m pytest tests/test_dbcr_internal_paths.py -q

## Step 2: run on a FREE GPU (example GPU 0, not training GPU 7)

    nvidia-smi -i 0
    python -m mscloudr.cli.diagnose_dbcr_internal_paths \
      --checkpoint ../mscloudr/outputs/DBCR_SourceGuided_perblock_full_seed42/checkpoints/latest.pt \
      --device cuda:0 \
      --num-samples 4 \
      --batch-size 1 \
      --num-workers 2

Default output:
    ../mscloudr/outputs/DBCR_SourceGuided_perblock_full_seed42/diagnostics/internal_paths_epochNNN_n004.json

The same deterministic evenly-spaced validation selection and frozen ROI
fingerprint checks are used as in earlier diagnostics. The snapshot
checkpoint epoch is printed and saved in JSON.

## Measurement meanings

- Time MLP: scale_conv, shift_conv, scale_ffn, shift_ffn magnitudes and RMS
  from final Linear layer, which is actually invoked by from_alpha().
- norm1/norm2 RMS and **post-modulation** conv1/ffn1 input RMS.
- Depthwise output RMS, SimpleGate output RMS, and SCA's effective
  per-channel gate |weight * spatial_mean(SimpleGate_output)|.
- SCA output RMS and beta-weighted SCA output RMS.
- FFN activation RMS before/after SimpleGate and gamma-scaled FFN output.
- Per-block input gradient RMS, beta/gamma gradient magnitudes, and
  weight gradient RMS for conv1/depthwise/sca/ffn1/ffn2/time-MLP-linear3.
- Per-stage means and first/middle/last input gradients for deep 28-block
  optical and SAR encoders.

## Interpretation limitations

Endpoint gradients reflect a single small validation subset and t=T.
The original training samples random t; gradient magnitudes will differ.
Block activation and gradient magnitudes alone cannot prove a bug,
and raw parameter gradient RMS is not invariant to parameterization or width.
Gradient comparisons are diagnostic rather than estimates of learning speed.
Since the MLP modulation uses x*scale+shift (as implemented), small scale
magnitudes *can* suppress normalized input, but nonzero shift and
convolution bias can still drive activation.

Do not modify the current training job or interpret this as a new baseline.
After observing the outputs, we can decide whether to inspect time MLP,
SCA, or gradient attenuation first.
