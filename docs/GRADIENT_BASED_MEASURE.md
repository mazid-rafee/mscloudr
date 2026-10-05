# Gradient-based physical-alpha measure

This branch implements the training experiment that follows the fixed-alpha
gradient diagnostic.

## Parent branch

`exp/physical-alpha-conditioning`

The training branch is intentionally a sibling of the diagnostic branches so
that the only intervention inherited by the model is physical-alpha
conditioning plus the new training measure.

## Calibration source

The fixed CanonicalAlpha pilot checkpoint
`DBCR_CanonicalAlpha_pilot10_seed42/checkpoints/best_endpoint.pt` (epoch 20)
was evaluated at alpha = 0.0, 0.1, ..., 1.0 on 256 matched training batches
(1024 samples per alpha).  For each alpha the diagnostic recorded

`G_rms(alpha) = sqrt(E_B[||grad_theta L1_B(alpha)||_2^2])`.

The exact calibration values and provenance are frozen in
`src/mscloudr/gradient_measure.py`.

## Training measure

The experiment uses a continuous coverage-preserving mixture

`q_eta(alpha) = (1-eta) U(0,1) + eta G_tilde_rms(alpha)`

where `G_tilde_rms` is the piecewise-linear interpolation of the calibrated RMS
gradient profile, normalized to integrate to one.  The first pilot uses
`eta=0.5`, leaving half of the density uniform and allocating the other half
according to the measured optimization signal.

Sampling is implemented by applying the inverse CDF of this density to the
existing uniform discrete timestep variable `u=t/T`.  This preserves the
existing seeded timestep RNG stream while changing the physical-alpha training
measure.  Model conditioning remains `c=T*alpha`.

This measure has no endpoint atom and is deliberately distinct from the Beta
and endpoint-mixture experiments.

## First pilot

Use:

```bash
python -m mscloudr.cli.pilot_train \
  --run-name DBCR_GradientRMS_eta050_pilot10_seed42 \
  --schedule canonical_alpha \
  --conditioning physical_alpha \
  --gradient-mix 0.5 \
  --epochs 25 \
  --device cuda \
  --batch-size 4 \
  --num-workers 4
```

Checkpoint selection remains `val_endpoint_l1`, as in the other NFE=1 pilots.
