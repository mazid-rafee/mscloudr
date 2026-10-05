# Generalization-Based Physical-Alpha Measure

## Purpose

Experiment #5 converts the completed fixed-alpha train/validation diagnostic into
a training measure. The goal is to emphasize bridge corruption levels where the
CanonicalAlpha checkpoint shows the largest held-out validation gap, while
preserving broad support over the full physical-alpha interval.

This branch is a sibling of the other measure experiments and is based directly
on `exp/physical-alpha-conditioning`.

## Calibration signal

The sampler uses the sample-weighted held-out gap

```text
D_gen(alpha) = L_val(alpha) - L_train(alpha)
```

measured at alpha = 0.0, 0.1, ..., 1.0 from the seed-42 CanonicalAlpha pilot
checkpoint selected at epoch 20. The final test split was not used for sampler
design.

The measured gaps are:

| alpha | val - train L1 |
|---:|---:|
| 0.0 | 0.0022379570 |
| 0.1 | 0.0028030839 |
| 0.2 | 0.0035307490 |
| 0.3 | 0.0042798940 |
| 0.4 | 0.0048074870 |
| 0.5 | 0.0060273567 |
| 0.6 | 0.0067868381 |
| 0.7 | 0.0073319608 |
| 0.8 | **0.0076342532** |
| 0.9 | 0.0066968164 |
| 1.0 | 0.0064804411 |

The gap peaks at alpha=0.8 and then decreases toward the endpoint. This makes the
signal distinct from simply weighting by absolute corruption difficulty or by a
monotone endpoint-biased density.

## Training measure

Let `D_tilde_gen(alpha)` be the normalized piecewise-linear interpolation of the
measured positive gap profile. The training density is

```text
q_eta(alpha) = (1-eta) U(0,1) + eta D_tilde_gen(alpha)
```

The first pilot uses `eta=0.5`. Thus half of the density remains uniform and half
is allocated according to the generalization-gap signal.

Sampling is performed through the inverse CDF of this continuous density using
the existing deterministic base quantile `u=t/T`. The model still receives
physical-alpha conditioning `c=T*alpha`.

## First pilot

```bash
python -m mscloudr.cli.pilot_train \
  --run-name DBCR_GeneralizationMix_eta050_pilot10_seed42 \
  --schedule canonical_alpha \
  --conditioning physical_alpha \
  --generalization-mix 0.5 \
  --epochs 25
```

Use the same pilot data, optimizer, seed, batch size, and NFE=1 endpoint
checkpoint-selection protocol as the common physical-alpha baseline.

## Interpretation limits

`D_gen(alpha)` is a held-out validation-gap signal. It should not be interpreted
as a causal estimate of geographic shift because train and validation differ in
ROI/season composition. The untouched final test set remains the evaluation set
for comparing the trained measure against the other controls.
