# Difficulty-based physical-alpha training measure

This branch implements experiment #3 after the completed train-only fixed-alpha
difficulty diagnostic.

## Diagnostic source

The calibration comes from `exp/difficulty-adaptive-measure` at commit
`54db89eb65e6ddf68b5b6a96d1fdaac9b955eba1` using the CanonicalAlpha pilot
checkpoint `DBCR_CanonicalAlpha_pilot10_seed42/checkpoints/best_endpoint.pt`
(epoch 20).

The diagnostic evaluated the full deterministic pilot training split only
(10,714 samples) at physical-alpha knots `0.0, 0.1, ..., 1.0`, with physical
conditioning `c=T*alpha`. Validation and test were not used to design the
sampler.

The measured train L1 profile was:

| alpha | train mean L1 |
|---:|---:|
| 0.0 | 0.0077307615 |
| 0.1 | 0.0088208230 |
| 0.2 | 0.0102809153 |
| 0.3 | 0.0119112128 |
| 0.4 | 0.0135630580 |
| 0.5 | 0.0151414039 |
| 0.6 | 0.0168842498 |
| 0.7 | 0.0182927899 |
| 0.8 | 0.0203341824 |
| 0.9 | 0.0231053244 |
| 1.0 | 0.0261503302 |

The endpoint difficulty is about 3.38x the alpha=0 difficulty.

## Training measure

Let `E(alpha)` be the piecewise-linear interpolation of the measured train L1
profile and let `E_tilde(alpha)` be that interpolation normalized to integrate
to one over `[0,1]`.

The training density is

`q_eta(alpha) = (1-eta) U(0,1) + eta E_tilde(alpha)`.

The first pilot uses `eta=0.5`. The 50% uniform component preserves broad
physical-alpha coverage while the remaining 50% shifts training probability
toward empirically harder bridge states.

Sampling is implemented by inverse-CDF transformation of the existing uniform
discrete timestep variable `u=t/T`, so no new RNG stream is introduced. The
model continues to use physical-alpha conditioning `c=T*alpha`.

This measure is continuous and has no endpoint atom. Therefore it is distinct
from the endpoint-mixture experiment while still naturally emphasizing the
high-alpha region because the measured training difficulty increases toward the
cloudy endpoint.

## Intended pilot

```bash
python -m mscloudr.cli.pilot_train \
  --run-name DBCR_DifficultyMix_eta050_pilot10_seed42 \
  --schedule canonical_alpha \
  --conditioning physical_alpha \
  --difficulty-mix 0.5 \
  --epochs 25 \
  --device cuda \
  --batch-size 4 \
  --num-workers 4
```

Checkpoint selection remains `val_endpoint_l1`, consistent with the NFE=1
pilot protocol.
