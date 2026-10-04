# Uniform-alpha conditioning-coordinate control

## Goal

Separate the effect of the physical bridge training measure from the scalar
coordinate passed to DB-CR's time embedding.

The existing `canonical_alpha` pilot samples

\[
t \sim \mathrm{Uniform}\{0,\ldots,T\},\qquad \alpha=t/T,
\]

so the physical bridge-state measure is uniform on the discrete alpha grid:

\[
q(\alpha)=\mathrm{Uniform}\{0,1/T,\ldots,1\}.
\]

CanonicalAlpha also conditions the legacy DB-CR network with `t`, which is
identical to `T*alpha` under this parameterization.

## UniformAlpha-SineCond

This control keeps the *same* sampled alpha and therefore the same bridge state

\[
x_\alpha=(1-\alpha)x_0+\alpha y,
\]

but changes only the conditioning coordinate to the inverse of DB-CR's
original sine schedule:

\[
c_{\mathrm{sine}}(\alpha)
=T\,\frac{2}{\pi}\arcsin(\alpha).
\]

The intended CLI configuration is therefore:

```bash
--schedule canonical_alpha --conditioning inverse_sine
```

The CLI rejects `inverse_sine` with `original` or `mr_r3` schedules so that the
experiment cannot accidentally change the physical training measure.

## Controlled comparison

| Experiment | physical q(alpha) | model conditioning |
|---|---|---|
| CanonicalAlpha | uniform discrete alpha | `T*alpha` |
| UniformAlpha-SineCond | uniform discrete alpha | `T*(2/pi)*asin(alpha)` |

Everything else should remain matched: pilot subset, seed, architecture,
optimizer, epoch count, checkpoint selection, and NFE=1 endpoint evaluation.

A performance difference between these two runs is therefore evidence that the
conditioning coordinate matters even when the physical bridge-state training
measure is held fixed.

At alpha=1 both coordinates equal T, so NFE=1 endpoint evaluation uses the
same endpoint conditioning value.
