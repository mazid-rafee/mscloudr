# Canonical-alpha pilot

## Research question

DB-CR constructs the same straight physical bridge

\[
x_\alpha=(1-\alpha)x_0+\alpha y,
\]

for every strictly monotone scalar parameterization. The original sine and
MR-r3 choices therefore differ primarily in how uniform discrete timestep
sampling distributes training mass over physical corruption level alpha.

This pilot removes that arbitrary reparameterization by identifying the model's
conditioning coordinate with physical corruption:

\[
\alpha=t/T.
\]

## Controlled implementation

The historical DB-CR code samples integer timesteps uniformly from
`{0, ..., T}`. We keep that sampler unchanged and use

```text
alpha = t / T
x_alpha = (1 - alpha) * clean + alpha * cloudy
conditioning = t = T * alpha
```

Thus, with `T=1000`, training is uniform over the canonical physical-alpha grid
`{0, 0.001, ..., 1}`.

This discrete implementation is intentional. Sampling continuous alpha would
also change the support of the legacy raw-t time embedding and would introduce
an avoidable confound. The canonical experiment changes the bridge-state
measure while preserving the architecture, optimizer, loss, timestep support,
and sampler implementation.

## Pilot protocol

- branch: `exp/canonical-alpha`
- command: `mscloudr.cli.pilot_train`
- schedule identity: `canonical_alpha`
- data: deterministic 10% ROI-disjoint, season/ROI-stratified pilot subset
- epochs: 25
- seed: 42
- total bridge steps: 1000
- checkpoint selection: `val_endpoint_l1`
- NFE=1 endpoint evaluation unchanged

## Comparisons

Compare against matched pilot runs:

1. Original sine parameterization
2. MR-r3 parameterization
3. Canonical-alpha uniform physical-corruption measure

The primary screening metric is endpoint validation/test performance at
`alpha=1`. The fixed-alpha sweep can then diagnose performance across the
physical bridge without schedule-dependent bridge states.
