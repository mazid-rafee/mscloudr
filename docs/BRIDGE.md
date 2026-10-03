# Bridge mechanics

The bridge module is intentionally independent from the neural network.

For clean optical target `x0`, cloudy optical input `y`, and bridge weight
`alpha_t`:

```text
x_t = (1 - alpha_t) x0 + alpha_t y
```

The deterministic reverse update used by the historical DB-CR implementation
is:

```text
x_s = (1 - alpha_s / alpha_t) x0_hat
      + (alpha_s / alpha_t) x_t
```

## Supported schedules at this stage

Original DB-CR:

```text
alpha(t) = sin(pi t / (2T))
```

MR_r3:

```text
s = t / T
alpha(t) = (1 - exp(-3s)) / (1 - exp(-3))
```

No RiskEq, CurvedMR, SpectralMR, or other experimental schedules are migrated
at this stage.

## Randomness

Training timestep sampling requires an explicit `torch.Generator`. The
generator should be created from the experiment's `sampler_seed`; it is
separate from model initialization and DataLoader randomness.

Example:

```python
generator = make_torch_generator(seeds.sampler_seed)
t = sample_timesteps(
    batch_size,
    total_steps,
    generator=generator,
    device=device,
)
```

## NFE=1 invariant

Both supported schedules satisfy:

```text
alpha(0) = 0
alpha(T) = 1
```

Therefore for one deterministic endpoint step `T -> 0`:

```text
x_0 = x0_hat
```

for either schedule.

This means changing sine to MR_r3 cannot improve NFE=1 inference by changing
the algebra of the final reverse update. Any NFE=1 difference must arise from
training: the schedule changes which bridge states are presented to the model
for uniformly sampled timesteps (and, in the legacy model, how those states are
associated with time conditioning).

That distinction is a core control for the paper.
