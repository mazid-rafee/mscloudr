# DB-CR patent architecture update

A U.S. patent application for DB-CR, **US20260289738A1**, was published on
2026-09-24. It discloses implementation details that are not present in the
paper and materially changes the architecture audit.

## Newly disclosed details

The patent states that the diffusion timestep is first converted to the
original sine bridge weight

```text
alpha_t = sin(pi t / (2T))
```

and that **alpha_t**, rather than raw `t`, is sinusoidally position-embedded.
The time embedding then passes through three Linear layers with SimpleGates
between adjacent Linear layers. Its output is reshaped into four vectors of
dimension `C`:

```text
Scale_conv
Shift_conv
Scale_FFN
Shift_FFN
```

The time-conditioned NAFBlock applies those vectors as scale/shift modulation
to normalized features before its MBConv and FFN paths.

The patent also states that the normalization uses the mean and standard
deviation of **all elements in each feature map**. For NCHW tensors this means
normalizing over C, H, and W for each sample. This is much closer to the
historical `GroupNorm(1, C)` reduction axes than to the channel-only
`LayerNorm2d` used in the first clean-room block audit.

Finally, beta and gamma are described as learnable scalar residual weights.

## Important unresolved details

The disclosure still does **not** specify:

- the sinusoidal positional-embedding dimension;
- the hidden width of the three-Linear time MLP;
- whether each NAFBlock owns an independent time MLP or whether time MLPs are
  shared;
- whether affine normalization parameters are per-channel;
- every tensor width inside the time MLP.

These unknowns prevent a unique reconstruction of the published 18.06M
parameter count.

## Clean implementation

`src/mscloudr/models/patent_blocks.py` now implements only what can be
supported directly, while exposing unresolved choices explicitly.

The implementation provides:

- `sine_bridge_alpha`;
- `PatentFeatureMapLayerNorm`;
- `PatentTimeEmbedding`;
- `PatentTimeEmbeddedNAFBlock`;
- `PatentSimplifiedChannelAttention`.

The default time dimensions are 128 only because the historical ms-cloudR
implementation used `time_dim=128`; this is **not** a patent-derived value.

The block also exposes `PatentTimeEmbedding.from_alpha(...)`. This is
important for the later MR/RiskEq experiments: if the bridge schedule changes,
conditioning can follow the actual alpha state rather than silently continuing
to encode raw timestep.

## Parameter-count audit

Run:

```bash
python -m mscloudr.cli.audit_patent_time
```

The command reports multiple scenarios instead of pretending there is a single
recovered count.

With the diagnostic defaults `positional_dim=128` and `hidden_dim=128`:

- one independent time MLP per NAFBlock is expected to produce a model around
  19.54M parameters if the rest of the current audited scaffold is held fixed;
- one time MLP shared per unique feature width produces a much smaller model,
  around 10.20M.

The published DB-CR count, 18.06M, lies between these scenarios. That fact alone
is **not** evidence for a particular sharing policy or hidden dimension.

Do not parameter-fit the architecture to 18.06M. The goal is source-faithful
reconstruction, not reverse engineering a count.
