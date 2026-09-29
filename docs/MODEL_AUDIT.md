# DB-CR architecture audit

This audit compares the historical `mazid-rafee/ms-cloudR` implementation
(`src/models/dbcr.py`) against the architecture described in the DB-CR paper.

No model code is migrated into the clean repository until the fidelity risks
below are resolved explicitly.

## Confirmed matches

The historical implementation matches several published high-level settings:

- two input modalities: 13-channel optical and 2-channel SAR;
- separate optical and SAR stems using 1x1 convolutions;
- channel widths: `[22, 44, 88, 176]`;
- encoder NAFBlock counts: `[1, 1, 1, 28]`;
- optical decoder NAFBlock counts: `[1, 1, 1, 1]`;
- SFBlock head counts: `[1, 1, 2, 4]`;
- SFBlock MLP hidden width is twice the input width;
- optical queries with SAR keys/values for channel-wise cross-modal attention;
- direct prediction of the clean optical image.

These points are sufficiently specified by the paper and should be preserved.

## Fidelity risks that must not be silently ignored

### 1. NAFBlock normalization

The paper states that LayerNorm is applied before both the MBConv and FFN
parts of each NAFBlock.

The historical implementation instead uses:

```python
nn.GroupNorm(1, channels)
```

for both normalization sites.

`GroupNorm(1, C)` is not equivalent to a 2D channel-wise LayerNorm: it
normalizes across channel and spatial dimensions for each sample, whereas the
usual NAFNet LayerNorm2d normalizes across channels at each spatial position.

**Status:** confirmed implementation mismatch.

### 2. SFBlock normalization

The paper states that optical and SAR features pass through normalization and
1x1 convolution before Q/K/V reshaping.

The historical SFBlock contains the 1x1 Q/K/V convolutions but no
normalization layers.

**Status:** confirmed implementation mismatch.

### 3. SFBlock projection/residual ordering

The paper describes, per attention head:

1. cross-modal attention;
2. add attended features to optical input;
3. residual MLP;
4. concatenate head outputs;
5. final 1x1 projection.

The historical implementation concatenates attention-head outputs implicitly,
applies the final projection first, and then performs the optical residual and
MLP:

```python
out = x_opt + self.proj(fused)
out = out + self.mlp(out)
```

**Status:** confirmed ordering mismatch.

### 4. Time embedding placement

The paper repeatedly describes the backbone as using *time-embedded
NAFBlocks*. The historical implementation computes one time embedding per
encoder level and adds it once to the optical feature map before the entire
sequential stack of NAFBlocks.

At the deepest level this means a single time bias precedes 28 NAFBlocks,
rather than each NAFBlock receiving time conditioning directly.

The SAR branch receives no explicit time embedding in the historical code.

The paper does not provide enough implementation detail to reconstruct the
exact time-conditioning operator from text alone.

**Status:** fidelity ambiguity; do not guess silently.

### 5. Shared downsampling weights across modalities

The paper describes dedicated optical-restoration and SAR-feature-extraction
branches. The historical implementation uses the same `self.downs[i]`
convolution module for both optical and SAR features at every encoder
transition.

This forces the two modalities to share downsampling weights.

The paper text does not explicitly state whether downsampling kernels are
shared, so this cannot be called a definitive contradiction from text alone,
but it is a meaningful fidelity risk relative to the stated dedicated-branch
design.

**Status:** fidelity ambiguity requiring an explicit choice.

## Parameter-count discrepancy

The historical implementation has approximately **13.59M parameters**.
DB-CR reports **18.06M parameters**.

The historical model is therefore about 24.8% smaller. This is not inherently
bad and may be useful as a lean controlled backbone, but it is evidence that
the implementation should not be labeled an exact reproduction of published
DB-CR.

## Migration policy

The clean repository will keep two identities separate if necessary:

1. **legacy/reimplemented DB-CR backbone** — preserves the 13.59M historical
   architecture used for prior Original-vs-MR experiments;
2. **paper-fidelity DB-CR backbone** — only after the unresolved architectural
   details have been reconstructed and validated.

We will not silently replace one with the other, because doing so would break
comparability with historical experiments.
