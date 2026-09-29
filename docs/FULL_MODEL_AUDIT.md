# Full DB-CR backbone audit scaffold

The clean repository now contains a runnable full-network scaffold in
`src/mscloudr/models/dbcr.py`. It is deliberately named `AuditedDBCRNet`, not
"paper-faithful DB-CR", because several implementation details are not fully
specified by the paper.

## Source-supported topology

The default configuration uses the discrete settings stated by DB-CR:

- optical input: 13 channels;
- SAR input: 2 channels;
- widths: `[22, 44, 88, 176]`;
- encoder NAFBlocks in both branches: `[1, 1, 1, 28]`;
- decoder NAFBlocks: `[1, 1, 1, 1]`;
- SFBlock heads: `[1, 1, 2, 4]`;
- optical U-Net plus SAR encoder;
- SFBlock fusion at every encoder depth.

## Explicit assumptions

1. **Stems:** 1x1 convolutions, following the implementation-details text.
2. **Output head:** 3x3 convolution, following Figure 3's green-convolution
   legend.
3. **Downsampling:** 2x2 stride-2 convolutions, preserved from the historical
   implementation because the paper does not specify the operator.
4. **Upsampling:** 2x2 transposed convolutions, preserved for the same reason.
5. **Branch separation:** optical and SAR use distinct downsampling weights by
   default because the paper describes dedicated feature-extraction branches.
6. **Time conditioning:** historical stage-level optical additive bias is
   retained only as `legacy_stage_bias`; it is not claimed to reconstruct the
   paper's unspecified "time-embedded NAFBlock" implementation.

## Parameter audit

Under those explicit defaults the scaffold is expected to have:

```text
13,566,113 trainable parameters
```

For comparison:

```text
historical ms-cloudR implementation: 13,588,641
DB-CR paper:                       18,060,000
current remaining gap:              4,493,887
```

Separate optical/SAR downsamplers account for only 81,620 parameters relative
to an otherwise identical shared-downsampler scaffold. Correcting the known
block-level issues therefore does not explain the roughly 4.49M remaining gap.

Run:

```bash
python -m mscloudr.cli.audit_model
```

At this stage the network is for architecture validation only. Do not start a
paper baseline training run until the remaining architecture ambiguities have
been explicitly resolved or frozen as the project's reimplementation protocol.
