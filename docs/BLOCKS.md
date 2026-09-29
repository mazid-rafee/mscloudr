# DB-CR building-block reconstruction

This stage reconstructs only the two reusable blocks needed by the published
DB-CR backbone. It does **not** yet assemble or train the complete network.

## NAFBlock

The implementation in `src/mscloudr/models/blocks.py` follows the official
NAFNet block from `megvii-research/NAFNet` at commit
`2b4af71ebe098a92a75910c233a3965a3e93ede4` and is consistent with the DB-CR
paper's description of LayerNorm before both the MBConv and FFN modules.

Compared with the historical `ms-cloudR` implementation, the main correction
is normalization semantics:

- historical: `GroupNorm(1, C)`;
- reconstructed: channel-wise `LayerNorm2d` at every spatial position.

The affine parameter count is unchanged by this correction because both
normalizers learn two length-C vectors. For example, the reconstructed NAFBlock
has 4,114 trainable parameters at C=22 and 222,640 at C=176.

DB-CR calls these blocks *time-embedded NAFBlocks*, but neither the text nor
Figure 3 specifies the exact time-injection operator or its placement inside the
block. The clean implementation therefore does not invent one at this stage.

## SFBlock

The implementation follows DB-CR Figure 3(b) and Eqs. (15)-(16): separate
Norm->1x1 paths generate optical Q and SAR K/V; Q^T K produces channel-wise
attention; the attended SAR feature is added to the optical residual; a
residual two-layer GELU MLP follows; multiple heads are concatenated; and a
final 1x1 projection combines the heads.

The paper does not explicitly spell out whether the MLP parameters are shared
across heads or how the symbol `c` maps to total-vs-per-head channels. The clean
implementation makes the standard explicit interpretation that total feature
width is partitioned evenly across heads and each head has its own MLP. This
assumption is documented so it can be changed later without pretending it was
fully specified by the paper.

### Parameter audit

| Level width | Heads | historical SFBlock | reconstructed SFBlock |
|---:|---:|---:|---:|
| 22 | 1 | 4,026 | 4,158 |
| 44 | 1 | 15,796 | 16,060 |
| 88 | 2 | 62,568 | 47,608 |
| 176 | 4 | 249,040 | 157,168 |
| **Total** | | **331,430** | **224,994** |

Under this explicit per-head interpretation, correcting the SFBlock does **not**
explain the historical full-model parameter gap (13.59M vs the 18.06M reported
by DB-CR); it makes the block set smaller at deeper multi-head levels. The
remaining gap must therefore be investigated at the network-integration level,
especially time embedding, branch-specific down/up sampling, and any omitted
paper implementation details.

## Tests

`tests/test_dbcr_blocks.py` checks LayerNorm semantics, exact NAFBlock identity
at initialization, finite gradients, DB-CR channel/head combinations, SFBlock
residual behavior, parameter counts, shape preservation, and finite outputs.
