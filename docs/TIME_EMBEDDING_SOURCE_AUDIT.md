# DB-CR time-embedding source-resolution audit

Date: 2026-09-29

This audit asks whether public DB-CR sources resolve the remaining time-
embedding ambiguities well enough to claim an exact architecture.

## Sources checked

1. Published DB-CR paper / MERL technical report TR2025-138.
2. U.S. patent application US20260289738A1, published 2026-09-24.
3. MERL publication page for TR2025-138.
4. Yuyang Hu's publication page.
5. Public GitHub repository search for DB-CR, the paper title, authors, and
   arXiv identifier.

## What is source-resolved

The patent discloses details absent from the paper:

- the original bridge state is
  `alpha_t = sin(pi t / (2T))`;
- `alpha_t`, not raw `t`, is converted to a sinusoidal positional embedding;
- the time module contains three Linear layers with SimpleGates between
  adjacent Linear layers;
- the final output is reshaped into four C-dimensional vectors:
  `Scale_conv`, `Shift_conv`, `Scale_FFN`, and `Shift_FFN`;
- these vectors modulate normalized features in the MBConv and FFN paths;
- LayerNorm is described as using the mean and standard deviation of all
  elements in each feature map;
- beta and gamma are learnable scalar residual weights;
- both the SAR encoder and the optical U-Net encoder/decoder are described as
  using time-embedded NAFBlocks.

The paper independently fixes the discrete backbone settings:

- widths `[22, 44, 88, 176]`;
- encoder NAFBlock counts `[1, 1, 1, 28]` in both branches;
- optical decoder NAFBlock counts `[1, 1, 1, 1]`;
- SFBlock heads `[1, 1, 2, 4]`;
- 1x1 input stems for optical and SAR;
- SFBlock MLP hidden dimension twice its input dimension.

## What remains unresolved

None of the checked public sources specifies:

- the dimension of the sinusoidal positional embedding;
- the hidden width(s) of the three-Linear time MLP;
- whether the time MLP is instantiated independently for every NAFBlock,
  shared per stage, shared per feature width, or shared globally;
- the exact affine parameterization of the patent-described LayerNorm;
- a public official DB-CR implementation from which those choices can be
  recovered.

The patent text describes the structure of the time embedding but gives no
numeric dimensions beyond the final four C-vectors. Searching its full text
also yields no statement that the time-embedding module is shared.

The MERL publication page links the paper/PDF and arXiv version but does not
list a code repository. Yuyang Hu's publication page lists DOI, arXiv, and
BibTeX for DB-CR but no project/code link. Public repository searches did not
locate an official implementation.

## Parameter count is not sufficient to recover the missing choices

DB-CR reports 18.06M parameters. With the current diagnostic assumptions
(positional_dim=128 and hidden_dim=128), the patent-guided audit gives:

- independent time MLP per NAFBlock: about 19.54M parameters;
- one time MLP per unique feature width: about 10.20M parameters.

Many unreported combinations of hidden width and sharing policy can land near
18.06M. Therefore the published parameter count does not uniquely identify the
implementation. Choosing dimensions solely to reproduce 18.06M would be
parameter fitting, not source-based reconstruction.

## Architecture policy

The clean repository should keep two identities separate:

1. **legacy_dbcr**: the historical ~13.59M implementation used for the existing
   Original/MR/RiskEq experimental lineage. It is the correct provenance model
   for reproducing those prior comparisons.
2. **patent_guided_dbcr**: a source-guided implementation using the newly
   disclosed alpha-based scale/shift time conditioning. Any dimensions or
   sharing choices not stated by the sources must be explicit configuration
   assumptions and must not be described as an exact reproduction.

For the paper's core bridge-schedule experiments, architecture should remain
fixed within each controlled comparison. The project should not alter hidden
time-embedding dimensions merely to match the published 18.06M count.

## Immediate consequence

The source audit is complete: there is no public evidence currently sufficient
to resolve the time-embedding width or sharing policy uniquely.

The next implementation step should therefore be to freeze explicit model
identities/configurations rather than continue reverse-engineering the reported
parameter count.
