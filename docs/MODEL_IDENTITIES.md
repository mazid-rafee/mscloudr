# Frozen model identities

The clean repository intentionally separates architecture provenance from
paper-faithfulness.

## legacy_dbcr

Implementation:

`src/mscloudr/models/legacy_dbcr.py`

Purpose:

- canonical backbone for reproducing the historical Original, MR_r3, RiskEq,
  CurvedMR, SAR-intervention, and related experiments that were run from the
  `mazid-rafee/ms-cloudR` codebase;
- exact architecture/state-dict naming migration of
  `mazid-rafee/ms-cloudR/src/models/dbcr.py`;
- checkpoint-compatible with historical payloads that saved
  `model_state = model.state_dict()`.

Frozen provenance:

- old repository main commit inspected:
  `e711c1eb7149197bf1cf1070f382b5bd878f1f02`;
- old `src/models/dbcr.py` blob:
  `2e7493d2b9bafc59899df1610ff3e0337b556eaa`;
- exact trainable parameter count:
  **13,588,641**.

The migration deliberately retains historical choices, including:

- `GroupNorm(1, C)` in NAFBlocks;
- stage-level additive time bias on the optical branch;
- no explicit time conditioning in the SAR branch;
- one shared downsampling module used sequentially for optical and SAR;
- historical SFBlock ordering and lack of pre-Q/K/V normalization;
- 1x1 output head.

These are not reinterpreted as exact published DB-CR details. They are frozen
because changing them would destroy architectural continuity with the existing
experimental lineage.

## patent_guided_dbcr

Building blocks:

`src/mscloudr/models/patent_blocks.py`

Purpose:

- architecture study using implementation details disclosed in
  US20260289738A1;
- future robustness/architecture-validation experiments;
- not used as a drop-in replacement for the historical bridge-schedule
  comparison unless an explicit new experiment says so.

Unresolved public-source details include the time-embedding hidden dimensions
and sharing policy, so this identity must not be described as an exact
reproduction.

## audited_dbcr

Implementation:

`src/mscloudr/models/dbcr.py`

Purpose:

- intermediate architecture-audit scaffold;
- keeps paper-supported topology separate from unresolved assumptions;
- not the canonical bridge-schedule experiment backbone.

## Experimental rule

For the main bridge-schedule study, keep the architecture fixed to
`legacy_dbcr` while varying the bridge schedule. This preserves the scientific
meaning of the existing Original-vs-MR lineage and prevents architecture changes
from becoming a confound.

New architecture variants must use a new model identity and a separate
controlled experiment.
