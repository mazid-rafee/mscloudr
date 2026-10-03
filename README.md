# mscloudr

Clean, paper-oriented research code for **multimodal cloud removal on SEN12MS-CR**.

This repository is intentionally separate from the exploratory `ms-cloudR` repository. The old repository remains the experiment archive; this repository is for audited, reproducible experiments intended for the paper.

## Current status

The active development branch is `audit/paper-baseline`.

The first milestone is to establish a trustworthy experimental substrate before migrating any experimental bridge variants:

- deterministic training and evaluation;
- explicit, saved dataset split manifests;
- publication-compatible metrics;
- a clearly identified DB-CR reimplementation baseline;
- endpoint-based checkpoint selection for NFE=1 experiments;
- tests for bridge and evaluation invariants.

No MR, RiskEq, CurvedMR, SAR-gating, or other experimental method should be added to the baseline until this milestone is complete.

## Reproducibility policy

We distinguish three different sources of randomness:

- **split seed**: determines the train/validation/test membership;
- **train seed**: controls parameter initialization and stochastic training;
- **sampler seed**: controls stochastic sampling such as bridge timesteps.

Experiments must record all three. Dataset membership should be stored as sample IDs in a manifest rather than recreated implicitly from dataset length.

## Project layout

```text
src/mscloudr/           reusable library code
tests/                  unit tests for research invariants
docs/                   experimental protocol and audit notes
configs/                experiment configs (added as baselines are migrated)
```

## Research provenance

The exploratory implementation and historical experiment branches remain in:
`mazid-rafee/ms-cloudR`.

Results from that repository are not silently rewritten here. When an experiment is migrated, its exact assumptions and compatibility with the clean baseline will be documented.
