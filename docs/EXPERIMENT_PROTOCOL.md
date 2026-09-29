# Experimental protocol

This document defines the minimum controls required before a result is treated as paper-grade evidence.

## 1. Separate randomness sources

Every run records three seeds:

1. `split_seed`: dataset membership only.
2. `train_seed`: model initialization, augmentation, DataLoader shuffle, and other training stochasticity.
3. `sampler_sed`: bridge-timestep or inference sampling stochasticity.

A result named `seed42` is ambiguous unless the saved config states which subsystem the seed controls.

## 2. Freeze dataset membership

Do not recreate train/validation/test membership from `len(dataset)` with an implicit `random_split` call.

Create a versioned manifest containing the exact sample IDs assigned to each split. The manifest must be validated against the dataset before training.

The historical 80/10/10 split from `ms-cloudR` may be preserved as a `legacy` protocol for reproducing old experiments, but it must not be silently presented as the official SEN12MS-CR/DB-CR benchmark protocol.

## 3. NFE=1 model selection

For experiments whose reported inference uses NFE=1, checkpoint selection should use a schedule-independent endpoint validation state:

```text
x_T = y
t = T
```

Random-t validation may still be logged diagnostically, but it must not be used to compare schedules whose alpha(t) functions induce different corruption distributions.

## 4. Metrics

Benchmark metrics must be calculated per sample and then averaged over samples. The clean implementation will match the evaluation convention used by the reference benchmark code before paper tables are produced.

Historical/legacy metrics should remain reproducible under explicit `legacy_*` names rather than being overwritten.

## 5. Baseline identity

The existing 13.59M-parameter implementation is a valuable controlled backbone for the historical Original-vs-MR experiments, but it should be described as a reimplementation until architectural fidelity to the published DB-CR model is resolved.

Any paper-faithful DB-CR implementation should live alongside, not silently replace, the historical reimplementation.

## 6. Migration rule

Experimental methods are migrated from `ms-cloudR` only after the baseline substrate is verified. Initial priority:

1. baseline reproducibility;
2. split protocol;
3. metrics;
4. baseline architecture audit;
5. Original vs MR_r3 controlled rerun;
6. RiskEq mechanism experiments.
