# Generalization-aware bridge-measure diagnostic

This experiment asks whether the training measure should emphasize bridge
regions that are not merely hard on the training set, but specifically fragile
under geographic/seasonal generalization.

The diagnostic evaluates the CanonicalAlpha pilot checkpoint on the fixed pilot
validation split. Validation ROIs are disjoint from training ROIs under the
UnCRtainTS split, and the final test split is not used for sampler design.

For every fixed physical alpha in `{0.0, 0.1, ..., 1.0}`, compute per-sample L1
and aggregate by season-scoped ROI group. The main summaries are:

- ordinary validation sample mean L1;
- equal-ROI mean L1;
- ROI-to-ROI standard deviation;
- worst-ROI mean L1;
- group CVaR (default: average of worst 20% ROI groups);
- equal-group seasonal means.

If `difficulty_profile_train.json` from experiment #3 is supplied, the tool also
records validation-minus-training gaps at each alpha. This allows us to
separate high absolute difficulty from corruption levels where held-out ROI
performance degrades disproportionately.

Potential future sampler signals include

`q(alpha) proportional to CVaR_group(alpha)^gamma`

or a clipped/regularized function of the held-out generalization gap. We do not
choose that rule before observing the profile.

Example smoke run:

```bash
python -m mscloudr.cli.generalization_profile \
  --checkpoint outputs/DBCR_CanonicalAlpha_pilot10_seed42/checkpoints/best_endpoint.pt \
  --alphas 0,0.5,1 \
  --max-batches 2 \
  --device cuda \
  --output outputs/DBCR_CanonicalAlpha_pilot10_seed42/generalization_profile_smoke.json \
  --overwrite
```

After experiment #3 finishes, add:

```bash
--train-profile outputs/DBCR_CanonicalAlpha_pilot10_seed42/difficulty_profile_train.json
```

for explicit validation-minus-training gaps.
