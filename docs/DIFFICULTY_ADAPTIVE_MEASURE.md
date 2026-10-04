# Difficulty-adaptive bridge measure: diagnostic stage

This branch starts the difficulty-adaptive measure experiment with a diagnostic,
not a new training sampler.

The diagnostic evaluates the existing CanonicalAlpha pilot checkpoint on the
pilot **training split only** at fixed physical corruption levels
`alpha = 0.0, 0.1, ..., 1.0`.

For each alpha,

`x_alpha = (1-alpha) * x0 + alpha * y`

and the model is conditioned with the physical coordinate

`c = T * alpha`.

The primary curve is

`E(alpha) = mean_train L1(R_theta(x_alpha, T*alpha, z), x0)`.

Validation and test splits are deliberately disabled for this command so that a
future adaptive sampler is not designed from held-out performance.

Example:

```bash
python -m mscloudr.cli.difficulty_profile \
  --checkpoint outputs/DBCR_CanonicalAlpha_pilot10_seed42/checkpoints/best_endpoint.pt \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  --progress-every 100
```

The default output is `difficulty_profile_train.json` next to the checkpoint's
run directory. It contains the full fixed-alpha reference metrics; L1 is the
primary difficulty signal used to decide the later adaptive sampling rule.
