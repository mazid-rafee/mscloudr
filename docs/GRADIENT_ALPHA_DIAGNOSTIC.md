# Fixed-alpha gradient profile

This diagnostic measures how strongly different physical bridge states drive
optimization for the existing CanonicalAlpha pilot checkpoint.

For each physical corruption level `alpha`, the same ordered, equal-size pilot
TRAIN batches are evaluated and the direct-x0 L1 loss is backpropagated without
an optimizer step.  The reported quantity is

`G_B(alpha) = ||grad_theta L_B(alpha)||_2`.

The output records the mean, population variance, standard deviation, RMS,
minimum, maximum, and coefficient of variation of the batch gradient L2 norm.
This is deliberately called **batch gradient-norm variability**; it is not the
full covariance or variance of the gradient vector.

The default diagnostic uses 256 batches per alpha.  With batch size 4 this is
1024 training samples and 11 fixed alpha levels, for 2816 forward/backward
passes. `drop_last=True` and `shuffle=False` ensure every alpha sees the same
sample batches with equal batch size.

Only the pilot training split is used. Validation and test data are not used to
design a future adaptive bridge measure.

Example smoke run:

```bash
python -m mscloudr.cli.gradient_profile \
  --checkpoint outputs/DBCR_CanonicalAlpha_pilot10_seed42/checkpoints/best_endpoint.pt \
  --alphas 0,0.5,1 \
  --max-batches 2 \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  --progress-every 1 \
  --overwrite
```

Default profile:

```bash
python -m mscloudr.cli.gradient_profile \
  --checkpoint outputs/DBCR_CanonicalAlpha_pilot10_seed42/checkpoints/best_endpoint.pt \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  --max-batches 256 \
  --progress-every 25 \
  --overwrite
```

The first analysis should compare `batch_l1.mean` and
`batch_gradient_l2.mean/std_population` across alpha.  If high-loss regions are
not the same regions that generate large or unstable gradients, the optimization
profile provides information beyond the raw difficulty curve from experiment 3.
