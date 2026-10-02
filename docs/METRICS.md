# Metric audit

The paper-facing SEN12MS-CR reconstruction metrics in this repository follow the
public UnCRtainTS implementation at commit
`5e1f1b58e993645e765b64e10b6e9c7ff828b36f`.

For each **individual image**, UnCRtainTS computes:

- RMSE: square root of mean squared error over all bands and pixels;
- MAE: mean absolute error over all bands and pixels;
- PSNR: `20 * log10(1 / RMSE)`, assuming data in [0, 1];
- SAM: mean spectral angle in degrees over pixels;
- SSIM: 11x11 Gaussian-window SSIM (sigma 1.5), averaged over channels and
  spatial positions.

The reference evaluation loop explicitly iterates through the batch and calls
the metric function once per image.  Dataset metrics are then arithmetic means
over images.

This is materially different from the historical `ms-cloudR` evaluator,
which computed several metrics over whole batches and used a global-statistics
SSIM approximation.  Historical values therefore remain useful for controlled
comparisons inside that repository but are not treated as benchmark-compatible
paper metrics here.

The implementation is in `src/mscloudr/metrics/reference.py`.
