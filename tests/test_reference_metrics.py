import math

import torch

from mscloudr.metrics.reference import (
    ReferenceMetricAccumulator,
    reference_batch_metrics,
    reference_image_metrics,
    reference_ssim,
)


def test_identity_metrics():
    x = torch.linspace(0.01, 1.0, steps=13 * 16 * 16).reshape(1, 13, 16, 16)
    metrics = reference_image_metrics(x, x)

    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert math.isinf(metrics["PSNR"]) and metrics["PSNR"] > 0
    assert abs(metrics["SAM"]) < 1e-5
    assert abs(metrics["SSIM"] - 1.0) < 1e-6


def test_psnr_matches_reference_formula():
    target = torch.zeros(1, 13, 8, 8)
    pred = torch.full_like(target, 0.1)
    metrics = reference_image_metrics(target, pred)

    assert abs(metrics["RMSE"] - 0.1) < 1e-6
    assert abs(metrics["MAE"] - 0.1) < 1e-6
    assert abs(metrics["PSNR"] - 20.0) < 1e-5


def test_ssim_is_windowed_not_global():
    target = torch.zeros(1, 1, 32, 32)
    pred = target.clone()
    pred[:, :, 8:24, 8:24] = 1.0

    score = reference_ssim(target, pred)
    assert 0.0 <= float(score) < 1.0


def test_batch_metrics_are_computed_per_image():
    target = torch.zeros(2, 13, 8, 8)
    pred = target.clone()
    pred[0] += 0.1
    pred[1] += 0.2

    metrics = reference_batch_metrics(target, pred)

    assert len(metrics) == 2
    assert abs(metrics[0]["MAE"] - 0.1) < 1e-6
    assert abs(metrics[1]["MAE"] - 0.2) < 1e-6
    assert abs(metrics[0]["PSNR"] - 20.0) < 1e-5
    assert abs(metrics[1]["PSNR"] - (20.0 * math.log10(5.0))) < 1e-5


def test_accumulator_weights_images_not_batches():
    target_a = torch.zeros(1, 13, 8, 8)
    pred_a = torch.full_like(target_a, 0.1)

    target_b = torch.zeros(2, 13, 8, 8)
    pred_b = torch.empty_like(target_b)
    pred_b[0].fill_(0.2)
    pred_b[1].fill_(0.4)

    accumulator = ReferenceMetricAccumulator()
    accumulator.update_batch(target_a, pred_a)
    accumulator.update_batch(target_b, pred_b)
    result = accumulator.compute()

    assert abs(result["MAE"] - ((0.1 + 0.2 + 0.4) / 3.0)) < 1e-6
    assert accumulator.metric_counts()["MAE"] == 3
