"""Standalone SAR-intervention and SFBlock snapshot tests (no SEN12MS files)."""

from __future__ import annotations

import pytest
import torch

from mscloudr.cli.diagnose_dbcr_residuals import evenly_spaced_indices
from mscloudr.models.patent_blocks import PatentTimeEmbeddingConfig
from mscloudr.models.source_guided_dbcr import SourceGuidedDBCRConfig, SourceGuidedDBCRNet
from mscloudr.sar_sensitivity import (
    SFBlockSnapshot, per_sample_l1, per_sample_rms,
    rotate_other_sample_indices, summarize_samples, tensor_change,
    paired_reconstruction_statistics,
)


def _tiny_model():
    cfg = SourceGuidedDBCRConfig(
        widths=(8, 16, 32, 64),
        encoder_blocks=(1, 1, 1, 2),
        decoder_blocks=(1, 1, 1, 1),
        sf_heads=(1, 1, 2, 4),
        time_embedding=PatentTimeEmbeddingConfig(positional_dim=8, hidden_dim=8),
    )
    return SourceGuidedDBCRNet(cfg).eval()


def test_global_sar_derangement_is_unique_and_has_no_fixed_points():
    for count in (2, 3, 20):
        p = rotate_other_sample_indices(count)
        assert sorted(p) == list(range(count))
        assert all(p[i] != i for i in range(count))
    with pytest.raises(ValueError):
        rotate_other_sample_indices(1)


def test_metrics_are_per_sample_and_correct():
    baseline = torch.tensor([[[[3., 4.]]], [[[0., 5.]]]])
    intervention = baseline * 2
    delta = tensor_change(baseline, intervention)
    assert delta["l1"].shape == (2,)
    assert delta["l1"].tolist() == pytest.approx([3.5, 2.5])
    assert delta["relative_l2"].tolist() == pytest.approx([1., 1.])
    assert delta["rms"].tolist() == pytest.approx([
        5 / (2 ** 0.5), 5 / (2 ** 0.5)
    ])
    assert per_sample_l1(baseline, baseline).tolist() == [0., 0.]
    assert per_sample_rms(baseline).tolist() == pytest.approx([
        5 / (2 ** 0.5), 5 / (2 ** 0.5)
    ])


def test_snapshot_is_noninvasive_and_captures_all_fusion_stages():
    torch.manual_seed(42)
    model = _tiny_model()
    snapshot = SFBlockSnapshot(model)
    cloudy = torch.randn(2, 13, 16, 16)
    sar = torch.randn(2, 2, 16, 16)
    t = torch.tensor([1000., 1000.])
    with torch.inference_mode():
        expected = model(cloudy, t, sar)
        pred, normal_maps = snapshot.run(model, cloudy, t, sar)
        perturbed, altered_maps = snapshot.run(
            model, cloudy, t, torch.zeros_like(sar)
        )
    assert torch.equal(expected, pred)
    assert len(normal_maps) == len(altered_maps) == 4
    assert set(normal_maps) == set(altered_maps) == {
        "fusion_blocks.0", "fusion_blocks.1",
        "fusion_blocks.2", "fusion_blocks.3",
    }
    assert all(torch.isfinite(x).all() for x in normal_maps.values())
    assert all(not module._forward_hooks for module in snapshot.blocks.values())
    # Since both modalities feed the fusion blocks, zero-SAR perturbations
    # must not change the *shape* of the fused feature maps.
    for stage in normal_maps:
        assert normal_maps[stage].shape == altered_maps[stage].shape
        assert tensor_change(
            normal_maps[stage], altered_maps[stage]
        )["relative_l2"].shape == (2,)
    assert not snapshot._handles
    assert torch.isfinite(perturbed).all()


def test_snapshot_removes_hooks_even_if_model_call_fails():
    model = _tiny_model()
    capture = SFBlockSnapshot(model)
    with pytest.raises(ValueError):
        with torch.inference_mode():
            capture.run(
                model,
                torch.randn(1, 13, 16, 16),
                torch.tensor([1000.]),
                torch.randn(1, 3, 16, 16),
            )
    assert not capture._handles
    assert all(not layer._forward_hooks for layer in capture.blocks.values())


def test_fixed_validation_selection_and_aggregator():
    indices = evenly_spaced_indices(7176, 20)
    shuffled_sources = rotate_other_sample_indices(20)
    assert len(indices) == len(set(indices)) == 20
    assert all(indices[i] != indices[shuffled_sources[i]] for i in range(20))
    assert summarize_samples({"x": [1., 3.], "y": [4., 8.]}) == {
        "x_mean": 2., "y_mean": 6.
    }
    with pytest.raises(ValueError):
        summarize_samples({"x": [1.], "y": [2., 3.]})



def test_paired_analysis_improvement_and_roi_uncertainty_deterministic():
    normal = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    shuffled = [0.023, 0.032, 0.045, 0.047, 0.065, 0.077]
    roi_groups = ["roi_a", "roi_a", "roi_b", "roi_b", "roi_c", "roi_c"]
    results = paired_reconstruction_statistics(
        normal, shuffled, roi_groups,
        bootstrap_seed=42, bootstrap_repetitions=200,
    )
    again = paired_reconstruction_statistics(
        normal, shuffled, roi_groups,
        bootstrap_seed=42, bootstrap_repetitions=200,
    )
    assert results == again
    assert results["num_samples"] == 6
    assert results["num_roi_groups"] == 3
    assert results["normal_better_count"] == 5
    assert results["shuffled_better_count"] == 1
    assert results["tie_count"] == 0
    assert results["fraction_normal_better"] == pytest.approx(5 / 6)
    assert results["mean_delta_l1"] == pytest.approx(0.019 / 6)
    assert results["roi_group_sample_counts"] == {
        "roi_a": 2, "roi_b": 2, "roi_c": 2
    }
    assert results["roi_groups_favoring_normal"] == 3
    for key in ("patch_bootstrap_95ci", "roi_cluster_bootstrap_95ci"):
        lower, upper = results[key]
        assert lower <= results["mean_delta_l1"] <= upper


def test_paired_analysis_rejects_bad_alignment_and_works_with_single_roi():
    with pytest.raises(ValueError):
        paired_reconstruction_statistics([.1, .2], [.2], ["a", "b"])
    with pytest.raises(ValueError):
        paired_reconstruction_statistics([.1, .2], [.2, .3], ["a", ""])
    with pytest.raises(ValueError):
        paired_reconstruction_statistics(
            [.1, float("nan")], [.2, .3], ["a", "a"]
        )
    result = paired_reconstruction_statistics(
        [.1, .2], [.2, .3], ["one", "one"], bootstrap_repetitions=100
    )
    assert result["num_roi_groups"] == 1
    assert result["mean_delta_l1"] == pytest.approx(.1)
    assert result["roi_cluster_bootstrap_95ci"] == pytest.approx([.1, .1])
