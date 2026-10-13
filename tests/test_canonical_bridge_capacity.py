"""Capacity-only scaling regression tests for CanonicalBridge."""

import pytest
import torch

from mscloudr.cli.eval_canonical_bridge import _checkpoint_widths
from mscloudr.cli.train_canonical_bridge import (
    _resume_identity,
    _validate_experiment,
    _validate_resume_metadata,
    build_parser,
)
from mscloudr.models import CanonicalBridgeNet, count_canonical_bridge_parameters

BASE = (32, 64, 128, 256)
MEDIUM = (48, 96, 192, 384)


@pytest.mark.parametrize("widths,expected", [(BASE, 7653037), (MEDIUM, 16947773)])
def test_parameter_count(widths, expected):
    model = CanonicalBridgeNet(widths=widths)
    assert count_canonical_bridge_parameters(model) == expected


def test_medium_forward_backward():
    model = CanonicalBridgeNet(widths=MEDIUM)
    x = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    c = torch.tensor([1000.0])
    out = model(x, c, sar)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert model.opt_stem.weight.grad is not None
    assert torch.isfinite(model.opt_stem.weight.grad).all()


def test_width_cli_defaults_and_override():
    parser = build_parser()
    assert tuple(parser.parse_args(["--run-name", "base"]).widths) == BASE
    args = parser.parse_args(["--run-name", "medium", "--widths", "48", "96", "192", "384"])
    assert tuple(args.widths) == MEDIUM
    assert _validate_experiment(args) == 1.0


def test_invalid_widths_rejected():
    args = build_parser().parse_args(["--run-name", "bad", "--widths", "48", "96", "0", "384"])
    with pytest.raises(ValueError, match="four positive"):
        _validate_experiment(args)


def test_evaluation_checkpoint_architecture_roundtrip():
    assert _checkpoint_widths({}) == BASE  # older baseline checkpoint
    assert _checkpoint_widths({"architecture_widths": list(MEDIUM)}) == MEDIUM
    for widths in (BASE, MEDIUM):
        model = CanonicalBridgeNet(widths=widths)
        loaded = CanonicalBridgeNet(widths=_checkpoint_widths({"architecture_widths": list(widths)}))
        loaded.load_state_dict(model.state_dict(), strict=True)


def test_invalid_checkpoint_widths_rejected():
    with pytest.raises(ValueError, match="architecture_widths"):
        _checkpoint_widths({"architecture_widths": [48, 96, -1, 384]})


def test_resume_detects_capacity_change():
    shared = {
        "model_identity": "canonical_bridge_net", "trainable_parameters": 7653037,
        "bridge_geometry": "straight_linear", "schedule_name": "canonical_alpha",
        "training_bridge_measure": "uniform_discrete_alpha_grid",
        "beta_a": 1.0, "beta_b": 1.0, "conditioning_mode": "mr_inverse",
        "conditioning_definition": "mr_inverse", "total_steps": 1000,
        "batch_size": 4, "num_workers": 4, "learning_rate": 5e-5,
        "split_protocol": "reference", "split_audit": {},
        "reproducibility": {"seeds": {"train_seed": 42, "sampler_seed": 42}, "deterministic_algorithms": False},
    }
    baseline = {**shared, "architecture_widths": list(BASE)}
    medium = {**shared, "architecture_widths": list(MEDIUM)}
    assert _resume_identity(baseline)["architecture_widths"] == list(BASE)
    with pytest.raises(ValueError, match="architecture_widths"):
        _validate_resume_metadata(baseline, medium)
