"""Endpoint-mixture x CNN/NAF CanonicalBridge regression tests (CPU only)."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from mscloudr.bridge import get_bridge_schedule, sample_timesteps
from mscloudr.cli.eval_canonical_bridge import _checkpoint_backbone
from mscloudr.cli.train_canonical_bridge import (
    _resume_identity,
    _validate_experiment,
    _validate_resume_metadata,
    build_parser,
)
from mscloudr.models import CanonicalBridgeNet, count_canonical_bridge_parameters
from mscloudr.models.blocks import NAFBlock
from mscloudr.models.canonical_bridge import ResidualConvBlock
from mscloudr.training import (
    MR_INVERSE_CONDITIONING,
    random_t_bridge_step,
    random_t_validation_step,
    training_step,
)


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.x_alpha = None
        self.conditioning = None

    def forward(self, x_alpha, conditioning, sar):
        self.x_alpha = x_alpha.detach().clone()
        self.conditioning = conditioning.detach().clone()
        return self.scale * x_alpha


def sample_batch(batch_size=128):
    torch.manual_seed(100)
    return {
        "target": torch.rand(batch_size, 13, 4, 4),
        "cloudy": torch.rand(batch_size, 13, 4, 4),
        "sar": torch.rand(batch_size, 2, 4, 4),
    }


def generator(seed=123):
    return torch.Generator(device="cpu").manual_seed(seed)


@pytest.mark.parametrize("backbone", ["cnn", "naf"])
def test_forward_backward_and_optical_only_substitution(backbone):
    model = CanonicalBridgeNet(
        widths=(8, 16, 32, 64),
        backbone_variant=backbone,
    )
    block_type = NAFBlock if backbone == "naf" else ResidualConvBlock
    assert all(isinstance(b, block_type) for b in model.opt_blocks)
    assert all(isinstance(b, block_type) for b in model.mid_blocks)
    assert all(isinstance(b, block_type) for b in model.dec_blocks)
    assert all(isinstance(b, ResidualConvBlock) for b in model.sar_blocks)
    assert model.backbone_variant == backbone
    x = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    output = model(x, torch.tensor([1000.0]), sar)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.abs().mean().backward()
    assert model.opt_stem.weight.grad is not None
    assert torch.isfinite(model.opt_stem.weight.grad).all()


def test_default_cnn_preserves_parameter_count_and_legacy_weights():
    old = CanonicalBridgeNet()
    explicit = CanonicalBridgeNet(backbone_variant="cnn")
    assert count_canonical_bridge_parameters(old) == 7653037
    explicit.load_state_dict(old.state_dict(), strict=True)
    assert list(old.state_dict()) == list(explicit.state_dict())


def test_naf_checkpoint_strict_roundtrip():
    source = CanonicalBridgeNet(
        widths=(8, 16, 32, 64), backbone_variant="naf"
    )
    metadata = {"architecture_widths": [8, 16, 32, 64], "backbone_variant": "naf"}
    target = CanonicalBridgeNet(
        widths=tuple(metadata["architecture_widths"]),
        backbone_variant=_checkpoint_backbone(metadata),
    )
    target.load_state_dict(source.state_dict(), strict=True)
    assert count_canonical_bridge_parameters(target) == count_canonical_bridge_parameters(source)


def test_zero_probability_is_identical_to_historical_sampler():
    batch = sample_batch(8)
    schedule = get_bridge_schedule("canonical_alpha")
    expected = sample_timesteps(
        8, 1000, generator=generator(42), device="cpu"
    )
    first = training_step(
        RecordingModel(), batch, schedule=schedule,
        total_steps=1000, sampler_generator=generator(42),
        conditioning_mode=MR_INVERSE_CONDITIONING,
    )
    second = training_step(
        RecordingModel(), batch, schedule=schedule,
        total_steps=1000, sampler_generator=generator(42),
        conditioning_mode=MR_INVERSE_CONDITIONING,
        endpoint_probability=0.0,
    )
    assert torch.equal(first.timesteps, expected)
    assert torch.equal(second.timesteps, expected)
    assert torch.equal(first.alpha, second.alpha)
    assert torch.equal(first.bridge_state, second.bridge_state)


def test_endpoint_probability_one_is_exact_cloudy_with_mr_endpoint_coordinate():
    batch = sample_batch(16)
    model = RecordingModel()
    result = training_step(
        model, batch,
        schedule=get_bridge_schedule("canonical_alpha"),
        total_steps=1000,
        sampler_generator=generator(42),
        conditioning_mode=MR_INVERSE_CONDITIONING,
        endpoint_probability=1.0,
    )
    assert torch.equal(result.timesteps, torch.full((16,), 1000, dtype=torch.long))
    assert torch.equal(result.alpha, torch.ones(16, 1, 1, 1))
    assert torch.equal(model.x_alpha, batch["cloudy"])
    assert torch.equal(model.conditioning, torch.full((16,), 1000.0))


@pytest.mark.parametrize("rho", [0.1, 0.25, 0.5])
def test_mixture_contains_endpoints_and_interior_and_is_reproducible(rho):
    batch = sample_batch(256)
    results = [
        training_step(
            RecordingModel(), batch,
            schedule=get_bridge_schedule("canonical_alpha"),
            total_steps=1000, sampler_generator=generator(7),
            conditioning_mode=MR_INVERSE_CONDITIONING,
            endpoint_probability=rho,
        )
        for _ in range(2)
    ]
    assert torch.equal(results[0].timesteps, results[1].timesteps)
    assert torch.equal(results[0].bridge_state, results[1].bridge_state)
    n_endpoints = int((results[0].timesteps == 1000).sum())
    assert 0 < n_endpoints < 256
    assert abs(n_endpoints / 256 - rho) < 0.12


def test_mixture_does_not_alter_standard_random_t_validation():
    batch = sample_batch(16)
    schedule = get_bridge_schedule("canonical_alpha")
    val = random_t_validation_step(
        RecordingModel(), batch, schedule=schedule,
        total_steps=1000, sampler_generator=generator(3),
        conditioning_mode=MR_INVERSE_CONDITIONING,
    )
    expected = sample_timesteps(
        16, 1000, generator=generator(3), device="cpu"
    )
    assert torch.equal(val.timesteps, expected)


@pytest.mark.parametrize("invalid", [-0.2, 1.1, float("nan"), float("inf")])
def test_mixture_probability_validation(invalid):
    batch = sample_batch(2)
    with pytest.raises(ValueError, match="endpoint_probability"):
        random_t_bridge_step(
            RecordingModel(), batch,
            schedule=get_bridge_schedule("canonical_alpha"),
            total_steps=1000, sampler_generator=generator(42),
            endpoint_probability=invalid,
        )


def test_training_cli_matrix_and_invalid_values():
    parser = build_parser()
    base = parser.parse_args(["--run-name", "base"])
    assert base.backbone == "cnn"
    assert base.endpoint_probability == 0.0
    for backbone in ("cnn", "naf"):
        for rho in (0.0, 0.10, 0.25, 0.50):
            args = parser.parse_args([
                "--run-name", "test",
                "--backbone", backbone,
                "--endpoint-probability", str(rho),
                "--beta-a", "1.0",
                "--conditioning-coordinate", "mr_inverse",
            ])
            assert _validate_experiment(args) == 1.0
    args = parser.parse_args([
        "--run-name", "bad",
        "--endpoint-probability", "0.25",
        "--beta-a", "2.0",
    ])
    with pytest.raises(ValueError, match="beta-a"):
        _validate_experiment(args)


def test_resume_identity_rejects_changed_endpoint_rate_and_backbone():
    common = {
        "model_identity": "canonical_bridge_net",
        "trainable_parameters": 7653037,
        "architecture_widths": [32, 64, 128, 256],
        "bridge_geometry": "straight_linear",
        "schedule_name": "canonical_alpha",
        "training_bridge_measure": "uniform_discrete_alpha_grid",
        "beta_a": 1.0, "beta_b": 1.0,
        "conditioning_mode": "mr_inverse",
        "conditioning_definition": "mr_inverse",
        "total_steps": 1000,
        "batch_size": 4,
        "num_workers": 4,
        "learning_rate": 5e-5,
        "split_protocol": "reference",
        "split_audit": {},
        "reproducibility": {
            "seeds": {"train_seed": 42, "sampler_seed": 42},
            "deterministic_algorithms": False,
        },
    }
    assert _resume_identity(common)["backbone_variant"] == "cnn"
    assert _resume_identity(common)["endpoint_probability"] == 0.0
    changed_rho = {**common, "endpoint_probability": 0.25}
    changed_backbone = {**common, "backbone_variant": "naf"}
    with pytest.raises(ValueError, match="endpoint_probability"):
        _validate_resume_metadata(common, changed_rho)
    with pytest.raises(ValueError, match="backbone_variant"):
        _validate_resume_metadata(common, changed_backbone)


def test_invalid_model_backbone_and_checkpoint_backbone_rejected():
    with pytest.raises(ValueError, match="backbone"):
        CanonicalBridgeNet(backbone_variant="unknown")
    with pytest.raises(ValueError, match="backbone_variant"):
        _checkpoint_backbone({"backbone_variant": "unknown"})
