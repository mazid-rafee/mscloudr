"""Mathematical and pipeline invariants for independent projected components."""
from functools import partial

import pytest
import torch

from mscloudr.bridge import get_bridge_schedule
from mscloudr.component_training import component_training_step
from mscloudr.models.component_bridge import (
    CanonicalComponentBridgeNet,
    low_pass_projection,
    high_pass_projection,
)
from mscloudr.runner import run_training_epoch


def _small_model():
    return CanonicalComponentBridgeNet(
        in_opt=2, in_sar=2, widths=(8, 16, 32, 64),
        alpha_dim=8, mid_blocks=1, total_steps=16,
    )


def _data():
    g = torch.Generator().manual_seed(81)
    return [{
        "cloudy": torch.randn(2, 2, 16, 16, generator=g),
        "target": torch.randn(2, 2, 16, 16, generator=g),
        "sar": torch.randn(2, 2, 16, 16, generator=g),
    }]


def test_projection_is_orthogonal_complete_idempotent():
    torch.manual_seed(2)
    x = torch.randn(2, 13, 16, 16)
    low, high = low_pass_projection(x), high_pass_projection(x)
    assert torch.allclose(low + high, x, atol=1e-6)
    assert torch.allclose(low_pass_projection(low), low, atol=1e-6)
    assert torch.allclose(low_pass_projection(high), torch.zeros_like(high), atol=1e-6)
    assert (low * high).sum().abs() < 1e-4
    with pytest.raises(ValueError):
        low_pass_projection(torch.ones(1, 1, 15, 16))


def test_reconstruction_and_both_heads_receive_gradients():
    torch.manual_seed(2)
    model = _small_model()
    batch = _data()[0]
    t = torch.tensor([2., 16.])
    low, high = model.forward_components(batch["cloudy"], t, batch["sar"])
    result = model(batch["cloudy"], t, batch["sar"])
    assert result.shape == batch["target"].shape
    assert torch.allclose(result, low + high, atol=1e-6)
    assert torch.allclose(low_pass_projection(high), torch.zeros_like(high), atol=1e-6)
    assert torch.isfinite(result).all()
    objective = (low - low_pass_projection(batch["target"])).square().mean() + (
        high - high_pass_projection(batch["target"])).square().mean()
    objective.backward()
    assert model.head.weight.grad is not None
    assert model.head.weight.grad[:2].abs().sum() > 0
    assert model.head.weight.grad[2:].abs().sum() > 0


def test_component_objective_training_step_is_finite():
    torch.manual_seed(3)
    model = _small_model()
    batch = _data()[0]
    kwargs = dict(
        schedule=get_bridge_schedule("canonical_alpha"),
        total_steps=16,
        sampler_generator=torch.Generator().manual_seed(42),
        conditioning_mode="mr_inverse",
    )
    result = component_training_step(model, batch, component_weight=0.5, **kwargs)
    assert result.loss.isfinite()
    assert result.prediction.shape == batch["target"].shape
    result.loss.backward()
    assert model.head.weight.grad is not None
    with pytest.raises(ValueError):
        component_training_step(model, batch, component_weight=-1, **kwargs)


def test_runner_accepts_custom_step_without_modifying_default():
    torch.manual_seed(4)
    model = _small_model()
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = run_training_epoch(
        model, _data(), optimizer,
        schedule=get_bridge_schedule("canonical_alpha"),
        total_steps=16,
        sampler_generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
        conditioning_mode="mr_inverse",
        train_step_fn=partial(component_training_step, component_weight=0.5),
        max_batches=1,
    )
    assert loss > 0
    assert not torch.equal(model.head.weight, before)
