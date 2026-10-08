import torch

from mscloudr.bridge import make_bridge_state
from mscloudr.models import (
    CANONICAL_DUAL_ROLE_SAR_BRIDGE_MODEL_IDENTITY,
    CanonicalDualRoleSARBridgeNet,
    count_canonical_bridge_parameters,
    count_canonical_dual_role_sar_bridge_parameters,
)
from mscloudr.training import make_model_training_bridge_state


def _inputs(batch_size=2, height=16, width=16):
    clean = torch.rand(batch_size, 13, height, width)
    cloudy = torch.rand(batch_size, 13, height, width)
    sar = torch.rand(batch_size, 2, height, width)
    return clean, cloudy, sar


def test_dual_role_sar_bridge_identity_and_parameter_increment():
    assert (
        CANONICAL_DUAL_ROLE_SAR_BRIDGE_MODEL_IDENTITY
        == "canonical_dual_role_sar_bridge_net"
    )
    assert (
        count_canonical_dual_role_sar_bridge_parameters()
        > count_canonical_bridge_parameters()
    )


def test_dual_role_model_keeps_existing_sar_restoration_input():
    model = CanonicalDualRoleSARBridgeNet(total_steps=1000)
    x_alpha = torch.rand(2, 13, 16, 16)
    sar = torch.rand(2, 2, 16, 16)
    conditioning = torch.tensor([250.0, 1000.0])

    with torch.no_grad():
        prediction = model(x_alpha, conditioning, sar)

    assert prediction.shape == x_alpha.shape
    assert torch.isfinite(prediction).all()


def test_sar_bridge_starts_as_exact_straight_bridge():
    model = CanonicalDualRoleSARBridgeNet(sar_bridge_kappa=0.05)
    clean, cloudy, sar = _inputs()
    alpha = torch.tensor([0.25, 0.75])

    straight = make_bridge_state(clean, cloudy, alpha)
    curved = make_model_training_bridge_state(
        model,
        clean,
        cloudy,
        sar,
        alpha,
    )

    assert torch.equal(curved, straight)


def test_sar_bridge_preserves_clean_and_cloudy_endpoints_after_learning():
    model = CanonicalDualRoleSARBridgeNet(sar_bridge_kappa=0.05)
    with torch.no_grad():
        model.sar_bridge_curvature.head.weight.fill_(0.01)
        model.sar_bridge_curvature.head.bias.fill_(0.01)

    clean, cloudy, sar = _inputs()
    alpha = torch.tensor([0.0, 1.0])

    curved = make_model_training_bridge_state(
        model,
        clean,
        cloudy,
        sar,
        alpha,
    )

    assert torch.equal(curved[0], clean[0])
    assert torch.equal(curved[1], cloudy[1])


def test_sar_curvature_increment_has_no_clean_target_access():
    model = CanonicalDualRoleSARBridgeNet(sar_bridge_kappa=0.05)
    with torch.no_grad():
        model.sar_bridge_curvature.head.weight.fill_(0.01)
        model.sar_bridge_curvature.head.bias.fill_(0.01)

    clean_a, cloudy, sar = _inputs()
    clean_b = torch.rand_like(clean_a)
    alpha = torch.tensor([0.4, 0.6])

    curved_a = make_model_training_bridge_state(
        model, clean_a, cloudy, sar, alpha
    )
    curved_b = make_model_training_bridge_state(
        model, clean_b, cloudy, sar, alpha
    )
    straight_a = make_bridge_state(clean_a, cloudy, alpha)
    straight_b = make_bridge_state(clean_b, cloudy, alpha)

    increment_a = curved_a - straight_a
    increment_b = curved_b - straight_b

    assert torch.allclose(increment_a, increment_b, atol=1e-7, rtol=0.0)


def test_sar_curvature_changes_when_paired_sar_changes():
    model = CanonicalDualRoleSARBridgeNet(sar_bridge_kappa=0.05)
    with torch.no_grad():
        model.sar_bridge_curvature.head.weight.fill_(0.01)

    _, cloudy, sar_a = _inputs()
    sar_b = torch.flip(sar_a, dims=[-1])

    with torch.no_grad():
        phi_a = model.bridge_curvature(cloudy, sar_a)
        phi_b = model.bridge_curvature(cloudy, sar_b)

    assert not torch.equal(phi_a, phi_b)
    assert torch.max(torch.abs(phi_a)) <= 0.05 + 1e-7
    assert torch.max(torch.abs(phi_b)) <= 0.05 + 1e-7


def test_zero_initialized_curvature_head_receives_gradient():
    model = CanonicalDualRoleSARBridgeNet(sar_bridge_kappa=0.05)
    clean, cloudy, sar = _inputs()
    alpha = torch.tensor([0.4, 0.6])

    curved = make_model_training_bridge_state(
        model,
        clean,
        cloudy,
        sar,
        alpha,
    )
    curved.mean().backward()

    grad = model.sar_bridge_curvature.head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
