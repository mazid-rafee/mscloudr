import torch

from mscloudr.models import (
    CanonicalSARResidualCoordinateBridgeNet,
    count_canonical_sar_residual_coordinate_parameters,
)
from mscloudr.training import make_model_training_bridge_state


def _inputs(batch=2, size=8):
    cloudy = torch.rand(batch, 13, size, size)
    sar = torch.rand(batch, 2, size, size)
    target = torch.rand(batch, 13, size, size)
    return target, cloudy, sar


def test_residual_coordinate_gate_zero_init_is_exact_straight_bridge():
    model = CanonicalSARResidualCoordinateBridgeNet(
        residual_coordinate_kappa=0.10
    )
    target, cloudy, sar = _inputs()
    alpha = torch.tensor([0.25, 0.75])

    gate = model.bridge_coordinate_gate(cloudy, sar)
    assert gate.shape == (2, 1, 8, 8)
    assert torch.equal(gate, torch.zeros_like(gate))

    effective = model.bridge_effective_alpha(cloudy, sar, alpha)
    expected = alpha.view(-1, 1, 1, 1).expand_as(effective)
    assert torch.equal(effective, expected)

    state = make_model_training_bridge_state(
        model,
        target,
        cloudy,
        sar,
        alpha,
    )
    expected_state = (
        (1.0 - alpha.view(-1, 1, 1, 1)) * target
        + alpha.view(-1, 1, 1, 1) * cloudy
    )
    assert torch.equal(state, expected_state)


def test_residual_coordinate_bridge_preserves_exact_endpoints():
    model = CanonicalSARResidualCoordinateBridgeNet(
        residual_coordinate_kappa=0.10
    )
    # Make the gate nonzero to verify endpoint preservation is structural.
    with torch.no_grad():
        model.sar_residual_coordinate_gate.head.bias.fill_(2.0)

    target, cloudy, sar = _inputs()

    state0 = make_model_training_bridge_state(
        model,
        target,
        cloudy,
        sar,
        torch.zeros(target.shape[0]),
    )
    state1 = make_model_training_bridge_state(
        model,
        target,
        cloudy,
        sar,
        torch.ones(target.shape[0]),
    )

    assert torch.allclose(state0, target)
    assert torch.allclose(state1, cloudy)


def test_residual_coordinate_effective_alpha_is_bounded_and_monotone():
    model = CanonicalSARResidualCoordinateBridgeNet(
        residual_coordinate_kappa=0.10
    )
    with torch.no_grad():
        model.sar_residual_coordinate_gate.head.bias.fill_(10.0)

    _, cloudy, sar = _inputs(batch=1, size=4)
    alphas = torch.linspace(0.0, 1.0, 11)

    values = []
    for alpha in alphas:
        effective = model.bridge_effective_alpha(
            cloudy,
            sar,
            alpha.view(1),
        )
        values.append(float(effective.mean()))

    assert min(values) >= 0.0
    assert max(values) <= 1.0
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_residual_coordinate_model_parameter_count():
    model = CanonicalSARResidualCoordinateBridgeNet(
        residual_coordinate_kappa=0.10
    )
    assert count_canonical_sar_residual_coordinate_parameters(model) == 7666926
