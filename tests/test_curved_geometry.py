import torch

from mscloudr.bridge import (
    make_bridge_state,
    orthogonal_residual_direction,
)


def _example_pair():
    clean = torch.tensor(
        [
            [[[0.10, 0.20], [0.30, 0.40]]],
            [[[0.40, 0.30], [0.20, 0.10]]],
        ],
        dtype=torch.float32,
    )
    cloudy = torch.tensor(
        [
            [[[0.70, 0.10], [0.80, 0.20]]],
            [[[0.20, 0.90], [0.10, 0.60]]],
        ],
        dtype=torch.float32,
    )
    return clean, cloudy


def test_orthogonal_direction_is_norm_preserving_and_orthogonal():
    clean, cloudy = _example_pair()
    residual = (cloudy - clean).reshape(clean.shape[0], -1)
    phi = orthogonal_residual_direction(clean, cloudy).reshape(clean.shape[0], -1)

    dot = torch.sum(residual * phi, dim=1)
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    phi_norm = torch.linalg.vector_norm(phi, dim=1)

    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-7, rtol=0.0)
    assert torch.allclose(phi_norm, residual_norm, atol=1e-7, rtol=1e-6)


def test_curved_geometry_preserves_endpoints():
    clean, cloudy = _example_pair()

    at_clean = make_bridge_state(
        clean,
        cloudy,
        torch.zeros(clean.shape[0]),
        curvature_kappa=0.1,
    )
    at_cloudy = make_bridge_state(
        clean,
        cloudy,
        torch.ones(clean.shape[0]),
        curvature_kappa=-0.1,
    )

    assert torch.equal(at_clean, clean)
    assert torch.equal(at_cloudy, cloudy)


def test_kappa_zero_is_exact_straight_bridge():
    clean, cloudy = _example_pair()
    alpha = torch.tensor([0.25, 0.75])

    expected = (
        (1.0 - alpha.view(-1, 1, 1, 1)) * clean
        + alpha.view(-1, 1, 1, 1) * cloudy
    )
    actual = make_bridge_state(
        clean,
        cloudy,
        alpha,
        curvature_kappa=0.0,
    )

    assert torch.equal(actual, expected)


def test_midpoint_perturbation_ratio_equals_abs_kappa():
    clean, cloudy = _example_pair()
    alpha = torch.full((clean.shape[0],), 0.5)
    kappa = 0.1

    straight = make_bridge_state(clean, cloudy, alpha, curvature_kappa=0.0)
    curved = make_bridge_state(clean, cloudy, alpha, curvature_kappa=kappa)

    perturbation = (curved - straight).reshape(clean.shape[0], -1)
    residual = (cloudy - clean).reshape(clean.shape[0], -1)
    ratio = (
        torch.linalg.vector_norm(perturbation, dim=1)
        / torch.linalg.vector_norm(residual, dim=1)
    )

    assert torch.allclose(
        ratio,
        torch.full_like(ratio, abs(kappa)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_positive_and_negative_kappa_bend_in_opposite_directions():
    clean, cloudy = _example_pair()
    alpha = torch.tensor([0.3, 0.7])

    straight = make_bridge_state(clean, cloudy, alpha, curvature_kappa=0.0)
    positive = make_bridge_state(clean, cloudy, alpha, curvature_kappa=0.1)
    negative = make_bridge_state(clean, cloudy, alpha, curvature_kappa=-0.1)

    assert torch.allclose(
        positive - straight,
        -(negative - straight),
        atol=1e-7,
        rtol=1e-6,
    )


def test_odd_flattened_dimension_is_rejected_for_curvature():
    clean = torch.zeros(1, 1, 1, 3)
    cloudy = torch.ones_like(clean)

    try:
        make_bridge_state(clean, cloudy, 0.5, curvature_kappa=0.1)
    except ValueError as exc:
        assert "even flattened" in str(exc)
    else:
        raise AssertionError("expected ValueError for odd flattened dimension")
