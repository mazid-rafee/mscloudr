import torch

from mscloudr.bridge import get_bridge_schedule
from mscloudr.cli.pilot_nfe_grid_eval import reverse_alpha_grid


def test_equal_alpha_grid_is_uniform_in_physical_coordinate():
    grid = reverse_alpha_grid(
        total_steps=1000,
        nfe=4,
        grid="equal_alpha",
        schedule=get_bridge_schedule("mr_r3"),
        device=torch.device("cpu"),
    )
    assert torch.allclose(grid, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_canonical_equal_t_matches_equal_alpha():
    schedule = get_bridge_schedule("canonical_alpha")
    equal_t = reverse_alpha_grid(
        total_steps=1000,
        nfe=5,
        grid="equal_t",
        schedule=schedule,
        device=torch.device("cpu"),
    )
    equal_alpha = reverse_alpha_grid(
        total_steps=1000,
        nfe=5,
        grid="equal_alpha",
        schedule=schedule,
        device=torch.device("cpu"),
    )
    assert torch.allclose(equal_t, equal_alpha)


def test_mr_equal_t_differs_from_equal_alpha_but_preserves_endpoints():
    schedule = get_bridge_schedule("mr_r3")
    equal_t = reverse_alpha_grid(
        total_steps=1000,
        nfe=3,
        grid="equal_t",
        schedule=schedule,
        device=torch.device("cpu"),
    )
    equal_alpha = reverse_alpha_grid(
        total_steps=1000,
        nfe=3,
        grid="equal_alpha",
        schedule=schedule,
        device=torch.device("cpu"),
    )
    assert equal_t[0].item() == 1.0
    assert equal_t[-1].item() == 0.0
    assert not torch.allclose(equal_t, equal_alpha)
    assert torch.all(equal_t[1:] <= equal_t[:-1])


def test_nfe1_grids_are_identical_for_any_endpoint_preserving_schedule():
    schedule = get_bridge_schedule("original")
    equal_t = reverse_alpha_grid(
        total_steps=1000,
        nfe=1,
        grid="equal_t",
        schedule=schedule,
        device=torch.device("cpu"),
    )
    equal_alpha = reverse_alpha_grid(
        total_steps=1000,
        nfe=1,
        grid="equal_alpha",
        schedule=schedule,
        device=torch.device("cpu"),
    )
    assert torch.equal(equal_t, equal_alpha)
    assert torch.equal(equal_t, torch.tensor([1.0, 0.0]))
