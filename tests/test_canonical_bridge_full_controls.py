import math

import torch
import torch.nn as nn

from mscloudr.bridge import (
    mean_reverting_alpha,
    sine_alpha,
)
from mscloudr.cli.train_canonical_bridge import _beta_a1_schedule
from mscloudr.reproducibility import make_torch_generator
from mscloudr.training import (
    MR_INVERSE_CONDITIONING,
    PHYSICAL_ALPHA_CONDITIONING,
    SINE_INVERSE_CONDITIONING,
    model_conditioning_coordinate,
    training_step,
)


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_x = None
        self.last_conditioning = None

    def forward(self, x, conditioning, sar):
        self.last_x = x.detach().clone()
        self.last_conditioning = conditioning.detach().clone()
        return self.scale * x


def _batch(batch_size=4):
    torch.manual_seed(11)
    return {
        "cloudy": torch.rand(batch_size, 13, 4, 4),
        "sar": torch.rand(batch_size, 2, 4, 4),
        "target": torch.rand(batch_size, 13, 4, 4),
    }


def test_physical_conditioning_is_T_times_alpha():
    t = torch.tensor([0, 250, 1000])
    alpha = torch.tensor([0.0, 0.25, 1.0])
    c = model_conditioning_coordinate(
        t,
        alpha,
        total_steps=1000,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )
    assert torch.equal(c, torch.tensor([0.0, 250.0, 1000.0]))


def test_sine_inverse_reconstructs_alpha():
    t = torch.tensor([0, 1, 2, 3])
    alpha = torch.tensor([0.0, 0.2, 0.7, 1.0])
    c = model_conditioning_coordinate(
        t,
        alpha,
        total_steps=1000,
        conditioning_mode=SINE_INVERSE_CONDITIONING,
    )
    u = c / 1000.0
    reconstructed = torch.sin((math.pi / 2.0) * u)
    assert torch.allclose(reconstructed, alpha, atol=1e-6, rtol=1e-6)


def test_mr_inverse_reconstructs_alpha():
    t = torch.tensor([0, 1, 2, 3])
    alpha = torch.tensor([0.0, 0.2, 0.7, 1.0])
    c = model_conditioning_coordinate(
        t,
        alpha,
        total_steps=1000,
        conditioning_mode=MR_INVERSE_CONDITIONING,
    )
    u = c / 1000.0
    reconstructed = mean_reverting_alpha(
        u * 1000.0,
        1000,
        rate=3.0,
    )
    assert torch.allclose(reconstructed, alpha, atol=1e-6, rtol=1e-6)


def test_all_canonical_conditioning_modes_share_exact_endpoints():
    t = torch.tensor([0, 1000])
    alpha = torch.tensor([0.0, 1.0])
    expected = torch.tensor([0.0, 1000.0])
    for mode in (
        PHYSICAL_ALPHA_CONDITIONING,
        SINE_INVERSE_CONDITIONING,
        MR_INVERSE_CONDITIONING,
    ):
        c = model_conditioning_coordinate(
            t,
            alpha,
            total_steps=1000,
            conditioning_mode=mode,
        )
        assert torch.equal(c, expected)


def test_conditioning_mode_does_not_change_sampled_alpha_or_bridge_state():
    batch = _batch()
    schedule = _beta_a1_schedule(1.0)

    outputs = []
    models = []
    for mode in (
        PHYSICAL_ALPHA_CONDITIONING,
        SINE_INVERSE_CONDITIONING,
        MR_INVERSE_CONDITIONING,
    ):
        model = RecordingModel()
        result = training_step(
            model,
            batch,
            schedule=schedule,
            total_steps=1000,
            sampler_generator=make_torch_generator(42),
            conditioning_mode=mode,
        )
        outputs.append(result)
        models.append(model)

    reference = outputs[0]
    for result in outputs[1:]:
        assert torch.equal(result.timesteps, reference.timesteps)
        assert torch.equal(result.alpha, reference.alpha)
        assert torch.equal(result.bridge_state, reference.bridge_state)

    assert not torch.equal(
        models[0].last_conditioning,
        models[1].last_conditioning,
    )
    assert not torch.equal(
        models[0].last_conditioning,
        models[2].last_conditioning,
    )


def test_beta_a_one_matches_uniform_canonical_alpha():
    t = torch.tensor([0.0, 125.0, 500.0, 1000.0])
    beta = _beta_a1_schedule(1.0)(t, 1000)
    expected = t / 1000.0
    assert torch.equal(beta, expected)


def test_beta_schedule_seeded_sampling_is_exactly_reproducible():
    batch = _batch()
    schedule = _beta_a1_schedule(1.5)

    first = training_step(
        RecordingModel(),
        batch,
        schedule=schedule,
        total_steps=1000,
        sampler_generator=make_torch_generator(123),
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )
    second = training_step(
        RecordingModel(),
        batch,
        schedule=schedule,
        total_steps=1000,
        sampler_generator=make_torch_generator(123),
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert torch.equal(first.timesteps, second.timesteps)
    assert torch.equal(first.alpha, second.alpha)
    assert torch.equal(first.bridge_state, second.bridge_state)
