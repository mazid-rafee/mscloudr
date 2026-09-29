import random

import numpy as np
import torch

from mscloudr.reproducibility import make_torch_generator, seed_everything


def _draw():
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )


def test_seed_everything_repeats_cpu_rng_streams():
    seed_everything(42)
    first = _draw()

    seed_everything(42)
    second = _draw()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_explicit_generators_are_reproducible_and_independent():
    a = torch.randint(0, 1000, (32,), generator=make_torch_generator(7))
    b = torch.randint(0, 1000, (32,), generator=make_torch_generator(7))
    c = torch.randint(0, 1000, (32,), generator=make_torch_generator(8))

    assert torch.equal(a, b)
    assert not torch.equal(a, c)
