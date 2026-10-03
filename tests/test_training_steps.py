import torch
import torch.nn as nn

from mscloudr.bridge import (
    mean_reverting_alpha,
    sine_alpha,
)
from mscloudr.reproducibility import make_torch_generator
from mscloudr.training import (
    CHECKPOINT_SELECTION_METRIC,
    DIAGNOSTIC_RANDOM_T_METRIC,
    WeightedMean,
    checkpoint_metric_payload,
    endpoint_validation_step,
    random_t_validation_step,
    training_step,
)


class TinyBridgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(0.75)
        )
        self.last_x = None
        self.last_t = None
        self.last_sar = None

    def forward(self, x_t, t, sar):
        self.last_x = x_t.detach().clone()
        self.last_t = t.detach().clone()
        self.last_sar = sar.detach().clone()
        return self.scale * x_t


def _batch(batch_size=4):
    torch.manual_seed(7)
    return {
        "cloudy": torch.rand(
            batch_size,
            13,
            4,
            4,
        ),
        "sar": torch.rand(
            batch_size,
            2,
            4,
            4,
        ),
        "target": torch.rand(
            batch_size,
            13,
            4,
            4,
        ),
    }


def test_training_step_is_reproducible_with_explicit_sampler_generator():
    batch = _batch()
    first_model = TinyBridgeModel()
    second_model = TinyBridgeModel()

    first = training_step(
        first_model,
        batch,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(123),
    )
    second = training_step(
        second_model,
        batch,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(123),
    )

    assert torch.equal(first.timesteps, second.timesteps)
    assert torch.equal(first.bridge_state, second.bridge_state)
    assert torch.equal(first.prediction, second.prediction)
    assert torch.equal(first.loss, second.loss)


def test_same_timesteps_different_schedule_changes_training_bridge_states():
    batch = _batch()

    sine = training_step(
        TinyBridgeModel(),
        batch,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(42),
    )

    def mr_r3(t, total_steps):
        return mean_reverting_alpha(
            t,
            total_steps,
            rate=3.0,
        )

    mr = training_step(
        TinyBridgeModel(),
        batch,
        schedule=mr_r3,
        total_steps=1000,
        sampler_generator=make_torch_generator(42),
    )

    assert torch.equal(sine.timesteps, mr.timesteps)
    assert not torch.allclose(sine.alpha, mr.alpha)
    assert not torch.allclose(sine.bridge_state, mr.bridge_state)


def test_training_step_preserves_gradient_path_to_model_parameters():
    model = TinyBridgeModel()
    result = training_step(
        model,
        _batch(),
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(4),
    )

    result.loss.backward()

    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_random_t_validation_is_gradient_free():
    model = TinyBridgeModel()
    result = random_t_validation_step(
        model,
        _batch(),
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(5),
    )

    assert not result.loss.requires_grad
    assert not result.prediction.requires_grad


def test_endpoint_validation_uses_cloudy_input_and_exact_T():
    model = TinyBridgeModel()
    batch = _batch(batch_size=3)

    result = endpoint_validation_step(
        model,
        batch,
        total_steps=1000,
    )

    assert torch.equal(result.bridge_state, batch["cloudy"])
    assert torch.equal(model.last_x, batch["cloudy"])
    assert torch.equal(
        model.last_t,
        torch.full((3,), 1000, dtype=torch.long),
    )
    assert torch.equal(
        result.alpha,
        torch.ones(3, 1, 1, 1),
    )


def test_endpoint_validation_does_not_depend_on_schedule():
    model = TinyBridgeModel()
    batch = _batch()

    result = endpoint_validation_step(
        model,
        batch,
        total_steps=1000,
    )

    expected = torch.mean(
        torch.abs(
            0.75 * batch["cloudy"]
            - batch["target"]
        )
    )
    assert torch.allclose(result.loss, expected)


def test_weighted_mean_is_sample_weighted_not_batch_weighted():
    meter = WeightedMean()
    meter.update(1.0, n=4)
    meter.update(3.0, n=1)

    assert meter.mean == 1.4


def test_checkpoint_payload_names_endpoint_as_canonical_metric():
    payload = checkpoint_metric_payload(
        endpoint_l1=0.02,
        random_t_l1=0.01,
    )

    assert CHECKPOINT_SELECTION_METRIC == "val_endpoint_l1"
    assert DIAGNOSTIC_RANDOM_T_METRIC == "val_random_t_l1"
    assert payload == {
        "val_endpoint_l1": 0.02,
        "val_random_t_l1": 0.01,
    }


def test_missing_required_batch_key_is_rejected():
    batch = _batch()
    del batch["sar"]

    try:
        training_step(
            TinyBridgeModel(),
            batch,
            schedule=sine_alpha,
            total_steps=1000,
            sampler_generator=make_torch_generator(1),
        )
    except KeyError as error:
        assert "sar" in str(error)
    else:
        raise AssertionError(
            "expected missing SAR key to fail"
        )
