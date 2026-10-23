"""Tests for CanonicalBridgeNet control-selection and alpha conditioning."""
import pytest
import torch
from torch import nn

from mscloudr.bridge import get_bridge_schedule
from mscloudr.cli.diagnose_canonical_alignment import validate_canonical_control
from mscloudr.endpoint_alignment import profile_endpoint_alignment


class TinyAlphaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))
        self.offset = nn.Parameter(torch.tensor(0.05))

    def forward(self, x_t, t, sar):
        return self.scale * x_t + (t[:, None, None, None] / 10) * self.offset + .1 * sar


def _batch():
    return [{
        "cloudy": torch.tensor([[[[.4, .3], [.8, .1]]]]),
        "target": torch.tensor([[[[.1, .7], [.2, .9]]]]),
        "sar": torch.tensor([[[[.2, .5], [.7, .3]]]]),
    }]


def _metadata():
    return dict(
        model_identity="canonical_bridge_net",
        architecture_widths=[32, 64, 128, 256],
        backbone_variant="cnn",
        conditioning_mode="mr_inverse",
        beta_a=1.0,
        endpoint_probability=0.0,
    )


def test_canonical_uniform_mr_inverse_control_is_accepted():
    validate_canonical_control(_metadata())
    legacy_metadata = _metadata()
    del legacy_metadata["backbone_variant"]
    del legacy_metadata["endpoint_probability"]
    validate_canonical_control(legacy_metadata)


@pytest.mark.parametrize("name,value", [
    ("backbone_variant", "naf"),
    ("architecture_widths", [48, 96, 192, 384]),
    ("conditioning_mode", "physical_alpha"),
    ("beta_a", 1.5),
    ("endpoint_probability", 0.2),
])
def test_reject_confounded_control(name, value):
    metadata = _metadata()
    metadata[name] = value
    with pytest.raises(ValueError):
        validate_canonical_control(metadata)


def test_mr_inverse_profiling_is_reproducible_without_updates():
    net = TinyAlphaModel()
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    kwargs = dict(
        schedule=get_bridge_schedule("original"),
        total_steps=10,
        device="cpu",
        num_bins=5,
        max_batches=1,
        sampler_seed=93,
        conditioning_mode="physical_alpha",
    )
    result1 = profile_endpoint_alignment(net, _batch(), **kwargs)
    result2 = profile_endpoint_alignment(net, _batch(), **kwargs)
    assert result1 == result2
    assert result1["conditioning_mode"] == "mr_inverse"
    assert result1["split_used"] == "train_only"
    assert len(result1["bins"]) == 5
    assert all(torch.equal(p, state[k]) for k, p in net.state_dict().items())
    assert all(p.grad is None for p in net.parameters())
