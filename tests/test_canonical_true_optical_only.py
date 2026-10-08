import torch

from mscloudr.models import (
    CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY,
    CanonicalBridgeOpticalOnlyNet,
    GatedSARFusion,
    count_canonical_bridge_optical_only_parameters,
    count_canonical_bridge_parameters,
)


def test_true_optical_only_identity():
    assert (
        CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY
        == "canonical_bridge_optical_only_net"
    )


def test_true_optical_only_forward_uses_13_channel_optical_input():
    model = CanonicalBridgeOpticalOnlyNet(total_steps=1000)
    x_alpha = torch.randn(2, 13, 32, 32)
    conditioning = torch.tensor([250.0, 1000.0])

    with torch.no_grad():
        prediction = model(x_alpha, conditioning)

    assert prediction.shape == x_alpha.shape
    assert torch.isfinite(prediction).all()


def test_true_optical_only_prediction_is_independent_of_sar_argument():
    model = CanonicalBridgeOpticalOnlyNet(total_steps=1000)
    model.eval()
    x_alpha = torch.randn(2, 13, 32, 32)
    conditioning = torch.tensor([500.0, 1000.0])
    sar_a = torch.randn(2, 2, 32, 32)
    sar_b = torch.randn(2, 2, 32, 32) * 100.0

    with torch.no_grad():
        pred_a = model(x_alpha, conditioning, sar_a)
        pred_b = model(x_alpha, conditioning, sar_b)

    assert torch.equal(pred_a, pred_b)


def test_true_optical_only_has_no_sar_modules_or_fusions():
    model = CanonicalBridgeOpticalOnlyNet()

    assert not hasattr(model, "sar_stem")
    assert not hasattr(model, "sar_blocks")
    assert not hasattr(model, "sar_downs")
    assert not hasattr(model, "fusions")
    assert not any(isinstance(module, GatedSARFusion) for module in model.modules())


def test_true_optical_only_has_fewer_parameters_than_multimodal():
    optical_only = count_canonical_bridge_optical_only_parameters()
    multimodal = count_canonical_bridge_parameters()

    assert optical_only < multimodal
    assert optical_only > 1_000_000
