import torch

from mscloudr.models import (
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CanonicalBridgeNet,
    LegacyNAFBlock,
    LegacySFBlock,
    count_canonical_bridge_parameters,
)


def test_canonical_bridge_identity_name():
    assert CANONICAL_BRIDGE_MODEL_IDENTITY == "canonical_bridge_net"


def test_canonical_bridge_forward_shape():
    model = CanonicalBridgeNet(total_steps=1000)
    x_alpha = torch.randn(2, 13, 64, 64)
    sar = torch.randn(2, 2, 64, 64)
    conditioning = torch.tensor([250.0, 900.0])

    with torch.no_grad():
        prediction = model(x_alpha, conditioning, sar)

    assert prediction.shape == x_alpha.shape
    assert torch.isfinite(prediction).all()


def test_canonical_bridge_has_no_legacy_dbcr_blocks():
    model = CanonicalBridgeNet()
    assert not any(isinstance(module, LegacyNAFBlock) for module in model.modules())
    assert not any(isinstance(module, LegacySFBlock) for module in model.modules())


def test_canonical_bridge_parameter_scale_is_reasonable():
    count = count_canonical_bridge_parameters()
    assert 5_000_000 < count < 10_000_000


def test_physical_alpha_conditioning_accepts_endpoint():
    model = CanonicalBridgeNet(total_steps=1000)
    x_alpha = torch.randn(1, 13, 32, 32)
    sar = torch.randn(1, 2, 32, 32)

    with torch.no_grad():
        prediction = model(x_alpha, torch.tensor([1000.0]), sar)

    assert prediction.shape == x_alpha.shape
