import torch

from mscloudr.models.blocks import (
    LayerNorm2d,
    NAFBlock,
    SFBlock,
    count_trainable_parameters,
)


def test_layernorm2d_normalizes_channels_per_pixel():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 5, 7)
    norm = LayerNorm2d(8)
    y = norm(x)

    mean = y.mean(dim=1)
    var = y.var(dim=1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=2e-4)


def test_nafblock_shape_and_initial_identity():
    torch.manual_seed(0)
    block = NAFBlock(22)
    x = torch.randn(2, 22, 32, 32)
    y = block(x)

    assert y.shape == x.shape
    # Official NAFNet initializes beta/gamma to zero, so a fresh block is an
    # exact identity despite nonzero internal convolution weights.
    assert torch.equal(y, x)


def test_nafblock_gradients_are_finite():
    torch.manual_seed(0)
    block = NAFBlock(22)
    # Turn on residual branches so internal convolutions receive gradients.
    with torch.no_grad():
        block.beta.fill_(1.0)
        block.gamma.fill_(1.0)

    x = torch.randn(2, 22, 16, 16, requires_grad=True)
    loss = block(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_nafblock_parameter_count_matches_official_structure():
    # Replacing GroupNorm(1,C) with LayerNorm2d changes normalization
    # semantics but not the number of affine parameters.
    assert count_trainable_parameters(NAFBlock(22)) == 4114
    assert count_trainable_parameters(NAFBlock(176)) == 222640


def test_sfblock_shapes_for_all_dbcr_levels():
    torch.manual_seed(0)
    for channels, heads in ((22, 1), (44, 1), (88, 2), (176, 4)):
        block = SFBlock(channels, heads)
        x_opt = torch.randn(2, channels, 8, 8)
        x_sar = torch.randn(2, channels, 8, 8)
        y = block(x_opt, x_sar)
        assert y.shape == x_opt.shape
        assert torch.isfinite(y).all()


def test_sfblock_gradients_are_finite():
    torch.manual_seed(0)
    block = SFBlock(88, 2)
    x_opt = torch.randn(1, 88, 8, 8, requires_grad=True)
    x_sar = torch.randn(1, 88, 8, 8, requires_grad=True)

    loss = block(x_opt, x_sar).square().mean()
    loss.backward()

    assert x_opt.grad is not None and torch.isfinite(x_opt.grad).all()
    assert x_sar.grad is not None and torch.isfinite(x_sar.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sfblock_residual_path_can_reduce_to_optical_identity():
    block = SFBlock(22, 1)

    # Remove SAR-attended and MLP contributions and make the final projection
    # the identity. Equation (16) should then preserve the optical residual.
    with torch.no_grad():
        block.v.weight.zero_()
        block.v.bias.zero_()
        for mlp in block.mlps:
            for parameter in mlp.parameters():
                parameter.zero_()
        block.proj.weight.zero_()
        block.proj.bias.zero_()
        for channel in range(22):
            block.proj.weight[channel, channel, 0, 0] = 1.0

    x_opt = torch.randn(1, 22, 8, 8)
    x_sar = torch.randn(1, 22, 8, 8)
    y = block(x_opt, x_sar)
    assert torch.allclose(y, x_opt, atol=1e-6)


def test_sfblock_rejects_invalid_head_partition():
    try:
        SFBlock(22, 4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected ValueError")
