import math

import torch

from mscloudr.models.patent_blocks import (
    PatentFeatureMapLayerNorm,
    PatentTimeEmbeddedNAFBlock,
    PatentTimeEmbedding,
    PatentTimeEmbeddingConfig,
    count_trainable_parameters,
    sine_bridge_alpha,
)


def test_sine_bridge_alpha_has_exact_intended_endpoints():
    t = torch.tensor([0.0, 500.0, 1000.0])
    alpha = sine_bridge_alpha(t, 1000)
    assert abs(float(alpha[0])) < 1e-7
    assert abs(float(alpha[-1]) - 1.0) < 1e-6
    assert 0.0 < float(alpha[1]) < 1.0


def test_patent_time_embedding_returns_four_channel_vectors():
    module = PatentTimeEmbedding(22)
    t = torch.tensor([0.0, 250.0, 1000.0])
    outputs = module(t, 1000)

    assert len(outputs) == 4
    for tensor in outputs:
        assert tensor.shape == (3, 22, 1, 1)
        assert torch.isfinite(tensor).all()


def test_time_embedding_from_alpha_matches_sine_schedule_path():
    torch.manual_seed(0)
    module = PatentTimeEmbedding(22)
    t = torch.tensor([100.0, 700.0])
    alpha = sine_bridge_alpha(t, 1000)

    from_t = module(t, 1000)
    from_alpha = module.from_alpha(alpha)

    for lhs, rhs in zip(from_t, from_alpha):
        assert torch.allclose(lhs, rhs)


def test_feature_map_layernorm_uses_all_feature_elements():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 5, 7)
    norm = PatentFeatureMapLayerNorm(
        8,
        affine=False,
    )
    y = norm(x)

    mean = y.mean(dim=(1, 2, 3))
    variance = y.var(
        dim=(1, 2, 3),
        unbiased=False,
    )
    assert torch.allclose(
        mean,
        torch.zeros_like(mean),
        atol=1e-5,
    )
    assert torch.allclose(
        variance,
        torch.ones_like(variance),
        atol=2e-4,
    )


def test_patent_nafblock_is_identity_at_zero_residual_initialization():
    torch.manual_seed(0)
    block = PatentTimeEmbeddedNAFBlock(22)
    x = torch.randn(2, 22, 16, 16)
    t = torch.tensor([10.0, 900.0])

    y = block(x, t, 1000)
    assert torch.equal(y, x)


def test_patent_nafblock_gradients_are_finite_when_residuals_enabled():
    torch.manual_seed(0)
    block = PatentTimeEmbeddedNAFBlock(22)
    with torch.no_grad():
        block.beta.fill_(1.0)
        block.gamma.fill_(1.0)

    x = torch.randn(
        2,
        22,
        8,
        8,
        requires_grad=True,
    )
    alpha = torch.tensor([0.2, 0.8])
    loss = block(x, alpha=alpha).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_external_modulation_supports_time_embedding_sharing():
    time_embedding = PatentTimeEmbedding(22)
    block = PatentTimeEmbeddedNAFBlock(
        22,
        own_time_embedding=False,
    )
    alpha = torch.tensor([0.3])
    modulation = time_embedding.from_alpha(alpha)
    x = torch.randn(1, 22, 8, 8)

    y = block(
        x,
        modulation=modulation,
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_default_parameter_counts_are_frozen_as_diagnostic_assumptions():
    # These counts depend on the explicit, non-patent defaults
    # positional_dim=hidden_dim=128 and per-block ownership.
    time_module = PatentTimeEmbedding(
        22,
        PatentTimeEmbeddingConfig(
            positional_dim=128,
            hidden_dim=128,
        ),
    )
    owned_block = PatentTimeEmbeddedNAFBlock(22)
    shared_block = PatentTimeEmbeddedNAFBlock(
        22,
        own_time_embedding=False,
    )

    assert count_trainable_parameters(time_module) == 77_400
    assert count_trainable_parameters(shared_block) == 3_082
    assert count_trainable_parameters(owned_block) == 80_482
