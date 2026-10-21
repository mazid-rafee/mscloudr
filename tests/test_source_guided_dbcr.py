"""Source-guided DB-CR integration tests; independent of SEN12MS-CR files."""

from __future__ import annotations

import pytest
import torch

from mscloudr.models.patent_blocks import (
    PatentTimeEmbeddedNAFBlock,
    PatentTimeEmbeddingConfig,
)
from mscloudr.models.source_guided_dbcr import (
    SOURCE_GUIDED_DBCR_IDENTITY,
    SourceGuidedDBCRConfig,
    SourceGuidedDBCRNet,
)


def compact_config(*, sharing: str = "per_block") -> SourceGuidedDBCRConfig:
    return SourceGuidedDBCRConfig(
        widths=(8, 16, 32, 64),
        encoder_blocks=(1, 1, 1, 2),
        decoder_blocks=(1, 1, 1, 1),
        sf_heads=(1, 1, 2, 4),
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=8,
            hidden_dim=8,
        ),
        time_sharing=sharing,
    )


def test_published_topology_defaults_are_frozen():
    cfg = SourceGuidedDBCRConfig()
    assert cfg.widths == (22, 44, 88, 176)
    assert cfg.encoder_blocks == (1, 1, 1, 28)
    assert cfg.decoder_blocks == (1, 1, 1, 1)
    assert cfg.sf_heads == (1, 1, 2, 4)
    assert cfg.time_sharing == "per_block"
    assert cfg.total_steps == 1000


@pytest.mark.parametrize("sharing", ["per_block", "per_width"])
def test_forward_shape_and_finite(sharing):
    torch.manual_seed(42)
    model = SourceGuidedDBCRNet(compact_config(sharing=sharing)).eval()
    cloudy = torch.randn(2, 13, 16, 16)
    sar = torch.randn(2, 2, 16, 16)
    with torch.no_grad():
        out = model(cloudy, torch.tensor([0., 1000.]), sar)
    assert out.shape == cloudy.shape
    assert torch.isfinite(out).all()


def test_time_embedding_reaches_all_nafblocks_including_sar():
    model = SourceGuidedDBCRNet(compact_config(sharing="per_block"))
    assert all(
        isinstance(block, PatentTimeEmbeddedNAFBlock)
        and block.time_embedding is not None
        for stream in (model.opt_encoders, model.sar_encoders)
        for stage in stream
        for block in stage
    )
    assert all(
        block.time_embedding is not None
        for stage in (*model.decoders, model.bottleneck)
        for block in stage
    )
    assert model.time_modules is None


def test_per_width_shares_four_embedding_modules_and_reduces_params():
    owned = SourceGuidedDBCRNet(compact_config(sharing="per_block"))
    shared = SourceGuidedDBCRNet(compact_config(sharing="per_width"))
    assert shared.time_modules is not None
    assert set(shared.time_modules.keys()) == {"8", "16", "32", "64"}
    assert all(
        block.time_embedding is None
        for stream in (shared.opt_encoders, shared.sar_encoders)
        for stage in stream
        for block in stage
    )
    assert owned.architecture_audit()["trainable_parameters"] > (
        shared.architecture_audit()["trainable_parameters"]
    )


def test_backpropagation_and_sar_gradients():
    torch.manual_seed(1)
    model = SourceGuidedDBCRNet(compact_config())
    # With zero initialized beta/gamma, scale/shift is dormant initially.
    # Activate them to test gradient propagation to the new embedding MLPs.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, PatentTimeEmbeddedNAFBlock):
                layer.beta.fill_(0.5)
                layer.gamma.fill_(0.5)
    optical = torch.randn(1, 13, 16, 16, requires_grad=True)
    sar = torch.randn(1, 2, 16, 16, requires_grad=True)
    loss = model(optical, torch.tensor([500.0]), sar).square().mean()
    loss.backward()
    assert optical.grad is not None and torch.isfinite(optical.grad).all()
    assert sar.grad is not None and torch.isfinite(sar.grad).all()
    assert model.opt_encoders[0][0].time_embedding.linear1.weight.grad is not None
    assert model.sar_encoders[0][0].time_embedding.linear1.weight.grad is not None


def test_model_rejects_wrong_shapes_and_out_of_range_time():
    model = SourceGuidedDBCRNet(compact_config()).eval()
    optical = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    with pytest.raises(ValueError, match="incorrect"):
        model(optical, torch.tensor([3.]), torch.randn(1, 3, 16, 16))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 13, 17, 16), torch.tensor([3.]), torch.randn(1, 2, 17, 16))
    with pytest.raises(ValueError, match="in"):
        model(optical, torch.tensor([1001.]), sar)


def test_model_identity_and_audit_report_unknowns():
    audit = SourceGuidedDBCRNet(compact_config()).architecture_audit()
    assert audit["model_identity"] == SOURCE_GUIDED_DBCR_IDENTITY
    assert audit["status"] == "source_guided_not_official_paper_exact"
    assert audit["num_time_conditioned_nafblocks"] == 12
    assert audit["unresolved_assumptions"]
    assert audit["architecture_config"]["time_sharing"] == "per_block"
