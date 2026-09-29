import torch

from mscloudr.models.dbcr import (
    AuditedDBCRNet,
    DBCRArchitectureConfig,
    HISTORICAL_MSCLOUDR_PARAMETER_COUNT,
    PUBLISHED_PARAMETER_COUNT,
)
from mscloudr.models.blocks import count_trainable_parameters


def test_default_architecture_matches_published_discrete_settings():
    cfg = DBCRArchitectureConfig()
    assert cfg.widths == (22, 44, 88, 176)
    assert cfg.encoder_blocks == (1, 1, 1, 28)
    assert cfg.decoder_blocks == (1, 1, 1, 1)
    assert cfg.sf_heads == (1, 1, 2, 4)


def test_audited_dbcr_forward_shape_and_finiteness():
    torch.manual_seed(0)
    model = AuditedDBCRNet().eval()
    x_t = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    t = torch.tensor([10.0])
    with torch.no_grad():
        output = model(x_t, t, sar)
    assert output.shape == x_t.shape
    assert torch.isfinite(output).all()


def test_default_uses_distinct_optical_and_sar_downsamplers():
    model = AuditedDBCRNet()
    assert model.opt_downsamplers is not None
    assert model.sar_downsamplers is not None
    assert model.shared_downsamplers is None
    for optical_down, sar_down in zip(
        model.opt_downsamplers, model.sar_downsamplers
    ):
        assert optical_down is not sar_down
        assert optical_down.weight.data_ptr() != sar_down.weight.data_ptr()


def test_shared_downsampling_mode_is_available_for_legacy_comparison():
    model = AuditedDBCRNet(
        DBCRArchitectureConfig(separate_modality_downsamplers=False)
    )
    assert model.shared_downsamplers is not None
    assert model.opt_downsamplers is None
    assert model.sar_downsamplers is None


def test_parameter_count_is_frozen_for_current_explicit_assumptions():
    assert count_trainable_parameters(AuditedDBCRNet()) == 13_566_113


def test_separate_downsamplers_add_expected_parameters():
    separate = AuditedDBCRNet()
    shared = AuditedDBCRNet(
        DBCRArchitectureConfig(separate_modality_downsamplers=False)
    )
    assert count_trainable_parameters(separate) == 13_566_113
    assert count_trainable_parameters(shared) == 13_484_493
    assert (
        count_trainable_parameters(separate)
        - count_trainable_parameters(shared)
        == 81_620
    )


def test_audit_exposes_large_remaining_gap_to_published_count():
    audit = AuditedDBCRNet().architecture_audit()
    assert audit["published_dbcr_parameters"] == PUBLISHED_PARAMETER_COUNT
    assert audit["historical_mscloudr_parameters"] == (
        HISTORICAL_MSCLOUDR_PARAMETER_COUNT
    )
    assert audit["trainable_parameters"] == 13_566_113
    assert audit["parameter_gap_vs_published"] == 4_493_887


def test_time_conditioning_can_be_disabled_without_changing_shape():
    model = AuditedDBCRNet(
        DBCRArchitectureConfig(time_conditioning="none")
    ).eval()
    x_t = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    t = torch.tensor([3.0])
    with torch.no_grad():
        output = model(x_t, t, sar)
    assert output.shape == x_t.shape
    assert torch.isfinite(output).all()
