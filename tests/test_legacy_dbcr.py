import torch

from mscloudr.models.legacy_dbcr import (
    DBCRNet,
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    LegacyNAFBlock,
    LegacySFBlock,
    count_legacy_parameters,
)


def test_legacy_alias_preserves_historical_model_name():
    assert DBCRNet is LegacyDBCRNet


def test_legacy_parameter_count_is_exactly_frozen():
    model = LegacyDBCRNet()
    assert count_legacy_parameters(model) == LEGACY_DBCR_PARAMETER_COUNT
    assert LEGACY_DBCR_PARAMETER_COUNT == 13_588_641


def test_legacy_model_forward_shape_and_finiteness():
    torch.manual_seed(0)
    model = LegacyDBCRNet().eval()

    x_t = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    t = torch.tensor([100.0])

    with torch.no_grad():
        output = model(x_t, t, sar)

    assert output.shape == x_t.shape
    assert torch.isfinite(output).all()


def test_legacy_model_construction_is_seed_deterministic():
    torch.manual_seed(1234)
    first = LegacyDBCRNet()

    torch.manual_seed(1234)
    second = LegacyDBCRNet()

    first_state = first.state_dict()
    second_state = second.state_dict()

    assert first_state.keys() == second_state.keys()
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key])


def test_legacy_checkpoint_schema_keeps_historical_attribute_names():
    model = LegacyDBCRNet()
    keys = set(model.state_dict().keys())

    required_keys = {
        "time_mlp.1.weight",
        "time_mlp.3.weight",
        "time_to_channels.0.weight",
        "opt_stem.weight",
        "sar_stem.weight",
        "opt_enc.0.0.norm1.weight",
        "sar_enc.3.27.conv5.weight",
        "downs.0.weight",
        "fuse.3.q.weight",
        "fuse.3.mlp.2.weight",
        "mid.0.gamma",
        "ups.0.weight",
        "opt_dec.2.0.conv5.weight",
        "head.weight",
    }

    assert required_keys.issubset(keys)


def test_legacy_state_dict_loads_strictly_and_reproduces_output():
    torch.manual_seed(7)
    source = LegacyDBCRNet().eval()

    target = LegacyDBCRNet().eval()
    target.load_state_dict(
        source.state_dict(),
        strict=True,
    )

    x_t = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    t = torch.tensor([1000])

    with torch.no_grad():
        source_output = source(x_t, t, sar)
        target_output = target(x_t, t, sar)

    assert torch.equal(source_output, target_output)


def test_legacy_nafblock_keeps_groupnorm_and_channelwise_residual_scales():
    block = LegacyNAFBlock(22)

    assert isinstance(block.norm1, torch.nn.GroupNorm)
    assert isinstance(block.norm2, torch.nn.GroupNorm)
    assert block.norm1.num_groups == 1
    assert block.norm2.num_groups == 1
    assert block.beta.shape == (1, 22, 1, 1)
    assert block.gamma.shape == (1, 22, 1, 1)


def test_legacy_sfblock_keeps_no_pre_qkv_normalization():
    block = LegacySFBlock(88, 2)

    assert not hasattr(block, "norm_q")
    assert not hasattr(block, "norm_k")
    assert not hasattr(block, "norm_v")
    assert block.heads == 2
    assert block.channels == 88


def test_legacy_downsampler_is_intentionally_shared_between_modalities():
    model = LegacyDBCRNet()

    # There is exactly one downsampler list. The historical forward calls the
    # same module first on optical features and then on SAR features.
    assert hasattr(model, "downs")
    assert not hasattr(model, "opt_downs")
    assert not hasattr(model, "sar_downs")
    assert len(model.downs) == 3
