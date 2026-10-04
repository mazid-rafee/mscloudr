import math

import pytest
import torch

from mscloudr.cli.gradient_profile import (
    DEFAULT_MAX_BATCHES,
    build_parser,
    full_model_gradient_l2,
    parse_alphas,
    summarize_batch_values,
    validate_checkpoint_metadata,
)
from mscloudr.data.pilot import PILOT_PROTOCOL
from mscloudr.training import (
    PHYSICAL_ALPHA_CONDITIONING,
    RAW_T_CONDITIONING,
)


def _metadata(conditioning_mode=RAW_T_CONDITIONING):
    return {
        "split_protocol": PILOT_PROTOCOL,
        "schedule_name": "canonical_alpha",
        "conditioning_mode": conditioning_mode,
    }


def test_parser_defaults_to_tenth_alpha_grid_and_256_batches():
    args = build_parser().parse_args(["--checkpoint", "checkpoint.pt"])
    assert args.alphas == "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    assert args.max_batches == DEFAULT_MAX_BATCHES == 256
    assert args.batch_size == 4


def test_parse_alphas_rejects_values_outside_unit_interval():
    assert parse_alphas("0,0.5,1") == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="\[0,1\]"):
        parse_alphas("0,1.1")


def test_canonical_raw_t_checkpoint_is_valid_for_gradient_profile():
    assert validate_checkpoint_metadata(_metadata()) == RAW_T_CONDITIONING


def test_canonical_physical_alpha_checkpoint_is_valid_for_gradient_profile():
    assert (
        validate_checkpoint_metadata(_metadata(PHYSICAL_ALPHA_CONDITIONING))
        == PHYSICAL_ALPHA_CONDITIONING
    )


def test_noncanonical_checkpoint_is_rejected():
    metadata = _metadata()
    metadata["schedule_name"] = "original"
    with pytest.raises(ValueError, match="canonical_alpha"):
        validate_checkpoint_metadata(metadata)


def test_nonpilot_checkpoint_is_rejected():
    metadata = _metadata()
    metadata["split_protocol"] = "other"
    with pytest.raises(ValueError, match="pilot10"):
        validate_checkpoint_metadata(metadata)


def test_full_model_gradient_l2_matches_known_linear_gradient():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    x = torch.tensor([[1.0, 2.0]])
    loss = model(x).sum()
    loss.backward()
    assert math.isclose(full_model_gradient_l2(model), math.sqrt(5.0), rel_tol=1e-7)


def test_gradient_l2_requires_existing_gradients():
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ValueError, match="no parameter gradients"):
        full_model_gradient_l2(model)


def test_summary_reports_population_variance_and_rms():
    summary = summarize_batch_values([1.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["std_population"] == 1.0
    assert summary["variance_population"] == 1.0
    assert math.isclose(summary["rms"], math.sqrt(5.0), rel_tol=1e-9)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["coefficient_of_variation"] == 0.5
