import pytest

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli.eval_canonical_bridge import _validate_checkpoint
from mscloudr.data import REFERENCE_PROTOCOL
from mscloudr.models import CANONICAL_BRIDGE_MODEL_IDENTITY
from mscloudr.training import (
    MR_INVERSE_CONDITIONING,
    PHYSICAL_ALPHA_CONDITIONING,
    SINE_INVERSE_CONDITIONING,
)


def _payload(conditioning_mode=PHYSICAL_ALPHA_CONDITIONING):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": {},
        "run_metadata": {
            "model_identity": CANONICAL_BRIDGE_MODEL_IDENTITY,
            "split_protocol": REFERENCE_PROTOCOL,
            "schedule_name": "canonical_alpha",
            "bridge_geometry": "straight_linear",
            "conditioning_mode": conditioning_mode,
            "total_steps": 1000,
            "beta_a": 1.0,
            "paper_grade": True,
        },
    }


@pytest.mark.parametrize(
    "conditioning_mode",
    [
        PHYSICAL_ALPHA_CONDITIONING,
        SINE_INVERSE_CONDITIONING,
        MR_INVERSE_CONDITIONING,
    ],
)
def test_full_canonical_eval_accepts_all_controlled_conditioning_modes(
    conditioning_mode,
):
    metadata = _validate_checkpoint(
        _payload(conditioning_mode),
        allow_smoke_checkpoint=False,
    )
    assert metadata["conditioning_mode"] == conditioning_mode


def test_full_canonical_eval_rejects_non_reference_split():
    payload = _payload()
    payload["run_metadata"]["split_protocol"] = "pilot10"
    with pytest.raises(ValueError, match="frozen full reference split"):
        _validate_checkpoint(payload, allow_smoke_checkpoint=False)


def test_full_canonical_eval_rejects_wrong_model_identity():
    payload = _payload()
    payload["run_metadata"]["model_identity"] = "legacy_dbcr"
    with pytest.raises(ValueError, match="canonical_bridge_net"):
        _validate_checkpoint(payload, allow_smoke_checkpoint=False)


def test_full_canonical_eval_rejects_smoke_checkpoint_by_default():
    payload = _payload()
    payload["run_metadata"]["paper_grade"] = False
    with pytest.raises(ValueError, match="non-paper-grade"):
        _validate_checkpoint(payload, allow_smoke_checkpoint=False)
