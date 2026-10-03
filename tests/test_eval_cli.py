import argparse
from pathlib import Path

import mscloudr.cli.eval as eval_cli


def _metadata(**overrides):
    value = {
        "model_identity": "legacy_dbcr",
        "split_protocol": eval_cli.REFERENCE_PROTOCOL,
        "schedule_name": "original",
        "paper_grade": True,
        "total_steps": 1000,
        "reproducibility": {
            "seeds": {
                "train_seed": 42,
            },
            "deterministic_algorithms": False,
        },
    }
    value.update(overrides)
    return value


def _payload(metadata=None):
    return {
        "format_version": eval_cli.CHECKPOINT_FORMAT_VERSION,
        "model_state": {},
        "run_metadata": (
            _metadata()
            if metadata is None
            else metadata
        ),
    }


def test_parser_does_not_request_schedule_identity():
    parser = eval_cli.build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "run/checkpoints/best_endpoint.pt",
        ]
    )

    assert args.checkpoint.endswith(
        "best_endpoint.pt"
    )
    assert not hasattr(args, "schedule")


def test_checkpoint_validation_accepts_controlled_paper_checkpoint():
    metadata = eval_cli._validate_checkpoint(
        _payload(),
        allow_smoke_checkpoint=False,
    )

    assert metadata["model_identity"] == "legacy_dbcr"
    assert metadata["schedule_name"] == "original"
    assert metadata["total_steps"] == 1000


def test_checkpoint_validation_rejects_smoke_without_opt_in():
    payload = _payload(
        _metadata(
            paper_grade=False,
        )
    )

    try:
        eval_cli._validate_checkpoint(
            payload,
            allow_smoke_checkpoint=False,
        )
    except ValueError as error:
        assert "non-paper-grade" in str(
            error
        )
    else:
        raise AssertionError(
            "expected smoke checkpoint to be rejected"
        )

    accepted = eval_cli._validate_checkpoint(
        payload,
        allow_smoke_checkpoint=True,
    )
    assert accepted["paper_grade"] is False


def test_default_output_path_uses_run_directory():
    checkpoint = Path(
        "outputs/run/checkpoints/best_endpoint.pt"
    )
    assert eval_cli._default_output_path(
        checkpoint
    ) == Path(
        "outputs/run/eval_metrics.json"
    )


def test_validate_cli_args_marks_invalid_diagnostic_limit():
    args = argparse.Namespace(
        batch_size=4,
        num_workers=4,
        progress_every=100,
        max_batches=0,
    )

    try:
        eval_cli.validate_cli_args(args)
    except ValueError as error:
        assert "max-batches" in str(
            error
        )
    else:
        raise AssertionError(
            "expected invalid max-batches to fail"
        )
