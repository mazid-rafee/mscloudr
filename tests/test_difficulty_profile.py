from pathlib import Path

import pytest

from mscloudr.cli.difficulty_profile import (
    _default_profile_output,
    build_parser,
)


def test_difficulty_profile_defaults_to_train_split():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "checkpoint.pt"])
    assert args.split == "train"


def test_difficulty_profile_rejects_held_out_split_at_cli():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--split",
                "val",
            ]
        )


def test_difficulty_profile_default_alpha_grid_is_0_to_1_by_tenths():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "checkpoint.pt"])
    assert args.alphas == "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def test_default_profile_output_for_checkpoint_directory():
    checkpoint = Path("outputs/run/checkpoints/best_endpoint.pt")
    assert _default_profile_output(checkpoint) == Path(
        "outputs/run/difficulty_profile_train.json"
    )


def test_default_profile_output_for_standalone_checkpoint():
    checkpoint = Path("best_endpoint.pt")
    assert _default_profile_output(checkpoint) == Path(
        "difficulty_profile_train.json"
    )
