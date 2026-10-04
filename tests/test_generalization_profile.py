import json
import math
from pathlib import Path

import pytest

from mscloudr.cli.generalization_profile import (
    _default_output_path,
    _load_train_profile,
    _lookup_train_l1,
    build_parser,
    summarize_group_losses,
)


def test_parser_defaults_to_full_alpha_grid_and_val_group_cvar():
    args = build_parser().parse_args(["--checkpoint", "checkpoint.pt"])
    assert args.alphas == "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    assert math.isclose(args.cvar_fraction, 0.20)


def test_default_output_path_for_checkpoint_directory():
    checkpoint = Path("outputs/run/checkpoints/best_endpoint.pt")
    assert _default_output_path(checkpoint) == Path(
        "outputs/run/generalization_profile_val_groups.json"
    )


def test_group_summary_equal_weights_and_cvar():
    summary = summarize_group_losses(
        sample_losses=[1.0, 1.0, 2.0, 4.0, 4.0, 8.0],
        group_losses={
            "g1": [1.0, 1.0],
            "g2": [2.0],
            "g3": [4.0, 4.0],
            "g4": [8.0],
        },
        group_seasons={
            "g1": "spring",
            "g2": "spring",
            "g3": "summer",
            "g4": "winter",
        },
        cvar_fraction=0.25,
    )
    assert summary["num_samples"] == 6
    assert summary["num_groups"] == 4
    assert math.isclose(summary["sample_mean_l1"], 20.0 / 6.0)
    assert math.isclose(summary["equal_group_mean_l1"], 15.0 / 4.0)
    assert summary["worst_group"] == "g4"
    assert math.isclose(summary["worst_group_l1"], 8.0)
    assert summary["cvar_group_count"] == 1
    assert math.isclose(summary["cvar_group_l1"], 8.0)
    assert summary["cvar_groups"] == ["g4"]
    assert math.isclose(summary["season_equal_group_mean_l1"]["spring"], 1.5)


def test_group_cvar_uses_ceiling_for_fraction():
    summary = summarize_group_losses(
        sample_losses=[1.0, 2.0, 3.0, 4.0, 5.0],
        group_losses={f"g{i}": [float(i)] for i in range(1, 6)},
        group_seasons={f"g{i}": "spring" for i in range(1, 6)},
        cvar_fraction=0.21,
    )
    assert summary["cvar_group_count"] == 2
    assert math.isclose(summary["cvar_group_l1"], 4.5)


def test_group_summary_rejects_invalid_cvar_fraction():
    with pytest.raises(ValueError, match="cvar-fraction"):
        summarize_group_losses(
            sample_losses=[1.0],
            group_losses={"g1": [1.0]},
            group_seasons={"g1": "spring"},
            cvar_fraction=0.0,
        )


def test_load_train_profile_and_lookup(tmp_path):
    path = tmp_path / "difficulty_profile_train.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {"requested_alpha": 0.0, "metrics": {"L1": 0.01}},
                    {"requested_alpha": 0.5, "metrics": {"L1": 0.02}},
                    {"requested_alpha": 1.0, "metrics": {"L1": 0.04}},
                ]
            }
        ),
        encoding="utf-8",
    )
    mapping = _load_train_profile(path)
    assert math.isclose(_lookup_train_l1(mapping, 0.5), 0.02)
    assert _lookup_train_l1(mapping, 0.25) is None
