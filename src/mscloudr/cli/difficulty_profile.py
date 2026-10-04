"""Training-split fixed-alpha difficulty profile for adaptive bridge measures.

This diagnostic is intentionally restricted to the CanonicalAlpha pilot model
with physical-alpha conditioning. It evaluates the checkpoint on the pilot
TRAIN split at fixed physical corruption levels alpha and reuses the existing
fixed-alpha metric implementation.

The resulting L1 curve

    E(alpha) = E_train[|R_theta(x_alpha, T*alpha, z) - x0|]

is intended for designing a future adaptive training measure without using
validation or test information.
"""

from __future__ import annotations

from pathlib import Path

from mscloudr.cli import eval as eval_base
from mscloudr.cli import fixed_alpha_sweep
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


def build_parser():
    parser = fixed_alpha_sweep.build_parser()
    parser.description = (
        "Estimate the CanonicalAlpha checkpoint's fixed-alpha restoration "
        "difficulty profile on the pilot TRAIN split."
    )
    split_action = parser._option_string_actions["--split"]
    split_action.choices = ("train",)
    split_action.default = "train"
    split_action.help = (
        "Training split only. Held-out validation/test data are intentionally "
        "excluded from adaptive-sampler design."
    )
    parser.set_defaults(split="train")
    return parser


def _validate_checkpoint_role(checkpoint: str | Path) -> dict:
    payload = eval_base._torch_load_payload(Path(checkpoint))
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing run_metadata")

    if metadata.get("schedule_name") != "canonical_alpha":
        raise ValueError(
            "difficulty-profile design requires a canonical_alpha checkpoint"
        )

    conditioning_mode = fixed_alpha_sweep.conditioning_mode_from_metadata(metadata)
    if conditioning_mode != PHYSICAL_ALPHA_CONDITIONING:
        raise ValueError(
            "difficulty-profile design requires physical_alpha conditioning"
        )

    return metadata


def _default_profile_output(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    filename = "difficulty_profile_train.json"
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / filename
    return checkpoint.with_name(filename)


def run(args):
    if args.split != "train":
        raise ValueError("difficulty profile must use --split train")

    _validate_checkpoint_role(args.checkpoint)

    if args.output is None:
        args.output = str(_default_profile_output(args.checkpoint))

    payload = fixed_alpha_sweep.run(args)
    payload["analysis_role"] = "training_difficulty_profile_for_sampler_design"
    payload["held_out_data_used_for_sampler_design"] = False
    payload["difficulty_definition"] = (
        "E(alpha)=mean_train_L1(R_theta(x_alpha,T*alpha,z),x0)"
    )

    # fixed_alpha_sweep writes once before returning. Rewrite the same file so
    # the sampler-design provenance above is persisted alongside the metrics.
    eval_base._write_json_atomic(
        Path(args.output),
        payload,
        overwrite=True,
    )
    return payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
