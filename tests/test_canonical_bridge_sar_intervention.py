import torch

from mscloudr.cli.eval_canonical_bridge_sar_intervention import (
    CONDITIONS,
    _assert_output_schema,
    derangement_fixed_points,
    deterministic_noise_sar,
    intervention_sar,
    make_derangement,
)


def test_derangement_is_deterministic_and_has_zero_fixed_points():
    first = make_derangement(100, seed=42)
    second = make_derangement(100, seed=42)

    assert first == second
    assert sorted(first) == list(range(100))
    assert derangement_fixed_points(first) == 0


def test_derangement_changes_with_seed():
    assert make_derangement(50, seed=42) != make_derangement(50, seed=43)


def test_zero_intervention_is_exact_zero_without_touching_normal():
    normal = torch.rand(3, 2, 4, 4)
    original = normal.clone()

    zero = intervention_sar("ZERO", normal)

    assert torch.equal(zero, torch.zeros_like(normal))
    assert torch.equal(normal, original)


def test_shuffled_intervention_uses_supplied_sar_only():
    normal = torch.zeros(2, 2, 4, 4)
    shuffled = torch.ones_like(normal)

    result = intervention_sar(
        "SHUFFLED",
        normal,
        shuffled_sar=shuffled,
    )

    assert torch.equal(result, shuffled)
    assert torch.equal(normal, torch.zeros_like(normal))


def test_noise_is_deterministic_per_sample_and_batch_order_stable():
    reference = torch.zeros(3, 2, 4, 4)
    ids = ["a", "b", "c"]

    first = deterministic_noise_sar(reference, ids, seed=42)
    second = deterministic_noise_sar(reference, ids, seed=42)
    reversed_batch = deterministic_noise_sar(
        reference,
        list(reversed(ids)),
        seed=42,
    )

    assert torch.equal(first, second)
    assert torch.equal(first[0], reversed_batch[2])
    assert torch.equal(first[1], reversed_batch[1])
    assert torch.equal(first[2], reversed_batch[0])
    assert torch.all(first >= 0.0)
    assert torch.all(first <= 1.0)


def test_intervention_does_not_require_or_modify_optical_target():
    cloudy = torch.rand(2, 13, 4, 4)
    target = torch.rand(2, 13, 4, 4)
    cloudy_before = cloudy.clone()
    target_before = target.clone()
    normal_sar = torch.rand(2, 2, 4, 4)
    shuffled_sar = normal_sar.flip(0)

    _ = intervention_sar(
        "SHUFFLED",
        normal_sar,
        shuffled_sar=shuffled_sar,
    )

    assert torch.equal(cloudy, cloudy_before)
    assert torch.equal(target, target_before)


def test_output_schema_accepts_required_intervention_fields():
    payload = {
        "checkpoint": "best_endpoint.pt",
        "checkpoint_epoch": 24,
        "model_identity": "canonical_bridge_net",
        "trainable_parameters": 7_653_037,
        "num_samples": 790,
        "intervention_seed": 42,
        "pilot_manifest_sample_ids_sha256": "abc",
        "intervention_definitions": {name: name for name in CONDITIONS},
        "metrics": {name: {"L1": 0.1} for name in CONDITIONS},
        "deltas_vs_normal": {},
        "prediction_changes_vs_normal": {},
        "shuffled_permutation_fixed_points": 0,
        "shuffled_is_derangement": True,
        "conditioning_mode": "physical_alpha",
        "endpoint_conditioning_value": 1000,
        "nfe": 1,
        "inference": "endpoint_direct_x0",
    }

    _assert_output_schema(payload)
