# Reference DataLoaders

Paper-grade SEN12MS-CR experiments use the fixed ROI-level membership from the
UnCRtainTS reference split. No random_split call is allowed in the clean
training path.

Verified membership after excluding the one documented invalid SAR triplet:

    train  107,142 samples / 155 ROI groups
    val      7,176 samples /  10 ROI groups
    test     7,899 samples /  10 ROI groups
    total  122,217 samples

The corresponding sample-ID SHA256 fingerprints are frozen in
manifests/sen12mscr_reference_split_audit_v1.json and mirrored by the
DataLoader validation code.

## Randomness boundaries

The split itself is fixed and therefore consumes no split RNG.

train_seed controls training DataLoader shuffle order and DataLoader worker
random state.

sampler_seed is not used by the DataLoader layer. It is reserved for bridge
timestep/state sampling.

Validation and test use sequential samplers.

## Intended training setup

After dataset discovery and invalid-sample exclusion, call
build_reference_datasets with verify_frozen_membership=True, then construct
loaders with build_reference_dataloaders using the experiment train_seed.

Keeping membership validation separate from DataLoader construction allows unit
tests to use tiny fixtures while full experiments can require the exact frozen
benchmark fingerprint.

No training loop is introduced in this stage.
