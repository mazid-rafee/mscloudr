import torch

from mscloudr.data.sen12mscr import normalize_optical, normalize_sar


def test_optical_normalization_clips_and_scales():
    x = torch.tensor([-100.0, 0.0, 5000.0, 10000.0, 12000.0])
    actual = normalize_optical(x)
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    assert torch.allclose(actual, expected)


def test_sar_normalization_clips_and_scales():
    x = torch.tensor(
        [
            [[-30.0, -25.0, -12.5, 1.0]],
            [[-40.0, -32.5, -16.25, 1.0]],
        ]
    )
    actual = normalize_sar(x)
    expected = torch.tensor(
        [
            [[0.0, 0.0, 0.5, 1.0]],
            [[0.0, 0.0, 0.5, 1.0]],
        ]
    )
    assert torch.allclose(actual, expected)


from mscloudr.data.sen12mscr import discover_sen12mscr, resolve_dataset_root


def _touch_triplet(root, season="ROIs2017_winter", roi="1", patch="p1"):
    cloudy_dir = root / f"{season}_s2_cloudy" / f"s2_cloudy_{roi}"
    sar_dir = root / f"{season}_s1" / f"s1_{roi}"
    target_dir = root / f"{season}_s2" / f"s2_{roi}"
    cloudy_dir.mkdir(parents=True, exist_ok=True)
    sar_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    cloudy = cloudy_dir / f"{season}_s2_cloudy_{roi}_{patch}.tif"
    sar = sar_dir / f"{season}_s1_{roi}_{patch}.tif"
    target = target_dir / f"{season}_s2_{roi}_{patch}.tif"
    cloudy.touch()
    sar.touch()
    target.touch()
    return cloudy


def test_discovery_uses_stable_relative_sample_ids(tmp_path):
    dataset_root = tmp_path / "SEN12MS-CR"
    cloudy = _touch_triplet(dataset_root)

    report = discover_sen12mscr(dataset_root, seasons=["winter"])

    assert len(report.samples) == 1
    sample = report.samples[0]
    assert sample.sample_id == cloudy.relative_to(dataset_root).as_posix()
    assert sample.roi_id == "1"
    assert sample.patch_id == "p1"


def test_root_resolver_accepts_dataset_parent(tmp_path):
    dataset_root = tmp_path / "SEN12MS-CR"
    _touch_triplet(dataset_root)
    assert resolve_dataset_root(tmp_path) == dataset_root.resolve()


def test_discovery_can_ignore_by_relative_sample_id(tmp_path):
    dataset_root = tmp_path / "SEN12MS-CR"
    cloudy = _touch_triplet(dataset_root)
    sample_id = cloudy.relative_to(dataset_root).as_posix()

    report = discover_sen12mscr(
        dataset_root,
        seasons=["winter"],
        ignored_sample_ids={sample_id},
    )

    assert len(report.samples) == 0
    assert report.ignored_sample_ids == (sample_id,)
