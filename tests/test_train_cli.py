import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import mscloudr.cli.train as train_cli
from mscloudr.runner import EpochMetrics


def _args(tmp_path, **overrides):
    values = {
        "data_root": None,
        "ignore_file": "manifests/known_invalid_samples.txt",
        "output_root": str(tmp_path),
        "run_name": "unit_run",
        "schedule": "original",
        "epochs": 2,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 5e-5,
        "total_steps": 1000,
        "train_seed": 42,
        "sampler_seed": 42,
        "device": "cpu",
        "deterministic_algorithms": False,
        "smoke_run": False,
        "smoke_train_batches": 2,
        "smoke_val_batches": 2,
        "progress_every": 500,
        "resume": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parser_exposes_only_controlled_schedules():
    parser = train_cli.build_parser()

    original = parser.parse_args(
        [
            "--run-name",
            "original",
            "--schedule",
            "original",
        ]
    )
    mr = parser.parse_args(
        [
            "--run-name",
            "mr",
            "--schedule",
            "mr_r3",
        ]
    )

    assert original.schedule == "original"
    assert mr.schedule == "mr_r3"


def test_mr_r3_cli_identity_fixes_rate_to_three():
    name, schedule, rate = train_cli.canonical_schedule(
        "mr_r3"
    )

    assert name == "mr_r3"
    assert rate == 3.0

    t = torch.tensor([500.0])
    actual = schedule(t, 1000)
    expected = (
        torch.expm1(
            torch.tensor([-1.5])
        )
        / torch.expm1(
            torch.tensor([-3.0])
        )
    )
    assert torch.allclose(
        actual,
        expected,
    )


def test_run_name_cannot_escape_output_root(tmp_path):
    args = _args(
        tmp_path,
        run_name="../escape",
    )

    try:
        train_cli.validate_cli_args(
            args
        )
    except ValueError as error:
        assert "single directory" in str(
            error
        )
    else:
        raise AssertionError(
            "expected invalid run-name to fail"
        )


def test_resume_metadata_rejects_schedule_change(tmp_path):
    args = _args(
        tmp_path,
        schedule="original",
    )
    split_audit = {
        "protocol": "test",
        "splits": {},
    }

    previous = train_cli._run_metadata(
        args=args,
        dataset_root=Path("/data"),
        split_audit=split_audit,
        trainable_parameters=(
            train_cli.LEGACY_DBCR_PARAMETER_COUNT
        ),
        schedule_name="original",
        mean_reversion_rate=None,
        device=torch.device("cpu"),
    )

    changed_args = _args(
        tmp_path,
        schedule="mr_r3",
    )
    changed = train_cli._run_metadata(
        args=changed_args,
        dataset_root=Path("/data"),
        split_audit=split_audit,
        trainable_parameters=(
            train_cli.LEGACY_DBCR_PARAMETER_COUNT
        ),
        schedule_name="mr_r3",
        mean_reversion_rate=3.0,
        device=torch.device("cpu"),
    )

    try:
        train_cli.validate_resume_metadata(
            previous,
            changed,
        )
    except ValueError as error:
        assert "schedule_name" in str(
            error
        )
    else:
        raise AssertionError(
            "expected schedule-changing resume to fail"
        )


def test_resume_metadata_allows_epoch_target_and_device_change(tmp_path):
    first_args = _args(
        tmp_path,
        epochs=1,
    )
    second_args = _args(
        tmp_path,
        epochs=50,
    )
    split_audit = {
        "protocol": "test",
        "splits": {},
    }

    first = train_cli._run_metadata(
        args=first_args,
        dataset_root=Path("/first"),
        split_audit=split_audit,
        trainable_parameters=(
            train_cli.LEGACY_DBCR_PARAMETER_COUNT
        ),
        schedule_name="original",
        mean_reversion_rate=None,
        device=torch.device("cpu"),
    )
    second = train_cli._run_metadata(
        args=second_args,
        dataset_root=Path("/second"),
        split_audit=split_audit,
        trainable_parameters=(
            train_cli.LEGACY_DBCR_PARAMETER_COUNT
        ),
        schedule_name="original",
        mean_reversion_rate=None,
        device=torch.device("cpu"),
    )

    train_cli.validate_resume_metadata(
        first,
        second,
    )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(1.0)
        )


class _FakeDatasets:
    def __init__(self):
        self.train = [0, 1, 2]
        self.val = [0, 1]
        self.test = [0]

    def as_dict(self):
        return {
            "train": SimpleNamespace(
                samples=["train"]
            ),
            "val": SimpleNamespace(
                samples=["val"]
            ),
            "test": SimpleNamespace(
                samples=["test"]
            ),
        }


def test_run_wires_verified_protocol_into_fit(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(
        train_cli,
        "load_ignored_sample_ids",
        lambda path: {"bad"},
    )
    monkeypatch.setattr(
        train_cli,
        "discover_sen12mscr",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_root=Path("/verified/data"),
            samples=["all"],
        ),
    )
    monkeypatch.setattr(
        train_cli,
        "build_reference_datasets",
        lambda *args, **kwargs: _FakeDatasets(),
    )
    monkeypatch.setattr(
        train_cli,
        "reference_split_audit",
        lambda partitions: {
            "protocol": train_cli.REFERENCE_PROTOCOL,
            "dataset_num_samples_after_ignored": 122217,
            "splits": {
                "train": {"num_samples": 107142},
                "val": {"num_samples": 7176},
                "test": {"num_samples": 7899},
            },
        },
    )
    monkeypatch.setattr(
        train_cli,
        "build_reference_dataloaders",
        lambda *args, **kwargs: SimpleNamespace(
            train="train_loader",
            val="val_loader",
            test="test_loader",
        ),
    )
    monkeypatch.setattr(
        train_cli,
        "LegacyDBCRNet",
        _TinyModel,
    )
    monkeypatch.setattr(
        train_cli,
        "count_legacy_parameters",
        lambda model: train_cli.LEGACY_DBCR_PARAMETER_COUNT,
    )

    def fake_fit(model, **kwargs):
        captured.update(kwargs)
        return [
            EpochMetrics(
                epoch=1,
                train_l1=0.3,
                val_random_t_l1=0.2,
                val_endpoint_l1=0.25,
            )
        ]

    monkeypatch.setattr(
        train_cli,
        "fit",
        fake_fit,
    )

    args = _args(
        tmp_path,
        epochs=1,
        schedule="original",
    )
    history = train_cli.run(
        args
    )

    assert len(history) == 1
    assert captured["train_loader"] == "train_loader"
    assert captured["val_loader"] == "val_loader"
    assert captured["schedule_name"] == "original"
    assert captured["model_identity"] == "legacy_dbcr"
    assert captured["total_steps"] == 1000
    assert captured["resume_from"] is None
    assert captured["max_train_batches"] is None
    assert captured["max_val_batches"] is None

    config = (
        Path(tmp_path)
        / "unit_run"
        / "run_config.json"
    )
    assert config.is_file()



def test_smoke_run_requires_one_epoch_and_rejects_resume(tmp_path):
    wrong_epochs = _args(
        tmp_path,
        smoke_run=True,
        epochs=2,
    )
    try:
        train_cli.validate_cli_args(
            wrong_epochs
        )
    except ValueError as error:
        assert "requires --epochs 1" in str(
            error
        )
    else:
        raise AssertionError(
            "expected multi-epoch smoke run to fail"
        )

    resume_smoke = _args(
        tmp_path,
        smoke_run=True,
        epochs=1,
        resume="checkpoint.pt",
    )
    try:
        train_cli.validate_cli_args(
            resume_smoke
        )
    except ValueError as error:
        assert "does not support --resume" in str(
            error
        )
    else:
        raise AssertionError(
            "expected smoke resume to fail"
        )


def test_smoke_metadata_is_explicitly_non_paper_grade(tmp_path):
    args = _args(
        tmp_path,
        smoke_run=True,
        epochs=1,
        smoke_train_batches=3,
        smoke_val_batches=4,
    )
    metadata = train_cli._run_metadata(
        args=args,
        dataset_root=Path("/data"),
        split_audit={
            "protocol": "test",
            "splits": {},
        },
        trainable_parameters=(
            train_cli.LEGACY_DBCR_PARAMETER_COUNT
        ),
        schedule_name="original",
        mean_reversion_rate=None,
        device=torch.device("cpu"),
    )

    assert metadata["run_kind"] == "smoke"
    assert metadata["paper_grade"] is False
    assert metadata["max_train_batches"] == 3
    assert metadata["max_val_batches"] == 4



def test_progress_every_zero_is_allowed_and_negative_is_rejected(tmp_path):
    disabled = _args(
        tmp_path,
        progress_every=0,
    )
    train_cli.validate_cli_args(
        disabled
    )

    invalid = _args(
        tmp_path,
        progress_every=-1,
    )
    try:
        train_cli.validate_cli_args(
            invalid
        )
    except ValueError as error:
        assert "progress-every" in str(
            error
        )
    else:
        raise AssertionError(
            "expected negative progress interval to fail"
        )


def test_batch_progress_lines_are_grep_friendly(capsys):
    train_cli._batch_progress(
        train_cli.BatchProgress(
            epoch=3,
            phase="train",
            batch=500,
            total_batches=26786,
            metrics={
                "train_l1_running": 0.0123456,
            },
        )
    )
    train_cli._batch_progress(
        train_cli.BatchProgress(
            epoch=3,
            phase="val",
            batch=500,
            total_batches=1794,
            metrics={
                "val_random_t_l1_running": 0.02,
                "val_endpoint_l1_running": 0.03,
            },
        )
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        (
            "epoch=3 phase=train batch=500/26786 "
            "train_l1_running=0.012346"
        ),
        (
            "epoch=3 phase=val batch=500/1794 "
            "val_random_t_l1_running=0.020000 "
            "val_endpoint_l1_running=0.030000"
        ),
    ]
