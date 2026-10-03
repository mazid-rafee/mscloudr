import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mscloudr.bridge import sine_alpha
from mscloudr.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from mscloudr.reproducibility import make_torch_generator
from mscloudr.runner import (
    HISTORICAL_LEARNING_RATE,
    fit,
    make_adam_optimizer,
    run_training_epoch,
    run_validation_epoch,
)


class TinyDataset(Dataset):
    def __init__(self, n=6):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        value = float(index + 1) / float(self.n + 1)
        cloudy = torch.full(
            (1, 2, 2),
            value,
        )
        target = torch.full(
            (1, 2, 2),
            0.5 * value,
        )
        sar = torch.zeros(
            1,
            2,
            2,
        )
        return {
            "cloudy": cloudy,
            "sar": sar,
            "target": target,
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(0.8)
        )

    def forward(self, x_t, t, sar):
        del t, sar
        return self.scale * x_t


def _loader(seed, *, shuffle, n=6):
    return DataLoader(
        TinyDataset(n=n),
        batch_size=2,
        shuffle=shuffle,
        generator=make_torch_generator(seed),
        num_workers=0,
    )


def _load_raw(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def test_historical_optimizer_default_learning_rate():
    optimizer = make_adam_optimizer(
        TinyModel()
    )
    assert optimizer.param_groups[0]["lr"] == HISTORICAL_LEARNING_RATE
    assert HISTORICAL_LEARNING_RATE == 5e-5


def test_training_epoch_updates_model_and_returns_finite_l1():
    model = TinyModel()
    loader = _loader(
        10,
        shuffle=True,
    )
    optimizer = make_adam_optimizer(
        model,
        lr=1e-2,
    )
    before = model.scale.detach().clone()

    mean_l1 = run_training_epoch(
        model,
        loader,
        optimizer,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(20),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(
        torch.tensor(mean_l1)
    )
    assert not torch.equal(
        before,
        model.scale.detach(),
    )


def test_validation_epoch_reports_random_and_endpoint_metrics():
    model = TinyModel()
    loader = _loader(
        11,
        shuffle=False,
    )

    metrics = run_validation_epoch(
        model,
        loader,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(21),
        device=torch.device("cpu"),
    )

    assert set(metrics) == {
        "val_random_t_l1",
        "val_endpoint_l1",
    }
    assert metrics["val_random_t_l1"] >= 0.0
    assert metrics["val_endpoint_l1"] >= 0.0


def test_checkpoint_roundtrip_restores_sampler_and_loader_generator_states(tmp_path):
    model = TinyModel()
    optimizer = make_adam_optimizer(
        model,
        lr=1e-3,
    )
    train_loader = _loader(
        30,
        shuffle=True,
    )
    val_loader = _loader(
        31,
        shuffle=False,
    )
    sampler = make_torch_generator(
        32
    )

    _ = torch.randint(
        0,
        100,
        (5,),
        generator=sampler,
    )
    _ = list(
        iter(train_loader.sampler)
    )

    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={
            "val_endpoint_l1": 0.2,
            "val_random_t_l1": 0.1,
        },
        best_endpoint_l1=0.2,
        sampler_generator=sampler,
        loaders={
            "train": train_loader,
            "val": val_loader,
        },
        run_metadata={
            "schedule_name": "original",
        },
    )

    expected_sampler = torch.randint(
        0,
        1000,
        (16,),
        generator=sampler,
    )
    expected_loader = torch.randint(
        0,
        1000,
        (16,),
        generator=train_loader.generator,
    )

    with torch.no_grad():
        model.scale.fill_(99.0)
    _ = torch.randint(
        0,
        1000,
        (50,),
        generator=sampler,
    )
    _ = torch.randint(
        0,
        1000,
        (50,),
        generator=train_loader.generator,
    )

    payload = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        sampler_generator=sampler,
        loaders={
            "train": train_loader,
            "val": val_loader,
        },
        map_location="cpu",
        restore_rng=True,
    )

    actual_sampler = torch.randint(
        0,
        1000,
        (16,),
        generator=sampler,
    )
    actual_loader = torch.randint(
        0,
        1000,
        (16,),
        generator=train_loader.generator,
    )

    assert payload["epoch"] == 1
    assert float(model.scale.detach().item()) != 99.0
    assert torch.equal(
        expected_sampler,
        actual_sampler,
    )
    assert torch.equal(
        expected_loader,
        actual_loader,
    )


def test_fit_saves_latest_and_endpoint_selected_checkpoint_only(tmp_path):
    torch.manual_seed(100)
    model = TinyModel()
    train_loader = _loader(
        40,
        shuffle=True,
    )
    val_loader = _loader(
        41,
        shuffle=False,
    )

    history = fit(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=sine_alpha,
        schedule_name="original",
        total_steps=1000,
        sampler_generator=make_torch_generator(42),
        output_dir=tmp_path,
        epochs=2,
        device="cpu",
        lr=1e-2,
        run_metadata={
            "train_seed": 40,
            "sampler_seed": 42,
        },
    )

    assert len(history) == 2

    latest = tmp_path / "checkpoints" / "latest.pt"
    best = tmp_path / "checkpoints" / "best_endpoint.pt"
    best_random = tmp_path / "checkpoints" / "best_random.pt"

    assert latest.is_file()
    assert best.is_file()
    assert not best_random.exists()

    latest_payload = _load_raw(latest)
    best_payload = _load_raw(best)

    assert latest_payload["epoch"] == 2
    assert (
        latest_payload["run_metadata"]["checkpoint_selection_metric"]
        == "val_endpoint_l1"
    )
    assert (
        latest_payload["run_metadata"]["diagnostic_random_t_metric"]
        == "val_random_t_l1"
    )
    assert latest_payload["run_metadata"]["model_identity"] == "legacy_dbcr"
    assert latest_payload["run_metadata"]["schedule_name"] == "original"

    best_history_value = min(
        item.val_endpoint_l1
        for item in history
    )
    assert best_payload["best_endpoint_l1"] == best_history_value
    assert (
        best_payload["metrics"]["val_endpoint_l1"]
        == best_history_value
    )


def test_resume_matches_uninterrupted_two_epoch_training(tmp_path):
    torch.manual_seed(555)
    full_model = TinyModel()
    full_train = _loader(
        50,
        shuffle=True,
    )
    full_val = _loader(
        51,
        shuffle=False,
    )

    full_history = fit(
        full_model,
        train_loader=full_train,
        val_loader=full_val,
        schedule=sine_alpha,
        schedule_name="original",
        total_steps=1000,
        sampler_generator=make_torch_generator(52),
        output_dir=tmp_path / "full",
        epochs=2,
        device="cpu",
        lr=1e-2,
    )
    full_state = {
        key: value.detach().clone()
        for key, value in full_model.state_dict().items()
    }

    torch.manual_seed(555)
    split_model = TinyModel()
    split_train = _loader(
        50,
        shuffle=True,
    )
    split_val = _loader(
        51,
        shuffle=False,
    )
    split_sampler = make_torch_generator(
        52
    )

    first_history = fit(
        split_model,
        train_loader=split_train,
        val_loader=split_val,
        schedule=sine_alpha,
        schedule_name="original",
        total_steps=1000,
        sampler_generator=split_sampler,
        output_dir=tmp_path / "split",
        epochs=1,
        device="cpu",
        lr=1e-2,
    )
    assert len(first_history) == 1

    resumed_model = TinyModel()
    resumed_train = _loader(
        50,
        shuffle=True,
    )
    resumed_val = _loader(
        51,
        shuffle=False,
    )
    resumed_sampler = make_torch_generator(
        52
    )

    resumed_history = fit(
        resumed_model,
        train_loader=resumed_train,
        val_loader=resumed_val,
        schedule=sine_alpha,
        schedule_name="original",
        total_steps=1000,
        sampler_generator=resumed_sampler,
        output_dir=tmp_path / "split",
        epochs=2,
        device="cpu",
        lr=1e-2,
        resume_from=(
            tmp_path
            / "split"
            / "checkpoints"
            / "latest.pt"
        ),
    )

    assert len(full_history) == 2
    assert len(resumed_history) == 2

    for key, expected in full_state.items():
        assert torch.equal(
            resumed_model.state_dict()[key],
            expected,
        )

    for full_item, resumed_item in zip(
        full_history,
        resumed_history,
    ):
        assert (
            full_item.to_dict()
            == resumed_item.to_dict()
        )

    history_file = json.loads(
        (
            tmp_path
            / "split"
            / "history.json"
        ).read_text()
    )
    assert [
        item["epoch"]
        for item in history_file
    ] == [1, 2]



class CountingTinyModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, x_t, t, sar):
        self.forward_calls += 1
        return super().forward(
            x_t,
            t,
            sar,
        )


def test_training_epoch_respects_max_batches():
    model = CountingTinyModel()
    loader = _loader(
        60,
        shuffle=True,
        n=6,
    )
    optimizer = make_adam_optimizer(
        model,
        lr=1e-2,
    )

    run_training_epoch(
        model,
        loader,
        optimizer,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(61),
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert model.forward_calls == 1


def test_validation_epoch_respects_max_batches():
    model = CountingTinyModel()
    loader = _loader(
        62,
        shuffle=False,
        n=6,
    )

    run_validation_epoch(
        model,
        loader,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(63),
        device=torch.device("cpu"),
        max_batches=1,
    )

    # One random-t forward plus one endpoint forward.
    assert model.forward_calls == 2



def test_training_progress_reports_interval_and_final_batch():
    model = CountingTinyModel()
    loader = _loader(
        70,
        shuffle=False,
        n=10,
    )
    optimizer = make_adam_optimizer(
        model,
        lr=1e-2,
    )
    events = []

    run_training_epoch(
        model,
        loader,
        optimizer,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(71),
        device=torch.device("cpu"),
        epoch=4,
        progress_every=2,
        progress_callback=events.append,
    )

    assert [
        event.batch
        for event in events
    ] == [2, 4, 5]
    assert all(
        event.epoch == 4
        and event.phase == "train"
        and event.total_batches == 5
        for event in events
    )
    assert all(
        "train_l1_running" in event.metrics
        for event in events
    )


def test_validation_progress_reports_interval_and_final_batch():
    model = CountingTinyModel()
    loader = _loader(
        72,
        shuffle=False,
        n=10,
    )
    events = []

    run_validation_epoch(
        model,
        loader,
        schedule=sine_alpha,
        total_steps=1000,
        sampler_generator=make_torch_generator(73),
        device=torch.device("cpu"),
        epoch=2,
        progress_every=3,
        progress_callback=events.append,
    )

    assert [
        event.batch
        for event in events
    ] == [3, 5]
    assert all(
        event.epoch == 2
        and event.phase == "val"
        and event.total_batches == 5
        for event in events
    )
    assert all(
        {
            "val_random_t_l1_running",
            "val_endpoint_l1_running",
        }
        <= set(event.metrics)
        for event in events
    )
