"""Monitor ongoing CanonicalBridge endpoint-mixture runs without touching them.

Examples:
  python scripts/monitor_canonical_endpoint.py
  watch -n 5 'python scripts/monitor_canonical_endpoint.py --write-logs'

The current GPU worker logs contain interleaved CNN/NAF batch updates.
This independent monitor reads each run's own history.json and emits
attributable per-EPOCH status logs, without claiming per-batch progress.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess


WORKTREE = Path(__file__).resolve().parent.parent
OUTPUT = (WORKTREE / ".." / "mscloudr" / "outputs").resolve()
RATES = ((0, 4), (10, 5), (25, 6), (50, 7))
BACKBONES = ("CNN", "NAF")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def process_lines():
    try:
        result = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def status(name: str, records: list[dict], test, process_args: list[str]):
    if isinstance(test, dict) and "metrics" in test:
        return "TEST_DONE"
    if any(
        "python -m mscloudr.cli.eval_canonical_bridge" in line
        and f"/{name}/" in line
        for line in process_args
    ):
        return "EVALUATING"
    if any(
        "python -m mscloudr.cli.train_canonical_bridge" in line
        and f"--run-name {name}" in line
        for line in process_args
    ):
        return "TRAINING"
    if records:
        if int(records[-1]["epoch"]) >= 50:
            return "TRAIN_DONE"
        return "STOPPED?"
    return "NOT_STARTED"


def existing_epochs(path: Path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set(), False
    found = set()
    completed = False
    for line in lines:
        for field in line.split():
            if field.startswith("epoch="):
                try:
                    found.add(int(field.split("=", 1)[1]))
                except ValueError:
                    pass
            if field == "TEST_DONE":
                completed = True
    return found, completed


def write_status_log(path: Path, records: list[dict], test):
    known, test_written = existing_epochs(path)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    messages = []
    for record in records:
        epoch = int(record["epoch"])
        if epoch in known:
            continue
        messages.append(
            f"{stamp} epoch={epoch} "
            f"train_l1={float(record['train_l1']):.6f} "
            f"val_random_t_l1={float(record['val_random_t_l1']):.6f} "
            f"val_endpoint_l1={float(record['val_endpoint_l1']):.6f}"
        )
    if isinstance(test, dict) and "metrics" in test and not test_written:
        metrics = test["metrics"]
        messages.append(
            f"{stamp} TEST_DONE "
            f"PSNR={metrics.get('PSNR')} SSIM={metrics.get('SSIM')} "
            f"L1={metrics.get('L1')} SAM={metrics.get('SAM')}"
        )
    if messages:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(messages) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Show status for all eight CanonicalBridge endpoint-mixture runs."
    )
    parser.add_argument(
        "--write-logs",
        action="store_true",
        help="Append newly completed epochs to eight individual .status.log files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT,
        help="Central original mscloudr/outputs folder.",
    )
    args = parser.parse_args()
    out = args.output_root.expanduser().resolve()
    commands = process_lines()

    print(f"Outputs: {out}")
    print(f"{'GPU':<4} {'MODEL':<5} {'RHO':>3} {'STATUS':<12} "
          f"{'EPOCH':>6} {'TRAIN_L1':>11} {'VAL_END_L1':>11} {'TEST_PSNR':>10}")
    for percent, gpu in RATES:
        for backbone in BACKBONES:
            name = f"CanonicalBridge_{backbone}_Endpoint{percent}_full_seed42"
            root = out / name
            history = read_json(root / "history.json")
            records = history if isinstance(history, list) else []
            test = read_json(root / "eval_metrics.json")

            if args.write_logs:
                write_status_log(out / (name + ".status.log"), records, test)

            state = status(name, records, test, commands)
            if records:
                last = records[-1]
                epoch = str(int(last["epoch"]))
                train_l1 = f"{float(last['train_l1']):.6f}"
                val_l1 = f"{float(last['val_endpoint_l1']):.6f}"
            else:
                epoch, train_l1, val_l1 = "-", "-", "-"
            psnr = (
                f"{float(test['metrics']['PSNR']):.4f}"
                if isinstance(test, dict)
                and isinstance(test.get("metrics"), dict)
                and "PSNR" in test["metrics"]
                else "-"
            )
            print(
                f"{gpu:<4} {backbone:<5} {percent:>3}% {state:<12} "
                f"{epoch:>6} {train_l1:>11} {val_l1:>11} {psnr:>10}"
            )


if __name__ == "__main__":
    main()
