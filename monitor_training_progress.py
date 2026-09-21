#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path


EPOCH_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* Epoch (?P<epoch>\d+) \| "
    r"train_loss=(?P<train_loss>[-+.\dnan]+).*?"
    r"train_dice=(?P<train_dice>[-+.\dnan]+).*?"
    r"val_soft_dice=(?P<val_soft_dice>[-+.\dnan]+).*?"
    r"val_dice@0\.5=(?P<val_dice_05>[-+.\dnan]+).*?"
    r"val_dice@0\.1=(?P<val_dice_01>[-+.\dnan]+).*?"
    r"prob_mean=(?P<prob_mean>[-+.\dnan]+)"
)


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def latest_log(log_arg: str | None) -> Path:
    if log_arg:
        return Path(log_arg)
    logs = sorted(Path("logs").glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError("No logs/train_*.log files found")
    return logs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize training progress from VascularSeg logs")
    parser.add_argument("--log", default=None, help="Specific log path. Defaults to latest logs/train_*.log")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/bioprint_v5_mps_retrain_evalprep_lowmem")
    args = parser.parse_args()

    log_path = latest_log(args.log)
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        data = match.groupdict()
        rows.append(
            {
                "time": datetime.strptime(data["ts"], "%Y-%m-%d %H:%M:%S"),
                "epoch": int(data["epoch"]),
                "train_loss": parse_float(data["train_loss"]),
                "train_dice": parse_float(data["train_dice"]),
                "val_soft_dice": parse_float(data["val_soft_dice"]),
                "val_dice@0.5": parse_float(data["val_dice_05"]),
                "val_dice@0.1": parse_float(data["val_dice_01"]),
                "prob_mean": parse_float(data["prob_mean"]),
            }
        )

    if not rows:
        print(f"No completed epochs found in {log_path}")
        return

    best = max(rows, key=lambda r: r["val_dice@0.5"])
    last = rows[-1]
    durations = [
        (rows[i]["time"] - rows[i - 1]["time"]).total_seconds() / 3600.0
        for i in range(1, len(rows))
    ]
    avg_hours = sum(durations[-5:]) / len(durations[-5:]) if durations else None
    eta = ""
    if avg_hours:
        remain = max(0, args.epochs - last["epoch"])
        eta_time = datetime.now() + timedelta(hours=avg_hours * remain)
        eta = f"ETA to epoch {args.epochs}: ~{avg_hours * remain:.1f} h ({eta_time:%Y-%m-%d %H:%M})"

    ckpt_dir = Path(args.checkpoint_dir)
    best_ckpt = ckpt_dir / "checkpoint_best.pt"
    best_size = f"{best_ckpt.stat().st_size / (1024**2):.1f} MB" if best_ckpt.exists() else "missing"

    print(f"Log: {log_path}")
    print(f"Completed epochs: {last['epoch']}/{args.epochs}")
    print(f"Latest: epoch {last['epoch']} | train_dice={last['train_dice']:.4f} | val_dice@0.5={last['val_dice@0.5']:.4f} | val_soft_dice={last['val_soft_dice']:.4f} | prob_mean={last['prob_mean']:.4f}")
    print(f"Best: epoch {best['epoch']} | val_dice@0.5={best['val_dice@0.5']:.4f} | val_soft_dice={best['val_soft_dice']:.4f}")
    print(f"checkpoint_best.pt: {best_size}")
    if eta:
        print(eta)
    print("Recent epochs:")
    for row in rows[-8:]:
        print(
            f"  {row['epoch']:03d}  train_dice={row['train_dice']:.4f}  "
            f"val_dice@0.5={row['val_dice@0.5']:.4f}  soft={row['val_soft_dice']:.4f}  "
            f"prob_mean={row['prob_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
