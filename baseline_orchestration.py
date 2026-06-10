#!/usr/bin/env python3
"""Prepare and summarize baseline runs with schemas matching the main A40 run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def command_table() -> List[Dict[str, str]]:
    cuda_env = "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128"
    return [
        {
            "baseline": "plain_unet",
            "stage": "train",
            "command": (
                f"{cuda_env} python train_a40_resume.py --experiment_name bioprint_v6_plain_unet_baseline "
                "--resume_checkpoint none --model_type plain_unet --roi_size 96,192,192 --batch_size 1 "
                "--accumulation_steps 2 --epochs 200 --device cuda --splits_json splits.json "
                "--output_dir outputs/baselines/plain_unet"
            ),
        },
        {
            "baseline": "plain_unet",
            "stage": "eval",
            "command": (
                f"{cuda_env} python evaluate_full_test_a40.py "
                "--checkpoint outputs/baselines/plain_unet/bioprint_v6_plain_unet_baseline/checkpoints/best_dice05.pt "
                "--splits_json splits.json --output_dir outputs/baselines/plain_unet/full_test_eval "
                "--device cuda --roi_size 96,192,192 --sw_batch_size 1 --resume --skip_existing --model_type plain_unet"
            ),
        },
    ]


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_metric(summary_json: Path, metric: str) -> str:
    if not summary_json.exists():
        return "missing"
    try:
        payload = json.loads(summary_json.read_text())
        value = payload.get("metrics", {}).get(metric, {}).get("mean")
        return str(value) if value is not None else "missing"
    except Exception as exc:
        return f"error: {exc}"


def generate_comparison(output_root: str | Path = "outputs/baselines") -> None:
    output_root = Path(output_root)
    commands = command_table()
    write_csv(output_root / "baseline_commands.csv", commands)
    rows = [
        {
            "run": "main_attention_unet",
            "summary_json": "outputs/full_test_eval_a40/summary_metrics.json",
            "mean_dice@0.5": collect_metric(Path("outputs/full_test_eval_a40/summary_metrics.json"), "dice@0.5"),
            "status": "available_if_full_eval_completed",
        },
        {
            "run": "plain_unet",
            "summary_json": "outputs/baselines/plain_unet/full_test_eval/summary_metrics.json",
            "mean_dice@0.5": collect_metric(output_root / "plain_unet/full_test_eval/summary_metrics.json", "dice@0.5"),
            "status": "primary_baseline",
        },
        {
            "run": "nnunet_style",
            "summary_json": "outputs/baselines/nnunet_style/full_test_eval/summary_metrics.json",
            "mean_dice@0.5": "optional_unavailable",
            "status": "optional; no nnU-Net environment was detected by this orchestrator",
        },
    ]
    write_csv(output_root / "baseline_comparison.csv", rows)
    md = ["# Baseline Comparison", "", "The primary baseline is a plain 3D UNet with the same channels/preprocessing/split.", ""]
    for cmd in commands:
        md.append(f"## {cmd['baseline']} {cmd['stage']}")
        md.append("")
        md.append("```bash")
        md.append(cmd["command"])
        md.append("```")
        md.append("")
    md.append("nnU-Net-style baseline: optional/unavailable unless an nnU-Net environment is installed and configured.")
    (output_root / "baseline_comparison.md").write_text("\n".join(md) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write baseline commands and comparison tables")
    parser.add_argument("--output_root", default="outputs/baselines")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    generate_comparison(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
