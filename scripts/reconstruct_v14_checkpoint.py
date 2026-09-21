#!/usr/bin/env python3
"""Reconstruct or audit the recorded v14 checkpoint soup.

This script is intentionally conservative. It will create the true recorded
v14 soup checkpoint only when the v14 ingredient checkpoints are present. It
will not silently substitute v11 checkpoints and call them v14.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_RUN = "bioprint_v14_tversky_fn"
DEFAULT_START = 75
DEFAULT_END = 80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_from_name(path: Path) -> int | None:
    match = re.search(r"(?:checkpoint_epoch_|epoch_)(\d+)", path.name)
    return int(match.group(1)) if match else None


def checkpoint_epoch(path: Path) -> Any:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"load_error": str(exc)}
    if isinstance(ckpt, dict):
        return {
            "epoch": ckpt.get("epoch"),
            "best_metric": ckpt.get("best_metric"),
            "experiment_name": (ckpt.get("config") or {}).get("experiment_name")
            if isinstance(ckpt.get("config"), dict)
            else None,
            "config": summarize_config(ckpt.get("config")),
        }
    return {"type": type(ckpt).__name__}


def summarize_config(config: Any) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    keys = [
        "experiment_name",
        "model_type",
        "roi_size",
        "lr",
        "weight_decay",
        "loss_mode",
        "pos_weight_fixed",
        "pos_neg_ratio",
        "num_samples",
        "output_dir",
    ]
    return {k: config.get(k) for k in keys if k in config}


def discover_epoch_checkpoints(source_dirs: Iterable[Path], epoch_start: int, epoch_end: int) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.pt")):
            epoch = epoch_from_name(path)
            if epoch is None or epoch < epoch_start or epoch > epoch_end:
                continue
            found.setdefault(epoch, path)
    return found


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError(f"{path} does not contain a model_state_dict")
    return ckpt["model_state_dict"]


def average_states(paths: list[Path]) -> dict[str, torch.Tensor]:
    states = [load_model_state(path) for path in paths]
    keys = list(states[0].keys())
    expected = set(keys)
    for path, state in zip(paths[1:], states[1:]):
        if set(state.keys()) != expected:
            diff = sorted(expected.symmetric_difference(state.keys()))
            raise ValueError(f"State keys differ for {path}: {diff[:10]}")

    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in states]
        first = tensors[0]
        if torch.is_tensor(first) and torch.is_floating_point(first):
            averaged[key] = torch.stack([t.detach().float() for t in tensors]).mean(dim=0).to(first.dtype)
        else:
            averaged[key] = tensors[-1]
    return averaged


def write_report(path: Path, status: dict[str, Any]) -> None:
    lines = [
        "# v14 Checkpoint Reconstruction Status",
        "",
        f"Generated: {status['generated_at']}",
        "",
        f"Expected run: `{status['expected_run']}`",
        f"Expected soup: `{status['expected_soup_name']}`",
        f"Epoch range: {status['epoch_start']}-{status['epoch_end']}",
        "",
        "## Source Directories Checked",
        "",
    ]
    lines.extend(f"- `{p}`" for p in status["source_dirs"])
    lines.extend(["", "## Result", ""])
    if status["reconstructed"]:
        lines.extend(
            [
                "The recorded v14 soup checkpoint was reconstructed from all required ingredient checkpoints.",
                "",
                f"- Output checkpoint: `{status['output_checkpoint']}`",
                f"- SHA256: `{status['output_sha256']}`",
            ]
        )
    else:
        lines.extend(
            [
                "The recorded v14 soup checkpoint was **not** reconstructed.",
                "",
                "Reason:",
                f"- {status['reason']}",
                "",
                "This script intentionally refuses to substitute v11 checkpoints and label them as v14.",
                "A proxy soup can be made for experimentation, but it cannot support the published v14 Dice claim.",
            ]
        )
    lines.extend(["", "## Ingredient Checkpoints", ""])
    for item in status["ingredients"]:
        lines.append(f"- epoch {item['epoch']}: `{item.get('path', 'MISSING')}`")
        if item.get("metadata"):
            lines.append(f"  - metadata: `{json.dumps(item['metadata'], default=str)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-run", default=DEFAULT_RUN)
    parser.add_argument("--epoch-start", type=int, default=DEFAULT_START)
    parser.add_argument("--epoch-end", type=int, default=DEFAULT_END)
    parser.add_argument("--source-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v14_reconstruction"))
    parser.add_argument("--allow-missing", action="store_true", help="Write an audit report even if ingredients are missing.")
    args = parser.parse_args()

    default_source_dirs = [
        Path("/tmp/train_runs") / args.expected_run / "checkpoints",
        Path("outputs/train_runs") / args.expected_run / "checkpoints",
        Path("outputs/v14_recovery") / args.expected_run / "checkpoints",
    ]
    source_dirs = args.source_dir or default_source_dirs
    found = discover_epoch_checkpoints(source_dirs, args.epoch_start, args.epoch_end)
    required_epochs = list(range(args.epoch_start, args.epoch_end + 1))
    missing = [epoch for epoch in required_epochs if epoch not in found]

    out_dir = args.output_root / args.expected_run / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / "v14_reconstruction_status.json"
    report_path = args.output_root / "V14_RECONSTRUCTION_STATUS.md"
    output_path = out_dir / f"soup_ep{args.epoch_start}-{args.epoch_end}.pt"

    ingredients = []
    for epoch in required_epochs:
        path = found.get(epoch)
        item: dict[str, Any] = {"epoch": epoch}
        if path is not None:
            item["path"] = str(path)
            item["sha256"] = sha256_file(path)
            item["metadata"] = checkpoint_epoch(path)
        ingredients.append(item)

    status: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_run": args.expected_run,
        "expected_soup_name": output_path.name,
        "epoch_start": args.epoch_start,
        "epoch_end": args.epoch_end,
        "source_dirs": [str(p) for p in source_dirs],
        "ingredients": ingredients,
        "missing_epochs": missing,
        "reconstructed": False,
        "output_checkpoint": None,
        "output_sha256": None,
        "reason": "",
    }

    if missing:
        status["reason"] = f"Missing ingredient epoch checkpoints: {missing}"
        metadata_path.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")
        write_report(report_path, status)
        print(f"Missing ingredient checkpoints: {missing}")
        print(f"Wrote audit report: {report_path}")
        return 0 if args.allow_missing else 2

    paths = [found[epoch] for epoch in required_epochs]
    averaged = average_states(paths)
    torch.save(
        {
            "model_state_dict": averaged,
            "epoch": -1,
            "best_metric": None,
            "soup_metadata": {
                "expected_run": args.expected_run,
                "epoch_start": args.epoch_start,
                "epoch_end": args.epoch_end,
                "source_checkpoints": [str(p) for p in paths],
                "source_sha256": {str(p): sha256_file(p) for p in paths},
                "construction": "uniform_average_floating_model_state_dict_values",
            },
        },
        output_path,
    )
    status["reconstructed"] = True
    status["output_checkpoint"] = str(output_path)
    status["output_sha256"] = sha256_file(output_path)
    status["reason"] = "all required ingredient checkpoints were present"
    metadata_path.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")
    write_report(report_path, status)
    print(f"Reconstructed checkpoint: {output_path}")
    print(f"SHA256: {status['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
