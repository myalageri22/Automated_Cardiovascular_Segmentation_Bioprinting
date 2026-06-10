#!/usr/bin/env python3
"""Average recent checkpoints for a free validation Dice bump.

Defaults are set for the current bioprint_v14_tversky_fn run. The script accepts
both checkpoint_epoch_075.pt-style and epoch_075.pt-style checkpoint names.
Floating tensors are averaged; non-floating buffers are copied from the newest
checkpoint so BatchNorm counters keep valid integer dtypes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import torch


DEFAULT_CKPT_DIR = Path("/tmp/train_runs/bioprint_v14_tversky_fn/checkpoints")
DEFAULT_OUTPUT = DEFAULT_CKPT_DIR / "soup_ep75-80.pt"


def epoch_from_name(path: Path) -> int | None:
    match = re.search(r"(?:checkpoint_epoch_|epoch_)(\d+)", path.name)
    return int(match.group(1)) if match else None


def discover_epoch_checkpoints(checkpoint_dir: Path, last_n: int, epoch_start: int | None, epoch_end: int | None) -> List[Path]:
    candidates: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.glob("*.pt"):
        epoch = epoch_from_name(path)
        if epoch is None:
            continue
        if epoch_start is not None and epoch < epoch_start:
            continue
        if epoch_end is not None and epoch > epoch_end:
            continue
        candidates.append((epoch, path))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")
    if epoch_start is None and epoch_end is None:
        candidates = candidates[-last_n:]
    return [path for _, path in candidates]


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise KeyError(f"{path} does not contain model_state_dict")
    return state["model_state_dict"]


def average_checkpoints(checkpoint_paths: Iterable[Path], output_path: Path) -> None:
    checkpoint_paths = [Path(p) for p in checkpoint_paths]
    if not checkpoint_paths:
        raise ValueError("No checkpoints were provided")

    print("Averaging checkpoints:")
    for path in checkpoint_paths:
        print(f"  - {path}")

    state_dicts = [load_model_state(path) for path in checkpoint_paths]
    keys = list(state_dicts[0].keys())
    for sd, path in zip(state_dicts[1:], checkpoint_paths[1:]):
        if set(sd.keys()) != set(keys):
            missing = set(keys) ^ set(sd.keys())
            raise ValueError(f"State dict keys differ for {path}: {sorted(missing)[:10]}")

    avg_state = {}
    for key in keys:
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if torch.is_tensor(first) and torch.is_floating_point(first):
            avg_state[key] = torch.stack([v.detach().float() for v in values], dim=0).mean(dim=0).to(dtype=first.dtype)
        else:
            avg_state[key] = values[-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": avg_state,
            "epoch": -1,
            "soup_metadata": {
                "num_checkpoints": len(checkpoint_paths),
                "source_checkpoints": [str(p) for p in checkpoint_paths],
            },
        },
        output_path,
    )
    print(f"Soup checkpoint saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--last_n", type=int, default=6)
    parser.add_argument("--epoch_start", type=int, default=75)
    parser.add_argument("--epoch_end", type=int, default=80)
    args = parser.parse_args()

    paths = discover_epoch_checkpoints(args.checkpoint_dir, args.last_n, args.epoch_start, args.epoch_end)
    expected = set(range(args.epoch_start, args.epoch_end + 1)) if args.epoch_start is not None and args.epoch_end is not None else set()
    found = {epoch_from_name(p) for p in paths}
    missing = sorted(e for e in expected if e not in found)
    if missing:
        print(f"Warning: missing epoch checkpoints {missing}; averaging the {len(paths)} found checkpoint(s).")
    average_checkpoints(paths, args.output)


if __name__ == "__main__":
    main()
