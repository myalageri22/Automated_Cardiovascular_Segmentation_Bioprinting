#!/usr/bin/env python3
"""Validate a checkpoint with 8-view flip TTA."""
from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_updated copy.py"
DEFAULT_RUN = Path("/tmp/train_runs/bioprint_v14_tversky_fn")
DEFAULT_SOUP = DEFAULT_RUN / "checkpoints" / "soup_ep75-80.pt"
DEFAULT_BEST = DEFAULT_RUN / "checkpoints" / "best_dice05.pt"


def import_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("repo_train_updated", str(TRAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_tuple(text: str, cast=int, expected: int | None = None) -> Tuple[Any, ...]:
    values = tuple(cast(x.strip()) for x in str(text).split(",") if x.strip())
    if expected is not None and len(values) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values, got {text!r}")
    return values


def flip_axis_combinations() -> list[tuple[int, ...]]:
    axes = (2, 3, 4)  # N,C,D,H,W spatial axes
    combos: list[tuple[int, ...]] = [()]
    for r in range(1, len(axes) + 1):
        combos.extend(combinations(axes, r))
    return combos


def tta_inference(model, image, roi_size, sw_batch_size, overlap, device):
    """8-view TTA: original plus all D/H/W flip combinations."""
    model.eval()
    preds = []
    with torch.no_grad():
        for axes in flip_axis_combinations():
            view = torch.flip(image, dims=axes) if axes else image
            logits = sliding_window_inference(
                view,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
                sw_device=device,
                device=device,
            )
            if axes:
                logits = torch.flip(logits, dims=axes)
            preds.append(torch.sigmoid(logits))
    return torch.stack(preds, dim=0).mean(dim=0)


def configure(train_mod: Any, args: argparse.Namespace):
    config = train_mod.Config()
    config.device = args.device
    config.imagecas_root = Path(args.imagecas_root)
    config.split_file = Path(args.splits_json)
    config.log_dir = DEFAULT_RUN / "logs"
    config.checkpoint_dir = DEFAULT_RUN / "checkpoints"
    config.dataset_preset = "imagecas"
    config.roi_size = parse_tuple(args.roi_size, int, 3)
    config.pixdim = (0.6, 0.6, 0.6)
    config.ct_window = (-200.0, 700.0)
    config.batch_size = 1
    config.sw_batch_size = args.sw_batch_size
    config.sw_overlap = args.sw_overlap
    config.cache_rate_val = 0.0
    config.num_workers = 0
    config.persistent_workers = False
    config.unet_channels = (32, 64, 128, 256, 512)
    config.unet_strides = (2, 2, 2, 2)
    config.unet_num_res_units = 3
    config.unet_dropout = 0.1
    config.unet_norm = "batch"
    config.use_attention = args.model_type == "attention_unet"
    config.compile_model = False
    config.__post_init__()
    return config


def load_checkpoint(model, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("model_state_dict", state)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible checkpoint: {incompatible}")


def dice_at_threshold(probs, labels, threshold: float) -> float:
    binary = (probs > threshold).float()
    dice = (2.0 * (binary * labels).sum() + 1e-6) / (binary.sum() + labels.sum() + 1e-6)
    return float(dice.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOUP if DEFAULT_SOUP.exists() else DEFAULT_BEST)
    parser.add_argument("--imagecas_root", type=Path, default=ROOT / "Data" / "all")
    parser.add_argument("--splits_json", type=Path, default=ROOT / "splits.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_type", choices=("attention_unet", "plain_unet"), default="attention_unet")
    parser.add_argument("--roi_size", default="96,192,192")
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--sw_overlap", type=float, default=0.625)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    train_mod = import_train_module()
    config = configure(train_mod, args)
    logger = train_mod.setup_logging(config.log_dir, "tta_eval")

    model = train_mod.build_model(config, logger)
    load_checkpoint(model, args.checkpoint)
    model.to(config.device)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    cases = train_mod.discover_cases(config.imagecas_root, logger)
    _, val_cases, _ = train_mod.split_dataset(
        cases,
        Path(args.splits_json),
        config.val_ratio,
        config.test_ratio,
        config.seed,
        logger,
        limit_val=args.limit_val,
    )
    if not val_cases:
        raise RuntimeError("No validation cases found")

    val_transform = train_mod.get_transforms(config, mode="val")
    val_ds = CacheDataset(val_cases, transform=val_transform, cache_rate=0.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    all_dice = []
    per_case = []
    for idx, batch in enumerate(tqdm(val_loader, desc="TTA Val")):
        images = train_mod.strip_meta_tensor(batch["image"]).to(config.device, non_blocking=True)
        labels = train_mod.strip_meta_tensor(batch["label"]).to(config.device, non_blocking=True)
        probs = tta_inference(model, images, config.roi_size, config.sw_batch_size, config.sw_overlap, config.device)
        dice = dice_at_threshold(probs, labels, args.threshold)
        all_dice.append(dice)
        per_case.append({"id": val_cases[idx].get("id", str(idx)), "dice": dice})

    mean_dice = float(np.mean(all_dice))
    result = {
        "checkpoint": str(args.checkpoint),
        "tta_views": 8,
        "val_cases": len(all_dice),
        "threshold": args.threshold,
        "mean_dice": mean_dice,
        "standard_reference": 0.797,
        "delta_vs_reference": mean_dice - 0.797,
        "per_case": per_case,
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "tta_results.json").write_text(json.dumps(result, indent=2) + "\n")
        csv_lines = ["id,dice"] + [f"{row['id']},{row['dice']:.8f}" for row in per_case]
        (args.output_dir / "tta_per_case.csv").write_text("\n".join(csv_lines) + "\n")

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"TTA views: 8")
    print(f"Val cases: {len(all_dice)}")
    print(f"TTA Val Dice@{args.threshold:g}: {mean_dice:.4f}")
    print(f"Standard reference ~0.797; delta: {mean_dice - 0.797:.4f}")
    if args.output_dir:
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
