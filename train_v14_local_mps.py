#!/usr/bin/env python3
"""Mac-local v14 reproduction wrapper.

This preserves ``train_a40_resume.py`` and calls its training function directly
with MPS-aware device selection. It is intended to reproduce the v14 fine-tune
from an archived v11 checkpoint when CUDA/RunPod is unavailable.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

import train_a40_resume as train_entry


def resolve_local_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = train_entry.build_parser()
    parser.description = "Mac-local v14 reproduction wrapper around train_a40_resume.py"
    for action in parser._actions:
        if action.dest == "experiment_name":
            action.required = False
        elif action.dest == "device":
            action.choices = ["auto", "cuda", "mps", "cpu"]
            action.default = "auto"
        elif action.dest == "output_dir":
            action.default = "outputs/train_runs"
    parser.set_defaults(
        experiment_name="bioprint_v14_tversky_fn_local",
        resume_checkpoint="outputs/train_runs/bioprint_v11_a40_roi96_posneg3to1_lr2e5/checkpoints/best_dice05.pt",
        strict_load=True,
        allow_partial_load=True,
        model_type="attention_unet",
        roi_size="96,192,192",
        batch_size=1,
        accumulation_steps=2,
        epochs=100,
        lr=2e-5,
        weight_decay=1e-5,
        grad_clip_norm=1.0,
        num_workers=0,
        cache_rate_train=0.0,
        cache_rate_val=0.0,
        scheduler="reduce_on_plateau",
        early_stopping_patience=15,
        pos_neg_ratio=(3, 1),
        num_samples=4,
        min_pos_voxels=1,
        pos_weight_fixed=0.0,
        loss_mode="tversky_dice",
        val_output_device="auto",
        pin_memory=False,
    )
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = build_parser().parse_args()
    args.device = resolve_local_device(args.device)
    if args.device == "cpu":
        print(
            "WARNING: resolved device is CPU. This will be extremely slow for v14 3D training. "
            "Install/use an MPS-enabled PyTorch environment if possible."
        )
    if args.device == "mps":
        args.pin_memory = False
        args.num_workers = max(0, args.num_workers)
        args.val_output_device = "auto"
    resume_path = Path(args.resume_checkpoint)
    if not resume_path.exists() and args.resume_checkpoint not in {"auto", "best", "last", "none", "None", ""}:
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    imagecas_root = Path(args.imagecas_root)
    if not imagecas_root.exists():
        raise FileNotFoundError(f"ImageCAS root not found: {imagecas_root}")
    return train_entry.train(args)


if __name__ == "__main__":
    raise SystemExit(main())
