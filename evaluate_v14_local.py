#!/usr/bin/env python3
"""Mac-local wrapper for held-out ImageCAS test evaluation.

This preserves ``evaluate_full_test_a40.py`` and only widens device selection
so a reproduced checkpoint can be evaluated on MPS or CPU when CUDA is absent.
"""
from __future__ import annotations

import os

import torch

import evaluate_full_test_a40 as eval_entry


def resolve_local_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    parser = eval_entry.build_parser()
    parser.description = "Mac-local wrapper for evaluate_full_test_a40.py"
    for action in parser._actions:
        if action.dest == "device":
            action.choices = ["auto", "cuda", "mps", "cpu"]
        elif action.dest == "output_dir":
            action.default = "outputs/final_test_250_local_reproduced"
        elif action.dest == "num_workers":
            action.default = 0
        elif action.dest == "val_output_device":
            action.default = "cpu"
    args = parser.parse_args()
    args.device = resolve_local_device(args.device)
    if args.device == "mps":
        args.val_output_device = "cpu"
        args.num_workers = max(0, args.num_workers)
    return eval_entry.evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
