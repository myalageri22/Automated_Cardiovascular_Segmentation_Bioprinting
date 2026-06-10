#!/usr/bin/env python3
"""A40-ready training/resume entry point for ImageCAS vessel segmentation.

This script reuses the repository's canonical MONAI transforms/model/loss code
from ``train_updated copy.py`` while adding CUDA-first resume selection,
reviewer-ready run artifacts, atomic checkpoints, CSV logs, and CPU-only smoke
tests. Heavy training is only started when the user runs this script directly.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

from scripts.checkpoint_utils import (
    load_checkpoint_for_model,
    list_checkpoints,
    safe_torch_load,
    select_best_checkpoint,
)


DEFAULT_CHANNELS = (32, 64, 128, 256, 512)
DEFAULT_STRIDES = (2, 2, 2, 2)
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_updated copy.py"


def parse_tuple(text: str, cast=int, expected: Optional[int] = None) -> Tuple[Any, ...]:
    values = tuple(cast(x.strip()) for x in str(text).split(",") if x.strip())
    if expected is not None and len(values) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values, got {text!r}")
    return values


def parse_pos_neg_ratio(text: str) -> Tuple[int, int]:
    values = parse_tuple(text, int, 2)
    if values[0] < 0 or values[1] < 0:
        raise argparse.ArgumentTypeError("--pos_neg_ratio values must be non-negative")
    if values[0] == 0 and values[1] == 0:
        raise argparse.ArgumentTypeError("--pos_neg_ratio cannot be 0,0")
    return values


def import_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("repo_train_updated", str(TRAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(device_arg: str, require_cuda: bool = False) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("A40 training command requested CUDA, but CUDA is unavailable.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
    return device


def seed_everything(seed: int, deterministic: bool = True) -> List[str]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    notes = []
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            notes.append("torch.use_deterministic_algorithms(True, warn_only=True)")
        except TypeError:
            torch.use_deterministic_algorithms(True)
            notes.append("torch.use_deterministic_algorithms(True)")
        except Exception as exc:
            notes.append(f"deterministic_algorithms_unavailable: {exc}")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return notes


def make_grad_scaler(device: torch.device, enabled: bool) -> Optional[torch.amp.GradScaler]:
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:  # older torch
        return torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    try:
        return torch.autocast("cuda", enabled=True, dtype=dtype)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True, dtype=dtype)  # type: ignore[attr-defined]


def atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        # Some network-mounted RunPod workspaces can lose dotfile temp paths
        # during replace. Fall back to a direct save instead of crashing.
        torch.save(obj, path)


def write_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            old = next(reader, None)
        if old:
            fieldnames = list(dict.fromkeys(old + fieldnames))
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unspecified"


def pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return f"pip_freeze_unavailable: {exc}\n"


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}

    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow and torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow


def build_run_config(args: argparse.Namespace, device: torch.device, run_dir: Path, notes: List[str]) -> Dict[str, Any]:
    cfg = vars(args).copy()
    cfg.update(
        {
            "device_resolved": str(device),
            "run_dir": str(run_dir),
            "model_channels": list(DEFAULT_CHANNELS),
            "model_strides": list(DEFAULT_STRIDES),
            "preprocessing": {
                "orientation": "RAS",
                "pixdim": [0.6, 0.6, 0.6],
                "ct_window": [-200, 700],
                "scale": [0, 1],
                "label_binarization": "label > 0",
            },
            "class_imbalance": {
                "positive_voxels": 78936341,
                "negative_positive_ratio": 597.98,
                "pos_weight_cap": args.pos_weight_cap,
                "pos_weight_fixed": args.pos_weight_fixed,
            },
            "loss": {
                "mode": args.loss_mode,
                "tversky_alpha_fn": args.tversky_alpha,
                "tversky_beta_fp": args.tversky_beta,
            },
            "scheduler_config": {
                "name": args.scheduler,
                "reduce_on_plateau_factor": args.reduce_on_plateau_factor,
                "reduce_on_plateau_patience": args.reduce_on_plateau_patience,
            },
            "sampling": {
                "pos_neg_ratio": list(args.pos_neg_ratio),
                "num_samples": args.num_samples,
                "rationale": "Mixed foreground/background patches improve full-volume false-positive control.",
            },
            "determinism_notes": notes,
            "git_commit": git_commit(),
        }
    )
    return cfg


def configure_repo_config(train_mod: Any, args: argparse.Namespace, device: torch.device, ckpt_dir: Path) -> Any:
    cfg = train_mod.Config()
    cfg.device = device.type
    cfg.imagecas_root = Path(args.imagecas_root)
    cfg.split_file = Path(args.splits_json)
    cfg.checkpoint_dir = ckpt_dir
    cfg.log_dir = Path(args.output_dir) / args.experiment_name / "logs"
    cfg.dataset_preset = "imagecas"
    cfg.roi_size = parse_tuple(args.roi_size, int, 3)
    cfg.pixdim = (0.6, 0.6, 0.6)
    cfg.ct_window = (-200.0, 700.0)
    cfg.batch_size = args.batch_size
    cfg.accumulation_steps = args.accumulation_steps
    cfg.epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip_norm = args.grad_clip_norm
    cfg.num_workers = args.num_workers
    cfg.cache_rate_train = args.cache_rate_train
    cfg.cache_rate_val = args.cache_rate_val
    cfg.sw_batch_size = 1
    cfg.val_output_device = args.val_output_device
    cfg.unet_channels = DEFAULT_CHANNELS
    cfg.unet_strides = DEFAULT_STRIDES
    cfg.unet_num_res_units = args.unet_res_units
    cfg.unet_dropout = args.dropout
    cfg.unet_norm = "batch"
    cfg.use_attention = args.model_type == "attention_unet"
    cfg.grad_checkpoint = args.grad_checkpoint
    cfg.pos_neg_ratio = args.pos_neg_ratio
    cfg.num_samples = args.num_samples
    cfg.force_pos_patches = args.pos_neg_ratio[0] > 0
    cfg.min_pos_voxels = args.min_pos_voxels
    cfg.loss_mode = args.loss_mode
    cfg.tversky_alpha = args.tversky_alpha
    cfg.tversky_beta = args.tversky_beta
    cfg.pos_weight_cap = args.pos_weight_cap
    cfg.pos_weight_fixed = args.pos_weight_fixed
    cfg.seed = args.seed
    cfg.deterministic = True
    cfg.amp = device.type == "cuda"
    cfg.amp_dtype = "auto"
    cfg.persistent_workers = args.num_workers > 0
    cfg.prefetch_factor = 2
    return cfg


def resolve_resume_checkpoint(args: argparse.Namespace) -> Tuple[Optional[Path], Dict[str, Any]]:
    mode = args.resume_checkpoint
    if mode in ("", "none", "None"):
        return None, {"mode": mode, "reason": "resume disabled"}
    if mode not in {"auto", "best", "last"}:
        path = Path(mode)
        return path if path.exists() else None, {"mode": "path", "requested": mode, "exists": path.exists()}
    candidates = list_checkpoints(".")
    if mode == "last":
        epoch_ckpts = [Path(p) for p in candidates if "checkpoint_epoch_" in p or "epoch_" in Path(p).name]
        if epoch_ckpts:
            return max(epoch_ckpts, key=lambda p: p.stat().st_mtime), {"mode": mode, "reason": "newest epoch checkpoint"}
    selected, rationale = select_best_checkpoint(candidates)
    return Path(selected) if selected else None, rationale


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.reduce_on_plateau_factor,
            patience=args.reduce_on_plateau_patience,
        )
    if args.scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    best_metrics: Dict[str, float],
    config: Dict[str, Any],
    ema: Optional[EMA],
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
        "best_metrics": best_metrics,
        "best_metric": best_metrics.get("dice@0.5", best_metrics.get("soft_dice")),
        "config": config,
        "git_commit": config.get("git_commit", "unspecified"),
    }
    atomic_torch_save(state, path)


def synthetic_smoke_test(model_type: str = "attention_unet") -> Dict[str, Any]:
    train_mod = import_train_module()
    device = torch.device("cpu")
    cfg = train_mod.Config()
    cfg.device = "cpu"
    cfg.roi_size = (16, 32, 32)
    cfg.unet_channels = (4, 8, 16, 32, 64)
    cfg.unet_strides = DEFAULT_STRIDES
    cfg.unet_dropout = 0.0
    cfg.use_attention = model_type == "attention_unet"
    smoke_log_dir = Path("outputs/smoke_logs")
    smoke_log_dir.mkdir(parents=True, exist_ok=True)
    logger = train_mod.setup_logging(smoke_log_dir, "synthetic_a40")
    model = train_mod.build_model(cfg, logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pos_weight = torch.tensor([2.0])
    loss_fn = train_mod.build_loss_fn(cfg, pos_weight)
    x = torch.randn(1, 1, 16, 32, 32)
    y = (torch.rand(1, 1, 16, 32, 32) > 0.97).float()
    logits = model(x)
    loss, stats = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    return {"ok": True, "output_shape": list(logits.shape), "loss": float(loss.item()), "stats": list(stats.keys())}


def train(args: argparse.Namespace) -> int:
    require_cuda = args.device == "cuda"
    device = resolve_device(args.device, require_cuda=require_cuda)
    notes = seed_everything(args.seed, deterministic=True)
    run_dir = Path(args.output_dir) / args.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(args, device, run_dir, notes)
    (run_dir / "used_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    (run_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    (run_dir / "git_commit.txt").write_text(run_config["git_commit"] + "\n")
    (run_dir / "pip_freeze.txt").write_text(pip_freeze())
    (run_dir / "training_notes.md").write_text(
        "# Training Notes\n\n"
        "- A40 campaign preserves Attention U-Net architecture `(32,64,128,256,512)` unless `--model_type plain_unet` is used for baseline.\n"
        "- ROI may change between Mac MPS and CUDA because it does not change checkpoint weight shapes.\n"
        "- Validation checkpoint selection uses Dice metrics; `val_loss` is not the primary selection criterion.\n"
        "- Determinism is requested where feasible; CUDA kernels may still have unavoidable nondeterminism.\n"
    )

    train_mod = import_train_module()
    cfg = configure_repo_config(train_mod, args, device, ckpt_dir)
    logger = train_mod.setup_logging(run_dir / "logs", args.experiment_name)
    logger.info(f"A40 entrypoint run_dir={run_dir}")
    logger.info(f"Resolved device={device}")

    all_cases = train_mod.discover_cases(cfg.imagecas_root, logger)
    train_cases, val_cases, _ = train_mod.split_dataset(
        all_cases, cfg.split_file, 0.2, 0.0, cfg.seed, logger, None, None, None
    )
    (run_dir / "resolved_split_counts.json").write_text(
        json.dumps({"train": len(train_cases), "val": len(val_cases)}, indent=2)
    )

    train_ds = train_mod.CacheDataset(
        data=train_cases,
        transform=train_mod.get_transforms(cfg, mode="train"),
        cache_rate=cfg.cache_rate_train,
        num_workers=cfg.num_workers,
    )
    val_ds = train_mod.CacheDataset(
        data=val_cases,
        transform=train_mod.get_transforms(cfg, mode="val"),
        cache_rate=cfg.cache_rate_val,
        num_workers=max(0, cfg.num_workers // 2),
    )
    train_loader = train_mod.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=train_mod.pad_list_data_collate,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = train_mod.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        collate_fn=train_mod.pad_list_data_collate,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    model = train_mod.build_model(cfg, logger)
    if args.grad_checkpoint:
        logger.info("Enabling whole-model gradient checkpointing")
        model = train_mod.GradientCheckpointWrapper(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = make_grad_scaler(device, enabled=device.type == "cuda")
    ema = EMA(model, decay=0.999) if args.ema else None
    pos_weight = train_mod.compute_class_weights(train_cases, cfg, logger)
    loss_fn = train_mod.build_loss_fn(cfg, pos_weight)
    metrics = {}

    start_epoch = 0
    resume_path, resume_rationale = resolve_resume_checkpoint(args)
    run_config["resume_rationale"] = resume_rationale
    if resume_path:
        try:
            logger.info(f"Loading resume checkpoint weights from {resume_path}")
            load_result = load_checkpoint_for_model(
                train_mod.unwrap_model_for_checkpoint(model), resume_path, strict=args.strict_load, device="cpu"
            )
            raw = load_result.pop("raw_checkpoint")
            start_epoch = int(raw.get("epoch", 0)) if isinstance(raw, dict) else 0
            if isinstance(raw, dict) and raw.get("optimizer_state_dict") and not args.allow_partial_load:
                try:
                    optimizer.load_state_dict(raw["optimizer_state_dict"])
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr
                        group["weight_decay"] = args.weight_decay
                except Exception as exc:
                    logger.warning(f"Optimizer state not restored: {exc}")
            logger.info(f"Resume diagnostics: {json.dumps(load_result, default=str)}")
        except Exception as exc:
            logger.error(f"Checkpoint resume failed: {exc}")
            if not args.clean_restart_if_invalid_checkpoint:
                raise
            logger.warning("Proceeding with clean random initialization because --clean_restart_if_invalid_checkpoint was set")
            start_epoch = 0
    elif args.init_from_pretrained_encoder:
        logger.warning("init_from_pretrained_encoder requested, but no compatible pretrained encoder is specified in repo; using random init")

    best = {"soft_dice": -math.inf, "dice@0.5": -math.inf}
    no_improve = 0
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    global_step = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        count = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            images = train_mod.strip_meta_tensor(batch["image"]).to(device, non_blocking=True)
            labels = train_mod.strip_meta_tensor(batch["label"]).to(device, non_blocking=True)
            with autocast_context(device, enabled=device.type == "cuda", dtype=amp_dtype):
                logits = model(images)
                loss, stats = loss_fn(logits, labels)
                loss = loss / args.accumulation_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
                global_step += 1
                if global_step % args.log_every_n_steps == 0:
                    write_csv_row(
                        run_dir / "train_step_log.csv",
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": float(loss.item() * args.accumulation_steps),
                            "grad_norm": float(grad_norm),
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                    )
            totals["loss"] += float(loss.item() * args.accumulation_steps)
            for key, value in stats.items():
                totals[key] += float(value.detach().cpu().item() if torch.is_tensor(value) else value)
            count += 1
        train_row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, count),
            "train_dice": 1.0 - totals.get("dice_loss", 0.0) / max(1, count),
            "train_bce": totals.get("bce_loss", float("nan")) / max(1, count),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        write_csv_row(run_dir / "train_epoch_log.csv", train_row)

        val_loss, val_metrics = train_mod.Trainer(cfg, logger).validate(model, val_loader, loss_fn, metrics, epoch)
        val_row = {"epoch": epoch, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        write_csv_row(run_dir / "val_epoch_log.csv", val_row)
        current_soft = float(val_metrics.get("soft_dice", -math.inf))
        current_dice05 = float(val_metrics.get("dice@0.5", -math.inf))
        improved = False
        if current_soft > best["soft_dice"]:
            best["soft_dice"] = current_soft
            save_checkpoint(ckpt_dir / "best_soft_dice.pt", model, optimizer, scheduler, scaler, epoch, best, run_config, ema)
            improved = True
        if current_dice05 > best["dice@0.5"]:
            best["dice@0.5"] = current_dice05
            save_checkpoint(ckpt_dir / "best_dice05.pt", model, optimizer, scheduler, scaler, epoch, best, run_config, ema)
            improved = True
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best, run_config, ema)
        if args.save_every_n_epochs and epoch % args.save_every_n_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, best, run_config, ema)
        if args.scheduler == "reduce_on_plateau":
            scheduler.step(current_dice05)
        else:
            scheduler.step()
        logger.info(
            f"Epoch {epoch:03d} | train_dice={train_row['train_dice']:.4f} | "
            f"val_soft_dice={current_soft:.4f} | val_dice@0.5={current_dice05:.4f} | best={best}"
        )
        no_improve = 0 if improved else no_improve + 1
        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            logger.info(f"Early stopping after {no_improve} epochs without improvement")
            break

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A40 CUDA training/resume entry point")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--resume_checkpoint", default="auto", help="auto|best|last|path|none")
    parser.add_argument("--allow_partial_load", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--clean_restart_if_invalid_checkpoint", action="store_true")
    parser.add_argument("--init_from_pretrained_encoder", action="store_true")
    parser.add_argument("--model_type", default="attention_unet", choices=["attention_unet", "plain_unet"])
    parser.add_argument("--roi_size", default="96,192,192")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--log_every_n_steps", type=int, default=25)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--splits_json", default="splits.json")
    parser.add_argument("--imagecas_root", default="Data/all")
    parser.add_argument("--cache_rate_train", type=float, default=0.0)
    parser.add_argument("--cache_rate_val", type=float, default=0.0)
    parser.add_argument("--val_output_device", default="same", choices=["auto", "cpu", "same"])
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "reduce_on_plateau", "exponential"])
    parser.add_argument("--reduce_on_plateau_patience", type=int, default=6)
    parser.add_argument("--reduce_on_plateau_factor", type=float, default=0.5)
    parser.add_argument("--output_dir", default="outputs/train_runs")
    parser.add_argument("--pos_neg_ratio", type=parse_pos_neg_ratio, default=(1, 0))
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--min_pos_voxels", type=int, default=400)
    parser.add_argument("--pos_weight_cap", type=float, default=200.0)
    parser.add_argument("--pos_weight_fixed", type=float, default=None)
    parser.add_argument(
        "--loss_mode",
        default="tversky_dice",
        choices=["tversky_dice", "all", "dice", "bce", "dice_focal"],
    )
    parser.add_argument(
        "--tversky_alpha",
        type=float,
        default=0.7,
        help="Tversky false-negative weight in this repo's implementation; higher values favor recall.",
    )
    parser.add_argument(
        "--tversky_beta",
        type=float,
        default=0.3,
        help="Tversky false-positive weight in this repo's implementation; higher values favor precision.",
    )
    parser.add_argument("--unet_res_units", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--synthetic_smoke_test", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.synthetic_smoke_test:
        print(json.dumps(synthetic_smoke_test(args.model_type), indent=2))
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
