#!/usr/bin/env python3
"""Evaluate one or more saved segmentation checkpoints on a fixed split.

This script is intentionally separate from training so you can:
- run repeatable checkpoint sweeps over the same cases
- compare checkpoints with the exact same inference path
- save per-case and per-checkpoint metrics to CSV/JSON
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    _tmp_base = Path(tempfile.gettempdir())
except Exception:
    _tmp_base = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str((_tmp_base / "mplconfig").resolve()))
# Allow unsupported MPS ops to fall back to CPU instead of hard-failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from tqdm import tqdm


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("checkpoint_eval")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def suppress_noisy_warnings() -> None:
    # MONAI orientation deprecation warning is noisy and non-fatal for eval.
    warnings.filterwarnings(
        "ignore",
        message=r".*Orientationd\.__init__:labels.*",
        category=FutureWarning,
    )
    # PyTorch warning emitted repeatedly from MONAI SW inference internals.
    warnings.filterwarnings(
        "ignore",
        message=r".*Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
    )


def _mps_available() -> bool:
    if not hasattr(torch.backends, "mps"):
        return False
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def resolve_device(device_arg: str, strict_device: bool, logger: logging.Logger) -> str:
    cuda_ok = torch.cuda.is_available()
    mps_ok = _mps_available()

    if device_arg == "auto":
        # On Apple Silicon, prefer MPS when available.
        if platform.system() == "Darwin" and mps_ok:
            return "mps"
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if device_arg == "cuda" and not cuda_ok:
        msg = "Requested --device cuda, but CUDA is unavailable."
        if strict_device:
            raise RuntimeError(msg)
        if mps_ok:
            logger.warning(f"{msg} Falling back to mps.")
            return "mps"
        logger.warning(f"{msg} Falling back to cpu.")
        return "cpu"

    if device_arg == "mps" and not mps_ok:
        msg = "Requested --device mps, but MPS is unavailable."
        if strict_device:
            raise RuntimeError(msg)
        logger.warning(f"{msg} Falling back to cpu.")
        return "cpu"

    return device_arg


def load_training_module(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    spec = importlib.util.spec_from_file_location("train_module", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training module from: {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_safe_globals(train_module):
    torch.serialization.add_safe_globals(
        [
            np._core.multiarray.scalar,
            np.dtype,
            train_module.Config,
        ]
    )


def autodetect_data_root(explicit_data_root: Optional[str], logger: logging.Logger) -> Optional[Path]:
    if explicit_data_root:
        root = Path(explicit_data_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--data-root does not exist: {root}")
        logger.info(f"Using data root from --data-root: {root}")
        return root

    candidates = [
        (Path.cwd() / "data" / "processed" / "all").resolve(),
        (Path(__file__).resolve().parent / "data" / "processed" / "all").resolve(),
        Path("/workspace/cloud_bundle/data/processed/all"),
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Auto-detected data root: {candidate}")
            return candidate

    logger.warning("Could not auto-detect data root for missing-path remap.")
    return None


def build_data_index(data_root: Optional[Path], logger: logging.Logger) -> Dict[str, str]:
    if data_root is None:
        return {}
    index: Dict[str, str] = {}
    for p in data_root.rglob("*.nii.gz"):
        index.setdefault(p.name, str(p.resolve()))
    logger.info(f"Indexed {len(index)} NIfTI filenames under {data_root}")
    return index


def remap_case_paths_by_id(case_id: str, data_index: Dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    if not data_index:
        return None, None

    normalized_ids = [case_id]
    if case_id.isdigit():
        normalized = str(int(case_id))
        if normalized not in normalized_ids:
            normalized_ids.append(normalized)

    image_candidates: List[str] = []
    label_candidates: List[str] = []
    for cid in normalized_ids:
        image_candidates.extend(
            [
                f"{cid}.img.nii.gz",
                f"{cid}_img.nii.gz",
                f"{cid}.image.nii.gz",
                f"{cid}_image.nii.gz",
            ]
        )
        label_candidates.extend(
            [
                f"{cid}.label.nii.gz",
                f"{cid}_label.nii.gz",
                f"{cid}-label.nii.gz",
                f"{cid}.mask.nii.gz",
                f"{cid}_mask.nii.gz",
            ]
        )

    image_path = next((data_index[name] for name in image_candidates if name in data_index), None)
    label_path = next((data_index[name] for name in label_candidates if name in data_index), None)
    return image_path, label_path


def load_split_cases(
    split_file: Path,
    split_name: str,
    limit: int | None,
    data_index: Dict[str, str],
    remap_missing_paths: bool,
    logger: logging.Logger,
) -> List[Dict[str, str]]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        payload = json.load(f)

    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {split_file}. Keys: {list(payload.keys())}")

    raw_cases = payload[split_name]
    if not isinstance(raw_cases, list):
        raise ValueError(f"Split '{split_name}' must be a list of case dicts")

    cases: List[Dict[str, str]] = []
    skipped_missing = 0
    skipped_malformed = 0
    remapped_missing = 0
    missing_examples: List[tuple[str, str]] = []
    for rec in raw_cases:
        if not isinstance(rec, dict):
            skipped_malformed += 1
            continue
        case_id = str(rec.get("id", "")).strip()
        image = str(rec.get("image", "")).strip()
        label = str(rec.get("label", "")).strip()
        if not case_id or not image or not label:
            skipped_malformed += 1
            continue
        image_path = Path(image)
        label_path = Path(label)
        if image_path.exists() and label_path.exists():
            cases.append({"id": case_id, "image": image, "label": label})
            continue

        if remap_missing_paths:
            remap_image, remap_label = remap_case_paths_by_id(case_id, data_index)
            if remap_image and remap_label:
                remapped_missing += 1
                cases.append({"id": case_id, "image": remap_image, "label": remap_label})
                continue

        skipped_missing += 1
        if len(missing_examples) < 3:
            missing_examples.append((image, label))

    if limit is not None:
        cases = cases[:limit]

    logger.info(f"Loaded {len(cases)} cases from split='{split_name}' ({split_file})")
    if skipped_missing:
        logger.warning(f"Skipped {skipped_missing} cases with missing image/label paths")
        for image, label in missing_examples:
            logger.warning(f"Missing example -> image: {image} | label: {label}")
    if remapped_missing:
        logger.info(f"Auto-remapped {remapped_missing} cases using case_id lookup in data root index")
    if skipped_malformed:
        logger.warning(f"Skipped {skipped_malformed} malformed split records")
    return cases


def collect_checkpoints(
    checkpoint_args: List[str],
    checkpoint_glob: str | None,
    checkpoint_dir: Path,
) -> List[Path]:
    checkpoints: List[Path] = []

    for item in checkpoint_args:
        p = Path(item)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        checkpoints.append(p)

    if checkpoint_glob:
        checkpoints.extend(sorted(checkpoint_dir.glob(checkpoint_glob)))

    if not checkpoints:
        checkpoints = [checkpoint_dir / "checkpoint_best.pt"]

    uniq: Dict[Path, None] = {}
    for p in checkpoints:
        uniq[p.resolve()] = None

    def sort_key(path: Path):
        name = path.name
        m = re.search(r"checkpoint_epoch_(\d+)\.pt$", name)
        if m:
            return (0, int(m.group(1)))
        if name == "checkpoint_best.pt":
            return (1, 0)
        return (2, name)

    return sorted(uniq.keys(), key=sort_key)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_checkpoint_archive(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    size = path.stat().st_size
    if size < 4096:
        raise RuntimeError(f"Checkpoint too small to be valid ({size} bytes): {path}")

    # PyTorch's modern checkpoint format is zip-based. A missing EOCD marker
    # strongly indicates truncation/corruption during copy/save.
    eocd_sig = b"PK\x05\x06"
    with open(path, "rb") as f:
        tail_read = min(size, 1024 * 1024)
        f.seek(size - tail_read)
        tail = f.read(tail_read)
    if eocd_sig not in tail:
        raise RuntimeError(
            "Checkpoint archive appears truncated/corrupt (zip end-of-central-directory missing). "
            "Re-copy or re-save this checkpoint."
        )


def extract_case_id(batch: Dict[str, Any], fallback_index: int) -> str:
    case = batch.get("id")
    if isinstance(case, list) and case:
        return str(case[0])
    if isinstance(case, str):
        return case
    return f"case_{fallback_index:04d}"


def sanitize_case_id(case_id: str) -> str:
    clean = re.sub(r"[^\w\-\.]+", "_", str(case_id).strip())
    return clean or "unknown_case"


def build_val_transforms(train_module, cfg):
    if hasattr(train_module, "get_transforms"):
        return train_module.get_transforms(cfg, mode="val")
    if hasattr(train_module, "build_transforms"):
        return train_module.build_transforms(cfg, mode="val")
    raise AttributeError("Training module must provide get_transforms(...) or build_transforms(...).")


def build_safe_eval_transforms(cfg):
    pixdim = getattr(cfg, "pixdim", (0.6, 0.6, 0.6))
    if not isinstance(pixdim, (tuple, list)) or len(pixdim) != 3:
        pixdim = (0.6, 0.6, 0.6)
    pixdim = tuple(float(x) for x in pixdim)

    transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
    ]

    modality = str(getattr(cfg, "modality", "ct")).lower()
    if modality == "ct":
        ct_window = getattr(cfg, "ct_window", (-200.0, 700.0))
        if not isinstance(ct_window, (tuple, list)) or len(ct_window) != 2:
            ct_window = (-200.0, 700.0)
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(ct_window[0]),
                a_max=float(ct_window[1]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        transforms.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
            )
        )

    transforms.extend(
        [
            Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    return Compose(transforms)


def compute_fallback_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    labels_bin = (labels > 0.5).float()

    eps = 1e-8
    tp = float((preds * labels_bin).sum().item())
    fp = float((preds * (1.0 - labels_bin)).sum().item())
    fn = float(((1.0 - preds) * labels_bin).sum().item())
    tn = float(((1.0 - preds) * (1.0 - labels_bin)).sum().item())

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def compute_validation_metrics_adapted(train_module, logits, labels, cfg, logger):
    if hasattr(train_module, "compute_validation_metrics"):
        return train_module.compute_validation_metrics(
            preds=logits,
            labels=labels,
            metrics={},
            min_cc_size=getattr(cfg, "min_cc_size", 0),
            logger=logger,
        )
    if not getattr(compute_validation_metrics_adapted, "_warned_missing_impl", False):
        logger.warning(
            "compute_validation_metrics(...) not found in training script; "
            "using fallback binary metrics (dice/iou/precision/recall/specificity)."
        )
        compute_validation_metrics_adapted._warned_missing_impl = True
    return compute_fallback_metrics(logits, labels)


def evaluate_single_checkpoint(
    checkpoint_path: Path,
    cases: List[Dict[str, str]],
    train_module,
    device: str,
    num_workers: int,
    cache_rate: float,
    sw_batch_size_override: int | None,
    save_preds_dir: Optional[Path],
    save_pred_type: str,
    save_transformed_gt: bool,
    pred_threshold: float,
    safe_val_transforms: bool,
    logger: logging.Logger,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Evaluating {checkpoint_path}")
    validate_checkpoint_archive(checkpoint_path)
    add_safe_globals(train_module)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = state.get("config")
    if not isinstance(cfg, train_module.Config):
        logger.warning("Checkpoint missing Config. Falling back to default Config().")
        cfg = train_module.Config()

    cfg.device = device
    cfg.compile_model = False
    cfg.amp = False
    cfg.num_workers = num_workers
    cfg.persistent_workers = num_workers > 0

    model = train_module.build_model(cfg, logger)

    state_dict = state.get("model_state_dict")
    if state_dict is None:
        # Support legacy/alternate checkpoint layouts.
        state_dict = state.get("model_state") or state.get("state_dict") or state.get("model")
    if not isinstance(state_dict, dict):
        raise KeyError(
            "Checkpoint does not contain a supported model state key. "
            f"Expected one of: model_state_dict, model_state, state_dict, model. "
            f"Found keys: {list(state.keys()) if isinstance(state, dict) else type(state)}"
        )

    # If checkpoint was saved under DataParallel, strip the 'module.' prefix.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    if safe_val_transforms:
        logger.warning(
            "Using --safe-val-transforms: applying conservative eval transforms with label>0 binarization."
        )
        transform = build_safe_eval_transforms(cfg)
    else:
        transform = build_val_transforms(train_module, cfg)
    dataset = CacheDataset(
        data=cases,
        transform=transform,
        cache_rate=max(0.0, min(1.0, cache_rate)),
        num_workers=num_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
        pin_memory=device.startswith("cuda"),
        persistent_workers=(num_workers > 0),
    )

    per_case_rows: List[Dict[str, Any]] = []
    metric_values: Dict[str, List[float]] = {}
    start = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f"{checkpoint_path.name}", unit="case")):
            non_blocking = device.startswith("cuda")
            images = batch["image"].to(device, non_blocking=non_blocking)
            labels = batch["label"].to(device, non_blocking=non_blocking)

            logits = sliding_window_inference(
                images,
                roi_size=getattr(cfg, "roi_size", (96, 96, 96)),
                sw_batch_size=sw_batch_size_override or getattr(cfg, "sw_batch_size", 1),
                predictor=model,
                overlap=getattr(cfg, "sw_overlap", 0.25),
                mode="gaussian",
            )

            metrics = compute_validation_metrics_adapted(train_module, logits, labels, cfg, logger)

            case_id = extract_case_id(batch, idx)
            case_key = sanitize_case_id(case_id)
            probs_cpu = torch.sigmoid(logits).detach().cpu()
            labels_cpu = labels.detach().cpu()
            probs_np = probs_cpu[0, 0].numpy().astype(np.float32)
            pred_bin_np = (probs_np > float(pred_threshold)).astype(np.uint8)
            label_bin_np = (labels_cpu[0, 0].numpy() > 0).astype(np.uint8)

            saved_prob_path = ""
            saved_binary_path = ""
            saved_label_path = ""
            if save_preds_dir is not None:
                save_preds_dir.mkdir(parents=True, exist_ok=True)
                if save_pred_type in {"prob", "both"}:
                    prob_path = save_preds_dir / f"{case_key}.pred.npy"
                    np.save(prob_path, probs_np)
                    saved_prob_path = str(prob_path)
                if save_pred_type in {"binary", "both"}:
                    bin_path = save_preds_dir / f"{case_key}.binary.npy"
                    np.save(bin_path, pred_bin_np)
                    saved_binary_path = str(bin_path)
                if save_transformed_gt:
                    lbl_path = save_preds_dir / f"{case_key}.label.npy"
                    np.save(lbl_path, label_bin_np)
                    saved_label_path = str(lbl_path)

            row: Dict[str, Any] = {
                "checkpoint": str(checkpoint_path),
                "case_id": case_id,
                "pred_threshold": float(pred_threshold),
                "saved_prob_path": saved_prob_path,
                "saved_binary_path": saved_binary_path,
                "saved_label_path": saved_label_path,
            }
            for key, val in metrics.items():
                row[key] = float(val)
                if np.isfinite(val):
                    metric_values.setdefault(key, []).append(float(val))
            per_case_rows.append(row)

    elapsed = time.time() - start
    summary: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "epoch": int(state.get("epoch", -1)),
        "n_cases": len(per_case_rows),
        "eval_seconds": round(elapsed, 3),
        "device": device,
    }

    for key, vals in metric_values.items():
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    return summary, per_case_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on a fixed split")
    parser.add_argument("--train-script", default="train_updated copy.py", help="Path to the training script module")
    parser.add_argument("--split-file", default="splits.json", help="Path to split JSON file")
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    parser.add_argument("--data-root", default=None, help="Dataset root used for auto-remapping stale split paths")
    parser.add_argument("--no-remap-missing-paths", action="store_true", help="Disable auto-remapping missing split paths by case_id")
    parser.add_argument("--write-resolved-split", default=None, help="Optional path to write resolved split JSON for the selected split")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path (repeatable)")
    parser.add_argument("--checkpoint-glob", default=None, help="Glob pattern under --checkpoint-dir (e.g. 'checkpoint_epoch_*.pt')")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory containing checkpoints")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Evaluation device")
    parser.add_argument("--strict-device", action="store_true", help="Error instead of fallback if requested device is unavailable")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--cache-rate", type=float, default=0.0, help="MONAI CacheDataset cache rate [0,1]")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Optional override for sliding window batch size")
    parser.add_argument("--limit-cases", type=int, default=None, help="Optional max number of cases from split")
    parser.add_argument("--save-preds-dir", default=None, help="Optional directory to save per-case predictions as .npy")
    parser.add_argument("--save-pred-type", default="binary", choices=["prob", "binary", "both"], help="Prediction representation to save when --save-preds-dir is set")
    parser.add_argument("--save-transformed-gt", action="store_true", help="Save transformed binary labels (.label.npy) alongside predictions")
    parser.add_argument("--pred-threshold", type=float, default=0.5, help="Threshold used when saving binary predictions")
    parser.add_argument(
        "--safe-val-transforms",
        action="store_true",
        help="Use conservative eval transforms with label>0 binarization (helps when training script val transforms corrupt labels).",
    )
    parser.add_argument("--outdir", default="eval_outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    suppress_noisy_warnings()

    train_script = Path(args.train_script).resolve()
    split_file = Path(args.split_file).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, args.strict_device, logger)
    logger.info(
        f"Using device: {device} | cuda_available={torch.cuda.is_available()} | mps_available={_mps_available()}"
    )

    train_module = load_training_module(train_script)
    data_root = None
    data_index: Dict[str, str] = {}
    if not args.no_remap_missing_paths:
        data_root = autodetect_data_root(args.data_root, logger)
        data_index = build_data_index(data_root, logger)

    cases = load_split_cases(
        split_file=split_file,
        split_name=args.split_name,
        limit=args.limit_cases,
        data_index=data_index,
        remap_missing_paths=(not args.no_remap_missing_paths),
        logger=logger,
    )
    if not cases:
        raise RuntimeError("No valid cases found. Check split paths and --split-name.")

    if args.write_resolved_split:
        resolved_path = Path(args.write_resolved_split).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w") as f:
            json.dump({args.split_name: cases}, f, indent=2)
        logger.info(f"Wrote resolved split ({args.split_name}) to: {resolved_path}")

    checkpoints = collect_checkpoints(args.checkpoint, args.checkpoint_glob, checkpoint_dir)
    checkpoints = [p for p in checkpoints if p.exists()]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found with the provided arguments.")

    logger.info(f"Found {len(checkpoints)} checkpoint(s) to evaluate")

    summary_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    failures = 0

    for ckpt in checkpoints:
        try:
            save_preds_dir: Optional[Path] = None
            if args.save_preds_dir:
                save_preds_dir = Path(args.save_preds_dir).expanduser().resolve() / ckpt.stem / args.split_name
            summary, per_case = evaluate_single_checkpoint(
                checkpoint_path=ckpt,
                cases=cases,
                train_module=train_module,
                device=device,
                num_workers=args.num_workers,
                cache_rate=args.cache_rate,
                sw_batch_size_override=args.sw_batch_size,
                save_preds_dir=save_preds_dir,
                save_pred_type=args.save_pred_type,
                save_transformed_gt=bool(args.save_transformed_gt),
                pred_threshold=float(args.pred_threshold),
                safe_val_transforms=bool(args.safe_val_transforms),
                logger=logger,
            )
            summary_rows.append(summary)
            all_case_rows.extend(per_case)

            ckpt_case_csv = outdir / f"{ckpt.stem}_per_case.csv"
            write_csv(ckpt_case_csv, per_case)
            logger.info(f"Wrote per-case metrics: {ckpt_case_csv}")
            if save_preds_dir is not None:
                logger.info(f"Wrote per-case predictions: {save_preds_dir}")
        except Exception as e:
            failures += 1
            logger.error(f"Failed checkpoint {ckpt}: {e}")
            summary_rows.append(
                {
                    "checkpoint": str(ckpt),
                    "checkpoint_name": ckpt.name,
                    "n_cases": 0,
                    "status": "failed",
                    "error": str(e),
                }
            )

    summary_csv = outdir / "checkpoint_summary.csv"
    details_csv = outdir / "checkpoint_case_metrics.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(details_csv, all_case_rows)

    with open(outdir / "checkpoint_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    logger.info(f"Wrote summary CSV: {summary_csv}")
    logger.info(f"Wrote combined per-case CSV: {details_csv}")
    if failures:
        logger.warning(f"{failures} checkpoint(s) failed. See checkpoint_summary.csv for error details.")
    logger.info(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
