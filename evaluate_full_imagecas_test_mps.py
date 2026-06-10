#!/usr/bin/env python3
"""Full ImageCAS 250-case held-out test evaluation for Phase A.

This is intentionally stricter than the checkpoint-sweep helper:
- evaluates only the `test` partition from splits.json
- asserts the full split size is 250 before any limit/case_start slicing
- writes outputs after every case so long MPS runs are resumable
- computes thresholded voxel metrics, optional HD95, and optional clDice
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    _tmp_base = Path(tempfile.gettempdir())
except Exception:
    _tmp_base = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str((_tmp_base / "mplconfig").resolve()))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
    try:
        high = float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
        low = float(os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.4"))
        if high > 0.0 and low >= high:
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(max(0.0, high * 0.8))
    except ValueError:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

import numpy as np
import torch
import yaml
from monai.data import CacheDataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, UNet
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
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    from skimage.morphology import skeletonize, skeletonize_3d  # type: ignore
except Exception:
    skeletonize = None  # type: ignore
    skeletonize_3d = None  # type: ignore


THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def setup_logger(outdir: Path) -> logging.Logger:
    logger = logging.getLogger("full_imagecas_test_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    outdir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(outdir / "eval.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def suppress_noisy_warnings() -> None:
    warnings.filterwarnings("ignore", message=r".*Orientationd\.__init__:labels.*", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=r".*Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
    )


def mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("mps" if mps_available() else "cpu")
    if device_arg == "mps" and not mps_available():
        raise RuntimeError("Requested --device mps, but MPS is not available.")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def parse_csv_tuple(text: str, cast=float, expected: Optional[int] = None) -> Tuple[Any, ...]:
    vals = [cast(x.strip()) for x in text.split(",")]
    if expected is not None and len(vals) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values, got: {text}")
    return tuple(vals)  # type: ignore[return-value]


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("repo_train_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_checkpoint_safe_globals(train_module: Any) -> None:
    safe = [np._core.multiarray.scalar, np.dtype]
    if hasattr(train_module, "Config"):
        safe.append(train_module.Config)
    torch.serialization.add_safe_globals(safe)


def import_eval_utils():
    import evaluate_checkpoints as ec  # type: ignore

    return ec


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def sanitize_case_id(case_id: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(case_id)).strip("_") or "unknown_case"


def extract_case_id(batch: Dict[str, Any], idx: int) -> str:
    raw = batch.get("id")
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if isinstance(raw, tuple) and raw:
        return str(raw[0])
    if isinstance(raw, str):
        return raw
    return f"case_{idx:04d}"


def get_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(state)}")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = state.get(key)
        if isinstance(value, dict):
            if any(k.startswith("module.") for k in value):
                return {k.removeprefix("module."): v for k, v in value.items()}
            return value
    raise KeyError(f"No supported model state key found. Keys: {list(state.keys())}")


def infer_unet_channels(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    found: List[int] = []
    for key, value in state_dict.items():
        if key.endswith("conv.unit0.conv.weight") and hasattr(value, "shape") and len(value.shape) == 5:
            out_ch = int(value.shape[0])
            if out_ch not in found and out_ch > 1:
                found.append(out_ch)
    if len(found) >= 2:
        return tuple(found[:5])
    return (64, 128, 256, 512, 1024)


def state_looks_like_attention(state_dict: Dict[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    return any("attention" in k.lower() or "att" in k.lower() or "gating" in k.lower() for k in keys)


def infer_unet_norm(state_dict: Dict[str, torch.Tensor], fallback: Optional[str]) -> Optional[str]:
    keys = list(state_dict.keys())
    if any(".adn.N." in k for k in keys):
        return fallback or "batch"
    return None


def build_model_for_checkpoint(
    state: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    train_module: Any,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
):
    cfg = state.get("config")
    if cfg is not None and hasattr(cfg, "__dict__"):
        channels = tuple(getattr(cfg, "unet_channels", (64, 128, 256, 512, 1024)))
        strides = tuple(getattr(cfg, "unet_strides", (2, 2, 2, 2)))
        dropout = float(getattr(cfg, "unet_dropout", 0.1))
        num_res_units = int(getattr(cfg, "unet_num_res_units", 3))
        norm = getattr(cfg, "unet_norm", "batch")
        use_attention = bool(getattr(cfg, "use_attention", False))
    else:
        channels = tuple(args.unet_channels) if args.unet_channels else infer_unet_channels(state_dict)
        strides = (2,) * (len(channels) - 1)
        dropout = float(args.dropout)
        num_res_units = int(args.unet_res_units)
        norm = infer_unet_norm(state_dict, args.unet_norm)
        use_attention = bool(args.use_attention or state_looks_like_attention(state_dict))
        logger.warning(
            "Checkpoint has no embedded training Config; inferred/CLI model settings are "
            f"use_attention={use_attention}, channels={channels}, strides={strides}, "
            f"res_units={num_res_units}, dropout={dropout}, norm={norm}."
        )

    if args.force_attention:
        use_attention = True
    if args.force_unet:
        use_attention = False

    if use_attention:
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )
        arch = "AttentionUnet"
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
            norm=norm,
        )
        arch = "UNet"

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {arch}. Missing keys={len(missing)} "
            f"unexpected keys={len(unexpected)}. First missing={missing[:5]}, first unexpected={unexpected[:5]}"
        )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {arch}: channels={channels}, params={param_count:,}")
    return model, {
        "architecture": arch,
        "channels": list(channels),
        "strides": list(strides),
        "dropout": dropout,
        "num_res_units": num_res_units,
        "norm": norm,
        "param_count": param_count,
    }


def build_eval_transforms(args: argparse.Namespace):
    transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=tuple(args.pixdim), mode=("bilinear", "nearest")),
    ]
    if args.modality == "ct":
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(args.ct_window[0]),
                a_max=float(args.ct_window[1]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        transforms.append(NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))
    transforms.extend(
        [
            Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    return Compose(transforms)


def soft_dice(probs: np.ndarray, label: np.ndarray) -> float:
    probs_f = probs.astype(np.float64, copy=False)
    label_f = label.astype(np.float64, copy=False)
    return float((2.0 * np.sum(probs_f * label_f) + 1e-6) / (np.sum(probs_f) + np.sum(label_f) + 1e-6))


def binary_metrics(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    pred_b = pred.astype(bool, copy=False)
    label_b = label.astype(bool, copy=False)
    tp = int(np.logical_and(pred_b, label_b).sum())
    fp = int(np.logical_and(pred_b, ~label_b).sum())
    fn = int(np.logical_and(~pred_b, label_b).sum())
    denom_d = 2 * tp + fp + fn
    denom_p = tp + fp
    denom_r = tp + fn
    return {
        "dice": float((2 * tp) / denom_d) if denom_d else 1.0,
        "precision": float(tp / denom_p) if denom_p else (1.0 if denom_r == 0 else 0.0),
        "recall": float(tp / denom_r) if denom_r else 1.0,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tp_voxels": tp,
    }


def hd95_cpu(pred: np.ndarray, label: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    pred_b = pred.astype(bool, copy=False)
    label_b = label.astype(bool, copy=False)
    if not pred_b.any() and not label_b.any():
        return 0.0
    if not pred_b.any() or not label_b.any():
        return float("nan")
    structure = ndi.generate_binary_structure(3, 1)
    pred_surface = np.logical_xor(pred_b, ndi.binary_erosion(pred_b, structure=structure, border_value=0))
    label_surface = np.logical_xor(label_b, ndi.binary_erosion(label_b, structure=structure, border_value=0))
    if not pred_surface.any() or not label_surface.any():
        return float("nan")
    dt_label = ndi.distance_transform_edt(~label_surface, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_label[pred_surface], dt_pred[label_surface]])
    return float(np.percentile(distances, 95)) if distances.size else float("nan")


def cldice_cpu(pred: np.ndarray, label: np.ndarray) -> float:
    if skeletonize_3d is None and skeletonize is None:
        raise RuntimeError("skimage.morphology.skeletonize_3d/skeletonize is unavailable")
    pred_b = pred.astype(bool, copy=False)
    label_b = label.astype(bool, copy=False)
    if not pred_b.any() and not label_b.any():
        return 1.0
    if not pred_b.any() or not label_b.any():
        return 0.0
    skel_fn = skeletonize_3d if skeletonize_3d is not None else skeletonize
    skel_pred = skel_fn(pred_b) > 0
    skel_label = skel_fn(label_b) > 0
    tprec = np.logical_and(skel_pred, label_b).sum() / max(1, skel_pred.sum())
    tsens = np.logical_and(skel_label, pred_b).sum() / max(1, skel_label.sum())
    return float((2.0 * tprec * tsens) / (tprec + tsens)) if (tprec + tsens) else 0.0


def spacing_from_batch(batch: Dict[str, Any], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    meta = batch.get("label_meta_dict") or batch.get("image_meta_dict") or {}
    pixdim = meta.get("pixdim") if isinstance(meta, dict) else None
    if pixdim is not None:
        arr = pixdim.detach().cpu().numpy() if isinstance(pixdim, torch.Tensor) else np.asarray(pixdim)
        flat = arr.reshape(-1)
        if flat.size >= 4:
            return tuple(float(x) for x in flat[1:4])  # type: ignore[return-value]
    return fallback


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "median": math.nan, "iqr": math.nan, "min": math.nan, "max": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    q25, q75 = np.percentile(arr, [25, 75])
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_low": float(np.mean(arr) - 1.96 * se),
        "ci95_high": float(np.mean(arr) + 1.96 * se),
    }


def existing_case_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as f:
        return {row["case_id"] for row in csv.DictReader(f) if row.get("case_id")}


def append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_cases(args: argparse.Namespace, logger: logging.Logger) -> List[Dict[str, str]]:
    ec = import_eval_utils()
    data_root = ec.autodetect_data_root(args.data_root, logger)
    data_index = ec.build_data_index(data_root, logger)
    full_cases = ec.load_split_cases(
        split_file=Path(args.split_file).resolve(),
        split_name="test",
        limit=None,
        data_index=data_index,
        remap_missing_paths=True,
        logger=logger,
    )
    if len(full_cases) != args.expected_test_cases:
        raise AssertionError(
            f"Expected {args.expected_test_cases} test cases in {args.split_file}, got {len(full_cases)}. "
            "This protects against validation/test leakage or accidental pilot-subset evaluation."
        )
    cases = full_cases[args.case_start :]
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def write_config(args: argparse.Namespace, outdir: Path, model_info: Dict[str, Any], checkpoint_info: Dict[str, Any]) -> None:
    config = vars(args).copy()
    config["model_info"] = model_info
    config["checkpoint_info"] = checkpoint_info
    config["thresholds"] = list(THRESHOLDS)
    config["mps_available"] = mps_available()
    config["cuda_available"] = torch.cuda.is_available()
    config["platform"] = platform.platform()
    with open(outdir / "used_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    (outdir / "eval_command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")


def finalize_outputs(outdir: Path, logger: logging.Logger) -> None:
    per_case_path = outdir / "per_case_metrics.csv"
    if not per_case_path.exists() or per_case_path.stat().st_size == 0:
        write_csv(outdir / "summary_metrics.csv", [])
        save_json(outdir / "summary_metrics.json", {"n_cases": 0, "metrics": []})
        return
    with open(per_case_path, newline="") as f:
        rows = list(csv.DictReader(f))
    metric_keys: List[str] = []
    for row in rows:
        for key, val in row.items():
            if key in {"case_id", "image", "label", "status", "error", "seconds", "device"}:
                continue
            try:
                float(val)
            except Exception:
                continue
            if key not in metric_keys:
                metric_keys.append(key)
    summary_rows = []
    summary_json: Dict[str, Any] = {"n_cases": len(rows), "metrics": {}}
    for key in metric_keys:
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(key, "")))
            except Exception:
                pass
        stats = summarize(vals)
        summary_rows.append({"metric": key, **stats})
        summary_json["metrics"][key] = stats
    write_csv(outdir / "summary_metrics.csv", summary_rows)
    save_json(outdir / "summary_metrics.json", summary_json)
    logger.info(f"Wrote summary for {len(rows)} cases")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full 250-case ImageCAS test split on MPS/CPU")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt")
    parser.add_argument(
        "--allow-external-checkpoint",
        action="store_true",
        help="Allow --checkpoint to point outside the current repository. Default is repo-local only.",
    )
    parser.add_argument("--train-script", default="train_updated copy.py")
    parser.add_argument("--split-file", default="splits.json")
    parser.add_argument("--data-root", default="Data/all")
    parser.add_argument("--outdir", default="outputs/full_test_eval_mps")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--expected-test-cases", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--roi-size", type=lambda s: parse_csv_tuple(s, int, 3), default=(96, 192, 192))
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--pixdim", type=lambda s: parse_csv_tuple(s, float, 3), default=(0.6, 0.6, 0.6))
    parser.add_argument("--ct-window", type=lambda s: parse_csv_tuple(s, float, 2), default=(-200.0, 700.0))
    parser.add_argument("--modality", default="ct", choices=["ct", "mri"])
    parser.add_argument("--hd95-threshold", type=float, default=0.5)
    parser.add_argument("--no-hd95", action="store_true")
    parser.add_argument("--compute-cldice", action="store_true")
    parser.add_argument("--cldice-threshold", type=float, default=0.5)
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--save-binary", action="store_true")
    parser.add_argument("--unet-channels", type=lambda s: parse_csv_tuple(s, int, 5), default=None, help="Fallback only when checkpoint has no Config.")
    parser.add_argument("--unet-res-units", type=int, default=3)
    parser.add_argument("--unet-norm", default="batch")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--force-attention", action="store_true")
    parser.add_argument("--force-unet", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    logger = setup_logger(outdir)
    suppress_noisy_warnings()
    device = resolve_device(args.device)
    logger.info(f"Using device={device} | mps_available={mps_available()} | cuda_available={torch.cuda.is_available()}")

    train_module = load_module(Path(args.train_script).resolve())
    add_checkpoint_safe_globals(train_module)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not args.allow_external_checkpoint and not is_relative_to(checkpoint, Path.cwd()):
        raise RuntimeError(
            f"Checkpoint is outside the current cloud_bundle repo: {checkpoint}. "
            "Copy the correct checkpoint into cloud_bundle/checkpoints/ or pass "
            "--allow-external-checkpoint intentionally."
        )
    import_eval_utils().validate_checkpoint_archive(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = get_state_dict(state)
    model, model_info = build_model_for_checkpoint(state, state_dict, train_module, args, device, logger)
    checkpoint_info = {
        "path": str(checkpoint),
        "epoch": state.get("epoch") if isinstance(state, dict) else None,
        "best_metric": state.get("best_metric") if isinstance(state, dict) else None,
        "best_loss": state.get("best_loss") if isinstance(state, dict) else None,
        "keys": list(state.keys()) if isinstance(state, dict) else [],
    }
    write_config(args, outdir, model_info, checkpoint_info)

    cases = load_cases(args, logger)
    logger.info(f"Evaluation cases after case_start/limit: {len(cases)}")
    transform = build_eval_transforms(args)
    dataset = CacheDataset(data=cases, transform=transform, cache_rate=max(0.0, min(1.0, args.cache_rate)), num_workers=args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_list_data_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    per_case_csv = outdir / "per_case_metrics.csv"
    failed_csv = outdir / "failed_cases.csv"
    completed = existing_case_ids(per_case_csv) if (args.resume or args.skip_existing) else set()
    pred_dir = outdir / "predictions"
    progress = {"completed": len(completed), "failed": 0, "last_case_id": None}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="full-test", unit="case")):
            case_id = extract_case_id(batch, idx)
            if args.skip_existing and case_id in completed:
                continue
            start = time.time()
            try:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = sliding_window_inference(
                    images,
                    roi_size=tuple(args.roi_size),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.sw_overlap,
                    mode="gaussian",
                )
                probs = torch.sigmoid(logits).detach().cpu().numpy()[0, 0].astype(np.float32)
                label = (labels.detach().cpu().numpy()[0, 0] > 0).astype(np.uint8)
                spacing = spacing_from_batch(batch, tuple(args.pixdim))
                row: Dict[str, Any] = {
                    "case_id": case_id,
                    "image": cases[idx]["image"],
                    "label": cases[idx]["label"],
                    "status": "ok",
                    "device": str(device),
                    "soft_dice": soft_dice(probs, label),
                    "pred_voxels@0.5": int((probs > 0.5).sum()),
                    "gt_voxels": int(label.sum()),
                }
                for threshold in THRESHOLDS:
                    pred = probs > threshold
                    m = binary_metrics(pred, label)
                    suffix = f"@{threshold:.1f}"
                    row[f"dice{suffix}"] = m["dice"]
                    row[f"precision{suffix}"] = m["precision"]
                    row[f"recall{suffix}"] = m["recall"]
                    row[f"fp_voxels{suffix}"] = m["fp_voxels"]
                    row[f"fn_voxels{suffix}"] = m["fn_voxels"]
                    row[f"pred_voxels{suffix}"] = int(pred.sum())
                if not args.no_hd95:
                    row[f"hd95@{args.hd95_threshold:.1f}"] = hd95_cpu(probs > args.hd95_threshold, label, spacing)
                if args.compute_cldice:
                    try:
                        row[f"cldice@{args.cldice_threshold:.1f}"] = cldice_cpu(probs > args.cldice_threshold, label)
                    except Exception as e:
                        row[f"cldice@{args.cldice_threshold:.1f}"] = float("nan")
                        row["cldice_error"] = str(e)
                if args.save_probs or args.save_binary:
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    key = sanitize_case_id(case_id)
                    if args.save_probs:
                        np.save(pred_dir / f"{key}.pred.npy", probs)
                    if args.save_binary:
                        np.save(pred_dir / f"{key}.binary.npy", (probs > 0.5).astype(np.uint8))
                row["seconds"] = round(time.time() - start, 3)
                append_row(per_case_csv, row)
                completed.add(case_id)
                progress["completed"] = len(completed)
                progress["last_case_id"] = case_id
            except Exception as e:
                fail = {
                    "case_id": case_id,
                    "image": cases[idx].get("image", ""),
                    "label": cases[idx].get("label", ""),
                    "error": repr(e),
                    "seconds": round(time.time() - start, 3),
                }
                append_row(failed_csv, fail)
                progress["failed"] = int(progress.get("failed", 0)) + 1
                logger.exception(f"Failed case {case_id}")
            save_json(outdir / "progress.json", progress)

    finalize_outputs(outdir, logger)
    logger.info(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
