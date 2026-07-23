#!/usr/bin/env python3
"""A40 full held-out ImageCAS test evaluation with optional Phase B QC."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.data import CacheDataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import evaluate_full_imagecas_test_mps as mps_eval
from scripts.checkpoint_utils import load_checkpoint_for_model, safe_torch_load


DEFAULT_CHANNELS = (32, 64, 128, 256, 512)
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_updated copy.py"


def parse_tuple(text: str, cast=float, expected: Optional[int] = None) -> Tuple[Any, ...]:
    vals = tuple(cast(x.strip()) for x in str(text).split(",") if x.strip())
    if expected is not None and len(vals) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values, got {text!r}")
    return vals


def import_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("repo_train_updated", str(TRAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but CUDA is unavailable")
    if device.type == "mps" and not _mps_available():
        raise RuntimeError("Requested MPS but MPS is unavailable")
    return device


def append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with open(path, newline="") as f:
            old = next(csv.reader(f), None)
        if old:
            fieldnames = list(dict.fromkeys(old + fieldnames))
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_cases(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row.get("case_id", "") for row in csv.DictReader(f)}


def affine_from_batch(batch: Dict[str, Any]) -> np.ndarray:
    image = batch.get("image")
    affine = None
    if hasattr(image, "meta"):
        affine = image.meta.get("affine")
    if affine is None and "image_meta_dict" in batch:
        affine = batch["image_meta_dict"].get("affine")
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if isinstance(affine, np.ndarray):
        affine = np.squeeze(affine)
        if affine.shape == (4, 4):
            return affine
    return np.eye(4)


def summarize(per_case_csv: Path, outdir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    if per_case_csv.exists():
        with open(per_case_csv, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    numeric_keys = set()
    for row in rows:
        for key, value in row.items():
            try:
                float(value)
                numeric_keys.add(key)
            except Exception:
                pass
    summary_rows = []
    summary_json: Dict[str, Any] = {"n_cases": len(rows), "failed_count": 0, "metrics": {}}
    failed_path = outdir / "failed_cases.csv"
    if failed_path.exists():
        with open(failed_path, newline="") as f:
            summary_json["failed_count"] = sum(1 for _ in csv.DictReader(f))
    for key in sorted(numeric_keys):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        q1, q3 = np.nanpercentile(arr, [25, 75])
        ci = 1.96 * float(np.nanstd(arr, ddof=1)) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        stats = {
            "metric": key,
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.nanmedian(arr)),
            "iqr": float(q3 - q1),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "ci95_low": float(np.nanmean(arr) - ci),
            "ci95_high": float(np.nanmean(arr) + ci),
            "n": len(arr),
        }
        summary_rows.append(stats)
        summary_json["metrics"][key] = {k: v for k, v in stats.items() if k != "metric"}
    write_csv(outdir / "summary_metrics.csv", summary_rows)
    (outdir / "summary_metrics.json").write_text(json.dumps(summary_json, indent=2))


def build_model(train_mod: Any, device: torch.device, model_type: str):
    cfg = train_mod.Config()
    cfg.device = device.type
    cfg.unet_channels = DEFAULT_CHANNELS
    cfg.unet_strides = (2, 2, 2, 2)
    cfg.unet_num_res_units = 3
    cfg.unet_dropout = 0.1
    cfg.unet_norm = "batch"
    cfg.use_attention = model_type == "attention_unet"
    log_dir = Path("outputs/eval_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = train_mod.setup_logging(log_dir, "a40_eval_model")
    return train_mod.build_model(cfg, logger)


def run_phaseb_and_fabrication(
    case_id: str,
    mask_path: Path,
    minimal_mesh_qc: bool = False,
    run_fabrication: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import phaseb_mesh_qc
        import fabrication_readiness_validation as fab

        mesh_row = phaseb_mesh_qc.run_phaseb_for_case(
            case_id,
            mask_path,
            "outputs/phase_b_mesh_qc",
            minimal=minimal_mesh_qc,
        )
        out.update({f"mesh_{k}": v for k, v in mesh_row.items() if k in {"status", "repaired_stl", "watertight", "non_manifold_edge_count"}})
        stl = mesh_row.get("repaired_stl")
        if stl and run_fabrication:
            fab_row = fab.validate_stl(
                stl,
                case_id=case_id,
                output_root="outputs/fabrication_readiness",
                wall_thickness_compliance_fraction=mesh_row.get("wall_thickness_compliance_fraction", "not_computed"),
                smoothing_volume_change_relative=mesh_row.get("smoothing_volume_change_relative", "not_computed"),
            )
            out.update({f"fabrication_{k}": v for k, v in fab_row.items() if k in {"status", "slicer_compatibility_proxy", "watertight_after_repair"}})
    except Exception as exc:
        out["phaseb_qc_error"] = repr(exc)
    return out


def evaluate(args: argparse.Namespace) -> int:
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    thresholds = parse_tuple(args.thresholds, float)
    if args.smoke_test:
        args.limit = args.limit or 2
    elif args.limit is None and not (args.resume or args.skip_existing):
        print("Default safety behavior: running 2-case smoke test. Pass --resume --skip_existing for full evaluation.")
        args.limit = 2

    train_mod = import_train_module()
    model = build_model(train_mod, device, args.model_type)
    load_diag = load_checkpoint_for_model(model, args.checkpoint, strict=True, device="cpu")
    model.to(device).eval()
    checkpoint_state = safe_torch_load(args.checkpoint, map_location="cpu")
    checkpoint_info = {
        "path": str(args.checkpoint),
        "epoch": checkpoint_state.get("epoch") if isinstance(checkpoint_state, dict) else None,
        "best_metric": checkpoint_state.get("best_metric") if isinstance(checkpoint_state, dict) else None,
        "load_diag": {k: v for k, v in load_diag.items() if k != "raw_checkpoint"},
    }

    if args.split != "test":
        raise ValueError("evaluate_full_test_a40.py currently supports only --split test")

    data_root = args.imagecas_root if args.imagecas_root is not None else args.data_root
    thresholds_arg = args.threshold if args.threshold is not None else args.thresholds
    thresholds = parse_tuple(thresholds_arg, float)

    proxy_args = argparse.Namespace(
        split_file=args.splits_json,
        data_root=data_root,
        expected_test_cases=args.expected_test_cases,
        case_start=args.case_start,
        limit=args.limit,
    )
    cases = mps_eval.load_cases(proxy_args, mps_eval.setup_logger(outdir))
    transform_args = argparse.Namespace(pixdim=(0.6, 0.6, 0.6), ct_window=(-200.0, 700.0), modality="ct")
    transform = mps_eval.build_eval_transforms(transform_args)
    ds = CacheDataset(data=cases, transform=transform, cache_rate=0.0, num_workers=args.num_workers)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=pad_list_data_collate)

    config = vars(args).copy()
    config.update({"thresholds": list(thresholds), "checkpoint": checkpoint_info, "device_resolved": str(device)})
    (outdir / "used_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (outdir / "eval_command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    (outdir / "README_results.md").write_text(
        "# Full Test Evaluation Results\n\n"
        "Headline manuscript metrics should come from this held-out test cohort. Training Dice is not the primary result.\n"
    )

    per_case_csv = outdir / "per_case_metrics.csv"
    failed_csv = outdir / "failed_cases.csv"
    done = existing_cases(per_case_csv) if (args.resume or args.skip_existing) else set()
    progress = {"completed": len(done), "failed": 0, "last_case_id": None}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="a40-test-eval", unit="case")):
            case_id = mps_eval.extract_case_id(batch, idx)
            if args.skip_existing and case_id in done:
                continue
            start = time.time()
            try:
                image = batch["image"].to(device, non_blocking=True)
                label_cpu = (batch["label"].detach().cpu().numpy()[0, 0] > 0).astype(np.uint8)
                logits = sliding_window_inference(
                    image,
                    roi_size=parse_tuple(args.roi_size, int, 3),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.sw_overlap,
                    mode="gaussian",
                    sw_device=device,
                    device="cpu" if args.val_output_device == "cpu" else device,
                )
                probs = torch.sigmoid(logits).detach().cpu().numpy()[0, 0].astype(np.float32)
                spacing = mps_eval.spacing_from_batch(batch, (0.6, 0.6, 0.6))
                row: Dict[str, Any] = {
                    "case_id": case_id,
                    "image": cases[idx]["image"],
                    "label": cases[idx]["label"],
                    "status": "ok",
                    "soft_dice": mps_eval.soft_dice(probs, label_cpu),
                    "gt_voxels": int(label_cpu.sum()),
                    "runtime_seconds": round(time.time() - start, 3),
                }
                for threshold in thresholds:
                    pred = probs > threshold
                    metrics = mps_eval.binary_metrics(pred, label_cpu)
                    suffix = f"@{threshold:.1f}"
                    row[f"dice{suffix}"] = metrics["dice"]
                    row[f"precision{suffix}"] = metrics["precision"]
                    row[f"recall{suffix}"] = metrics["recall"]
                    row[f"fp_voxels{suffix}"] = metrics["fp_voxels"]
                    row[f"fn_voxels{suffix}"] = metrics["fn_voxels"]
                    row[f"pred_voxels{suffix}"] = int(pred.sum())
                if args.compute_hd95:
                    try:
                        row["hd95@0.5"] = mps_eval.hd95_cpu(probs > 0.5, label_cpu, spacing)
                    except Exception as exc:
                        row["hd95@0.5"] = "missing_dependency"
                        row["hd95_error"] = repr(exc)
                if args.compute_cldice:
                    try:
                        row["cldice@0.5"] = mps_eval.cldice_cpu(probs > 0.5, label_cpu)
                    except Exception as exc:
                        row["cldice@0.5"] = "missing_dependency"
                        row["cldice_error"] = repr(exc)
                case_dir = outdir / "case_outputs" / mps_eval.sanitize_case_id(case_id)
                mask05_path: Optional[Path] = None
                if args.save_case_outputs or args.smoke_test:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    affine = affine_from_batch(batch)
                    if not args.skip_probability_outputs:
                        prob_path = case_dir / "seg_prob.nii.gz"
                        nib.Nifti1Image(probs, affine).to_filename(str(prob_path))
                        row["seg_prob_path"] = str(prob_path)
                    for threshold in thresholds:
                        mask_path = case_dir / f"seg_mask_{threshold:.1f}.nii.gz"
                        nib.Nifti1Image((probs > threshold).astype(np.uint8), affine).to_filename(str(mask_path))
                        if abs(threshold - 0.5) < 1e-6:
                            mask05_path = mask_path
                    (case_dir / "metadata.json").write_text(json.dumps({"case_id": case_id, "spacing": spacing}, indent=2))
                if args.run_phase_b_qc and mask05_path:
                    row.update(
                        run_phaseb_and_fabrication(
                            case_id,
                            mask05_path,
                            minimal_mesh_qc=args.minimal_phase_b_qc,
                            run_fabrication=not args.skip_fabrication_qc,
                        )
                    )
                append_row(per_case_csv, row)
                done.add(case_id)
                progress["completed"] = len(done)
                progress["last_case_id"] = case_id
            except Exception as exc:
                append_row(
                    failed_csv,
                    {"case_id": case_id, "image": cases[idx].get("image", ""), "label": cases[idx].get("label", ""), "error": repr(exc)},
                )
                progress["failed"] += 1
            (outdir / "progress.json").write_text(json.dumps(progress, indent=2))

    summarize(per_case_csv, outdir)
    if args.run_phase_b_qc:
        try:
            import phaseb_mesh_qc
            import fabrication_readiness_validation as fab

            phaseb_mesh_qc.summarize_mesh_qc("outputs/phase_b_mesh_qc")
            fab.summarize("outputs/fabrication_readiness")
        except Exception:
            pass
    if args.smoke_test:
        full_cmd = (
            f"CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 "
            f"python evaluate_full_test_a40.py --checkpoint {args.checkpoint} --splits_json {args.splits_json} "
            f"--output_dir {args.output_dir} --device cuda --roi_size {args.roi_size} --sw_batch_size 1 --resume --skip_existing"
        )
        print("\nFull 250-case command:\n" + full_cmd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate full 250-case ImageCAS test split on A40/CUDA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits_json", default="splits.json")
    parser.add_argument("--data_root", default="Data/all")
    parser.add_argument("--imagecas_root", default=None)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--output_dir", default="outputs/full_test_eval_a40")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--roi_size", default="96,192,192")
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--case_start", type=int, default=0)
    parser.add_argument("--save_case_outputs", action="store_true")
    parser.add_argument(
        "--skip_probability_outputs",
        action="store_true",
        help="save thresholded masks but omit large float probability NIfTI files",
    )
    parser.add_argument("--run_phase_b_qc", action="store_true")
    parser.add_argument(
        "--minimal_phase_b_qc",
        action="store_true",
        help="export the repaired affine-correct mesh but skip unrelated smoothing and thickness metrics",
    )
    parser.add_argument(
        "--skip_fabrication_qc",
        action="store_true",
        help="skip fabrication proxy calculations when only mesh alignment checks are required",
    )
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--val_output_device", default="same", choices=["same", "cpu"])
    parser.add_argument("--model_type", default="attention_unet", choices=["attention_unet", "plain_unet"])
    parser.add_argument("--expected_test_cases", type=int, default=250)
    parser.add_argument("--compute_hd95", action="store_true", default=True)
    parser.add_argument("--compute_cldice", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
