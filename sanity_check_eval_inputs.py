#!/usr/bin/env python3
"""Fast preflight sanity checker for Phase A evaluation inputs.

This script validates prediction/GT discovery, ID matching, mask sanity, and
prediction numeric type assumptions before full evaluation runs.

Example:
  python sanity_check_eval_inputs.py \
    --pred_dir eval_outputs/my_preds \
    --gt_dir data/processed/all \
    --output_dir eval_outputs/verification_runs/2026-02-22_sanity
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

try:
    import SimpleITK as sitk  # type: ignore
except Exception:
    sitk = None

try:
    import scipy  # type: ignore
except Exception:
    scipy = None

try:
    import skimage  # type: ignore
except Exception:
    skimage = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import monai  # type: ignore
except Exception:
    monai = None


ALLOWED_EXTS = (".nii.gz", ".nii", ".npy")
PRED_HINTS = ("pred", "prob", "logit", "mask", "binary", "output", "phasea")
GT_HINTS = ("label", "mask", "seg")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("sanity_eval_inputs")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _pkg_version(mod: Any) -> Optional[str]:
    if mod is None:
        return None
    return getattr(mod, "__version__", None)


def local_mps_available() -> bool:
    if torch is None:
        return False
    if not hasattr(torch.backends, "mps"):
        return False
    try:
        return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except Exception:
        return False


def collect_environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": _pkg_version(np),
        "torch_version": _pkg_version(torch),
        "monai_version": _pkg_version(monai),
        "nibabel_version": _pkg_version(nib),
        "scipy_version": _pkg_version(scipy),
        "skimage_version": _pkg_version(skimage),
        "cuda_available": False,
        "mps_available": local_mps_available(),
        "dependency_warnings": [],
    }
    if torch is not None:
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            info["cuda_available"] = False

    dep_warnings: List[str] = []
    if nib is None and sitk is None:
        dep_warnings.append("nibabel and SimpleITK both unavailable: cannot read NIfTI files.")
    if scipy is None:
        dep_warnings.append("scipy unavailable: connected-components/distance-based checks limited.")
    if skimage is None:
        dep_warnings.append("skimage unavailable: clDice skeleton-based validation unavailable.")
    info["dependency_warnings"] = dep_warnings
    return info


def strip_known_ext(name: str) -> str:
    lower = name.lower()
    for ext in ALLOWED_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def canonical_case_id(raw: Any) -> str:
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s.lower()


def case_id_candidates_from_text(text: str) -> List[str]:
    base = strip_known_ext(text)
    cleaned = re.sub(r"(?:_|-)?(?:pred|prob|logit|mask|binary|post|output|seg|label|img|image)$", "", base, flags=re.IGNORECASE)
    cands: List[str] = []
    for token in (base, cleaned):
        cid = canonical_case_id(token)
        if cid:
            cands.append(cid)
    for m in re.findall(r"\d+", base):
        cid = canonical_case_id(m)
        if cid:
            cands.append(cid)
    dedup: List[str] = []
    seen: set[str] = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def case_id_candidates_from_path(path: Path) -> List[str]:
    cands: List[str] = []
    cands.extend(case_id_candidates_from_text(path.name))
    for part in reversed(path.parts[-4:]):
        cands.extend(case_id_candidates_from_text(part))
    dedup: List[str] = []
    seen: set[str] = set()
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def scan_files(root: Path, allowed_exts: Sequence[str] = ALLOWED_EXTS) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for ext in allowed_exts:
        out.extend(root.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in out if p.is_file()})


def discover_pred_files(pred_dir: Optional[str]) -> Tuple[List[Path], str]:
    if pred_dir:
        root = Path(pred_dir).expanduser().resolve()
        return scan_files(root), str(root)
    roots = [Path("eval_outputs").resolve(), Path("pipeline_outputs").resolve()]
    discovered: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in scan_files(root):
            low = p.as_posix().lower()
            if any(k in low for k in PRED_HINTS):
                discovered.append(p)
    return sorted({p.resolve() for p in discovered}), "auto_discovery"


def autodetect_gt_root(gt_dir: Optional[str]) -> Optional[Path]:
    if gt_dir:
        p = Path(gt_dir).expanduser().resolve()
        return p if p.exists() else None
    candidates = [
        (Path.cwd() / "data" / "processed" / "all").resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def derive_primary_case_id(path: Path) -> str:
    cands = case_id_candidates_from_path(path)
    if not cands:
        return ""
    for c in cands:
        if re.fullmatch(r"\d+", c):
            return str(int(c))
    return cands[0]


def build_inventory(files: Sequence[Path], role: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in files:
        cands = case_id_candidates_from_path(p)
        rows.append(
            {
                "role": role,
                "path": str(p),
                "filename": p.name,
                "case_id_primary": derive_primary_case_id(p),
                "case_id_candidates": "|".join(cands),
                "suffix": "".join(p.suffixes),
            }
        )
    return rows


def build_id_map(files: Sequence[Path]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for p in files:
        cid = derive_primary_case_id(p)
        if not cid:
            continue
        out.setdefault(cid, []).append(p)
    return out


def choose_path(paths: Sequence[Path], role: str) -> Optional[Path]:
    if not paths:
        return None

    def score(p: Path) -> Tuple[int, int]:
        low = p.as_posix().lower()
        s = 0
        if role == "pred":
            if any(k in low for k in PRED_HINTS):
                s += 8
            if any(k in low for k in GT_HINTS):
                s -= 8
        elif role == "gt":
            if any(k in low for k in GT_HINTS):
                s += 8
            if any(k in low for k in PRED_HINTS):
                s -= 8
        ext_s = 0
        if low.endswith(".nii.gz"):
            ext_s = 3
        elif low.endswith(".nii"):
            ext_s = 2
        elif low.endswith(".npy"):
            ext_s = 1
        return (s, ext_s)

    return sorted(paths, key=score, reverse=True)[0]


def load_volume(path: Path) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]], Optional[np.ndarray]]:
    low = path.name.lower()
    if low.endswith(".npy"):
        arr = np.asarray(np.load(path))
        spacing = None
        affine = None
    elif low.endswith(".nii.gz") or low.endswith(".nii"):
        if nib is not None:
            img = nib.load(str(path))
            arr = np.asarray(img.get_fdata())
            spacing = None
            if hasattr(img, "header"):
                try:
                    z = img.header.get_zooms()[:3]
                    if len(z) == 3:
                        spacing = (float(z[0]), float(z[1]), float(z[2]))
                except Exception:
                    spacing = None
            affine = np.asarray(img.affine) if hasattr(img, "affine") else None
        elif sitk is not None:
            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img)
            sp = img.GetSpacing()
            spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
            affine = None
        else:
            raise RuntimeError("Cannot read NIfTI: nibabel and SimpleITK unavailable.")
    else:
        raise ValueError(f"Unsupported extension for {path}")

    arr = np.asarray(arr)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array after squeeze, got {arr.shape} for {path}")
    return arr.astype(np.float32), spacing, affine


def infer_pred_type(arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "binary"
    uniq = np.unique(finite)
    if uniq.size <= 2 and set(np.round(uniq).astype(int).tolist()).issubset({0, 1}):
        return "binary"
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin < 0.0 or vmax > 1.0:
        return "logit"
    return "prob"


def _safe_unique_summary(x: np.ndarray, max_items: int = 8) -> Tuple[int, str]:
    vals = np.unique(x)
    n = int(vals.size)
    if n <= max_items:
        return n, "|".join(str(float(v)) for v in vals.tolist())
    return n, ""


def _json_serializable_paths(paths: Iterable[Path], limit: int = 200) -> List[str]:
    out: List[str] = []
    for i, p in enumerate(paths):
        if i >= limit:
            break
        out.append(str(p))
    return out


def run_sanity_checks(
    pred_dir: Optional[str],
    gt_dir: Optional[str],
    output_dir: Path,
    pred_type: str = "auto",
    sample_n: int = 50,
    random_seed: int = 42,
    image_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    allow_many_empty_gt: bool = False,
) -> Dict[str, Any]:
    del image_dir  # intentionally unused for this preflight checker
    log = logger or setup_logger()
    rng = random.Random(random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment_info()
    save_json(output_dir / "environment_info.json", env)
    for w in env.get("dependency_warnings", []):
        log.warning(w)

    pred_files, pred_source = discover_pred_files(pred_dir)
    gt_root = autodetect_gt_root(gt_dir)
    gt_files = scan_files(gt_root) if gt_root is not None else []

    log.info(f"Prediction discovery source: {pred_source} | files={len(pred_files)}")
    log.info(f"GT root: {gt_root if gt_root else 'not_found'} | files={len(gt_files)}")

    pred_inventory = build_inventory(pred_files, role="pred")
    gt_inventory = build_inventory(gt_files, role="gt")
    write_rows_csv(output_dir / "file_inventory_pred.csv", pred_inventory)
    write_rows_csv(output_dir / "file_inventory_gt.csv", gt_inventory)

    pred_map = build_id_map(pred_files)
    gt_map = build_id_map(gt_files)
    pred_ids = set(pred_map.keys())
    gt_ids = set(gt_map.keys())
    matched_ids = sorted(pred_ids & gt_ids)
    missing_in_gt = sorted(pred_ids - gt_ids)
    missing_in_pred = sorted(gt_ids - pred_ids)

    pred_duplicates = {k: v for k, v in pred_map.items() if len(v) > 1}
    gt_duplicates = {k: v for k, v in gt_map.items() if len(v) > 1}

    # Hard-fail condition for suspiciously low intersection.
    min_side = min(len(pred_ids), len(gt_ids)) if pred_ids and gt_ids else 0
    if min_side <= 0:
        low_intersection_threshold = 1
    elif min_side < 20:
        low_intersection_threshold = 1
    else:
        low_intersection_threshold = max(10, int(0.10 * min_side))
    hard_fail_reasons: List[str] = []
    if min_side == 0:
        hard_fail_reasons.append("No GT or prediction IDs were discovered.")
    elif len(matched_ids) < low_intersection_threshold:
        hard_fail_reasons.append(
            f"Matched ID intersection too low: matched={len(matched_ids)} < threshold={low_intersection_threshold} "
            f"(pred_ids={len(pred_ids)}, gt_ids={len(gt_ids)})."
        )

    sample_match_ids = matched_ids.copy()
    rng.shuffle(sample_match_ids)
    sample_match_ids = sample_match_ids[: min(10, len(sample_match_ids))]
    sample_rows: List[Dict[str, Any]] = []
    for cid in sample_match_ids:
        p = choose_path(pred_map.get(cid, []), role="pred")
        g = choose_path(gt_map.get(cid, []), role="gt")
        sample_rows.append({"case_id": cid, "pred_path": str(p) if p else "", "gt_path": str(g) if g else ""})
    write_rows_csv(output_dir / "matched_pairs_sample.csv", sample_rows)

    id_report = {
        "pred_file_count": len(pred_files),
        "gt_file_count": len(gt_files),
        "pred_unique_ids": len(pred_ids),
        "gt_unique_ids": len(gt_ids),
        "intersection_count": len(matched_ids),
        "missing_in_gt_count": len(missing_in_gt),
        "missing_in_pred_count": len(missing_in_pred),
        "duplicate_pred_id_count": len(pred_duplicates),
        "duplicate_gt_id_count": len(gt_duplicates),
        "missing_in_gt_ids_sample": missing_in_gt[:200],
        "missing_in_pred_ids_sample": missing_in_pred[:200],
        "duplicate_pred_ids_sample": {
            k: _json_serializable_paths(v, limit=10) for k, v in list(pred_duplicates.items())[:50]
        },
        "duplicate_gt_ids_sample": {
            k: _json_serializable_paths(v, limit=10) for k, v in list(gt_duplicates.items())[:50]
        },
        "sample_pairs_path": str((output_dir / "matched_pairs_sample.csv").resolve()),
        "low_intersection_threshold": low_intersection_threshold,
        "pred_source": pred_source,
        "gt_root": str(gt_root) if gt_root is not None else None,
    }
    save_json(output_dir / "id_matching_report.json", id_report)

    eval_ids = matched_ids.copy()
    rng.shuffle(eval_ids)
    eval_ids = eval_ids[: min(sample_n, len(eval_ids))]

    gt_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    load_issues: List[str] = []
    pred_type_counts = {"binary": 0, "prob": 0, "logit": 0}

    for cid in eval_ids:
        pred_path = choose_path(pred_map.get(cid, []), role="pred")
        gt_path = choose_path(gt_map.get(cid, []), role="gt")
        if pred_path is None or gt_path is None:
            continue
        try:
            gt_arr, gt_spacing, _ = load_volume(gt_path)
        except Exception as e:
            load_issues.append(f"GT load failure case={cid}: {e}")
            continue
        try:
            pred_arr, pred_spacing, _ = load_volume(pred_path)
        except Exception as e:
            load_issues.append(f"Pred load failure case={cid}: {e}")
            continue

        gt_bin = (gt_arr > 0).astype(np.uint8)
        gt_fg = int(gt_bin.sum())
        gt_unique_n, gt_unique_small = _safe_unique_summary(gt_arr)
        gt_rows.append(
            {
                "case_id": cid,
                "gt_path": str(gt_path),
                "shape": str(tuple(int(x) for x in gt_arr.shape)),
                "spacing": str(gt_spacing) if gt_spacing is not None else "",
                "gt_foreground_voxels": gt_fg,
                "gt_empty": int(gt_fg == 0),
                "gt_min": float(np.min(gt_arr)),
                "gt_max": float(np.max(gt_arr)),
                "gt_unique_count": gt_unique_n,
                "gt_unique_values_if_small": gt_unique_small,
            }
        )

        pred_kind = infer_pred_type(pred_arr)
        pred_type_counts[pred_kind] = pred_type_counts.get(pred_kind, 0) + 1
        finite = pred_arr[np.isfinite(pred_arr)]
        pred_rows.append(
            {
                "case_id": cid,
                "pred_path": str(pred_path),
                "shape": str(tuple(int(x) for x in pred_arr.shape)),
                "spacing": str(pred_spacing) if pred_spacing is not None else "",
                "pred_min": float(np.min(finite)) if finite.size else float("nan"),
                "pred_max": float(np.max(finite)) if finite.size else float("nan"),
                "pred_mean": float(np.mean(finite)) if finite.size else float("nan"),
                "pred_std": float(np.std(finite)) if finite.size else float("nan"),
                "pred_inferred_type": pred_kind,
            }
        )

    write_rows_csv(output_dir / "gt_sanity_stats.csv", gt_rows)
    write_rows_csv(output_dir / "pred_sanity_stats.csv", pred_rows)

    gt_empty_fraction = (
        float(sum(int(r.get("gt_empty", 0)) for r in gt_rows) / len(gt_rows)) if gt_rows else float("nan")
    )
    gt_summary = {
        "sampled_cases": len(gt_rows),
        "gt_empty_cases": int(sum(int(r.get("gt_empty", 0)) for r in gt_rows)),
        "gt_empty_fraction": gt_empty_fraction,
        "load_issue_count": len(load_issues),
        "load_issues_sample": load_issues[:50],
        "warning_gt_empty_gt5pct": bool(np.isfinite(gt_empty_fraction) and gt_empty_fraction > 0.05),
        "failure_gt_empty_gt30pct": bool(np.isfinite(gt_empty_fraction) and gt_empty_fraction > 0.30 and not allow_many_empty_gt),
    }
    save_json(output_dir / "gt_sanity_summary.json", gt_summary)

    if gt_summary["warning_gt_empty_gt5pct"]:
        log.warning(f"GT empty fraction is high on sample: {gt_empty_fraction:.3f}")
    if gt_summary["failure_gt_empty_gt30pct"]:
        hard_fail_reasons.append(
            f"GT empty fraction {gt_empty_fraction:.3f} > 0.30. "
            "This likely indicates wrong GT masks unless dataset intentionally includes many negatives."
        )

    inferred_total = sum(pred_type_counts.values())
    top_kind = "unknown"
    top_count = 0
    if inferred_total > 0:
        top_kind, top_count = max(pred_type_counts.items(), key=lambda kv: kv[1])
    auto_choice = top_kind if inferred_total > 0 else "unknown"
    selected = pred_type if pred_type != "auto" else auto_choice
    confidence = float(top_count / inferred_total) if inferred_total > 0 else 0.0

    pred_type_report = {
        "user_pred_type": pred_type,
        "sampled_cases": inferred_total,
        "inferred_type_counts": pred_type_counts,
        "auto_selected_type": auto_choice,
        "selected_type_for_binarization": selected,
        "confidence": confidence,
        "note": (
            "confidence = fraction of sampled cases matching the dominant inferred type "
            "when user_pred_type=auto."
        ),
    }
    save_json(output_dir / "pred_type_detection.json", pred_type_report)

    if pred_type == "auto" and confidence < 0.70 and inferred_total > 0:
        log.warning(
            f"Prediction type auto-detection confidence is low ({confidence:.2f}). "
            "Consider setting --pred_type explicitly."
        )
    if pred_type != "auto" and inferred_total > 0 and pred_type_counts.get(pred_type, 0) / inferred_total < 0.5:
        log.warning(
            f"User-selected pred_type={pred_type} disagrees with sampled distributions "
            f"(counts={pred_type_counts})."
        )

    report: Dict[str, Any] = {
        "ok": len(hard_fail_reasons) == 0,
        "hard_fail_reasons": hard_fail_reasons,
        "id_matching_report": id_report,
        "gt_sanity_summary": gt_summary,
        "pred_type_detection": pred_type_report,
        "output_dir": str(output_dir.resolve()),
    }
    save_json(output_dir / "sanity_check_summary.json", report)

    if hard_fail_reasons:
        raise RuntimeError("Sanity check failed:\n- " + "\n- ".join(hard_fail_reasons))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check Phase A evaluation inputs.")
    parser.add_argument("--pred_dir", default=None, help="Directory of prediction files")
    parser.add_argument("--gt_dir", default=None, help="Directory of GT masks")
    parser.add_argument("--image_dir", default=None, help="Optional image directory (reserved for future checks)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--pred_type", default="auto", choices=["auto", "binary", "prob", "logit"])
    parser.add_argument("--sample_n", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--allow_many_empty_gt", action="store_true", help="Allow >30%% empty GT in sample without hard failure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_sanity"
    outdir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path("eval_outputs") / "verification_runs" / run_name).resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)
    report = run_sanity_checks(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        image_dir=args.image_dir,
        output_dir=outdir,
        pred_type=args.pred_type,
        sample_n=max(1, int(args.sample_n)),
        random_seed=int(args.random_seed),
        logger=logger,
        allow_many_empty_gt=bool(args.allow_many_empty_gt),
    )
    logger.info(f"Sanity check complete. ok={report.get('ok')} | output={outdir}")


if __name__ == "__main__":
    main()
