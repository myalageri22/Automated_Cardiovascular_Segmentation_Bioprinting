#!/usr/bin/env python3
"""Compute hard-skeleton clDice on aligned 3D prediction and label masks."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


def load_test_ids(path: Path) -> list[str]:
    split = json.loads(path.read_text())
    entries = split.get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain a test list")
    return [str(entry.get("id")) if isinstance(entry, dict) else str(entry) for entry in entries]


def skeletonize_volume(mask: np.ndarray) -> np.ndarray:
    """Return a shape-preserving boolean 3D skeleton using Lee thinning."""
    volume = np.asarray(mask, dtype=bool)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {volume.shape}")
    result = np.asarray(skeletonize(volume, method="lee"), dtype=bool)
    if result.shape != volume.shape:
        raise RuntimeError(f"Skeleton shape changed from {volume.shape} to {result.shape}")
    return result


def cldice_components(prediction: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """Return clDice, topology precision, and topology sensitivity."""
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(label, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"Masks must be aligned 3D arrays, got {pred.shape} and {gt.shape}")
    if not pred.any() and not gt.any():
        return {"cldice": 1.0, "topology_precision": 1.0, "topology_sensitivity": 1.0}
    if not pred.any() or not gt.any():
        return {"cldice": 0.0, "topology_precision": 0.0, "topology_sensitivity": 0.0}
    pred_skeleton = skeletonize_volume(pred)
    gt_skeleton = skeletonize_volume(gt)
    topology_precision = float(np.logical_and(pred_skeleton, gt).sum() / pred_skeleton.sum())
    topology_sensitivity = float(np.logical_and(gt_skeleton, pred).sum() / gt_skeleton.sum())
    denominator = topology_precision + topology_sensitivity
    cldice = 2.0 * topology_precision * topology_sensitivity / denominator if denominator else 0.0
    return {
        "cldice": float(cldice),
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
    }


def load_volume(path: Path, threshold: float | None) -> np.ndarray:
    data = np.asanyarray(nib.load(str(path)).dataobj)
    if threshold is None:
        return data > 0
    return data > threshold


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("clDice output contains non-finite values")
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "std": std,
        "median": float(np.median(array)),
        "ci95_low": float(array.mean() - margin),
        "ci95_high": float(array.mean() + margin),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute clDice at a fixed threshold for exactly aligned 3D prediction and label masks."
    )
    parser.add_argument("--split-json", type=Path, required=True, help="JSON file containing the authoritative test list")
    parser.add_argument("--prediction-pattern", required=True, help="Path pattern containing {case_id}")
    parser.add_argument("--label-pattern", required=True, help="Aligned label path pattern containing {case_id}")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed prediction threshold; no sweep is performed")
    parser.add_argument("--predictions-are-binary", action="store_true", help="Treat predictions as saved binary masks")
    args = parser.parse_args()

    if "{case_id}" not in args.prediction_pattern or "{case_id}" not in args.label_pattern:
        parser.error("Both path patterns must contain {case_id}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be within [0, 1]")
    case_ids = load_test_ids(args.split_json)
    if len(case_ids) != args.expected_cases or len(set(case_ids)) != args.expected_cases:
        raise SystemExit(f"Expected {args.expected_cases} unique test IDs, got {len(case_ids)} rows/{len(set(case_ids))} unique")

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        pred_path = Path(args.prediction_pattern.format(case_id=case_id))
        label_path = Path(args.label_pattern.format(case_id=case_id))
        if not pred_path.exists() or not label_path.exists():
            raise SystemExit(f"Missing case {case_id}: prediction={pred_path.exists()} label={label_path.exists()}")
        pred = load_volume(pred_path, None if args.predictions_are_binary else args.threshold)
        label = load_volume(label_path, None)
        components = cldice_components(pred, label)
        rows.append(
            {
                "case_id": case_id,
                "cldice@0.5": components["cldice"],
                "topology_precision": components["topology_precision"],
                "topology_sensitivity": components["topology_sensitivity"],
                "predicted_skeleton_voxels": int(skeletonize_volume(pred).sum()),
                "ground_truth_skeleton_voxels": int(skeletonize_volume(label).sum()),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize([float(row["cldice@0.5"]) for row in rows])
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps({"threshold": args.threshold, "cldice@0.5": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
