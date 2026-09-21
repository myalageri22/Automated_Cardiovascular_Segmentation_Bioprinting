#!/usr/bin/env python3
"""
compute_cldice.py
-----------------
Per-case clDice (Shit et al., CVPR 2021) from saved predictions vs ImageCAS ground truth.

PRIMARY ROUTE: your evaluate_full_test_a40.py already exposes `compute_cldice`. Run it with
that flag ON and you likely get clDice without this script. Use this only if the built-in
did not emit clDice, or as an independent cross-check.

What it does, per case:
  - loads the prediction mask (seg_mask.nii.gz, or seg_prob.nii.gz thresholded with --from-prob)
  - loads the ImageCAS ground-truth label and RESAMPLES it onto the prediction's voxel grid
    (nibabel.processing.resample_from_to, order=0 nearest) so both live on the same grid in
    world coordinates regardless of orientation/spacing differences
  - skeletonizes both, computes topology precision / sensitivity, and their harmonic mean

clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
  Tprec = |skel(pred) in GT volume| / |skel(pred)|   (predicted skeleton inside GT)
  Tsens = |skel(GT)   in pred volume| / |skel(GT)|    (GT skeleton inside prediction)

Dependencies: numpy, pandas, nibabel, scikit-image  (all CPU; runs fine on an M4 Pro)

Example:
  python compute_cldice.py \
      --pred-root outputs/v11_test \
      --gt-glob "Data/all/{case}/{case}.label.nii.gz" \
      --out cldice_per_case.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from skimage.morphology import skeletonize


def _cl_score(skel: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of skeleton voxels that fall inside mask."""
    s = int(skel.sum())
    if s == 0:
        return np.nan
    return float((skel & mask).sum()) / float(s)


def cldice(pred: np.ndarray, gt: np.ndarray):
    """Return (clDice, Tprec, Tsens) for two aligned binary volumes."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0
    tprec = _cl_score(skeletonize(pred), gt)
    tsens = _cl_score(skeletonize(gt), pred)
    if not np.isfinite(tprec) or not np.isfinite(tsens) or (tprec + tsens) == 0:
        return 0.0, tprec, tsens
    return float(2.0 * tprec * tsens / (tprec + tsens)), tprec, tsens


def load_pred(case_dir: str, from_prob: bool, thr: float):
    if from_prob:
        path = os.path.join(case_dir, "seg_prob.nii.gz")
        img = nib.load(path)
        arr = (np.asarray(img.dataobj) >= thr).astype(np.uint8)
    else:
        path = os.path.join(case_dir, "seg_mask.nii.gz")
        img = nib.load(path)
        arr = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    return img, arr


def gt_on_pred_grid(gt_path: str, pred_img) -> np.ndarray:
    """Resample GT label onto the prediction grid (world-coordinate aware, nearest)."""
    gt_img = nib.load(gt_path)
    matched = resample_from_to(gt_img, (pred_img.shape[:3], pred_img.affine), order=0)
    return (np.asarray(matched.dataobj) > 0).astype(np.uint8)


def resolve_cases(pred_root: str, splits_json: str | None):
    """Return list of case_ids. Prefer the test split from splits.json; else scan subdirs."""
    if splits_json and os.path.exists(splits_json):
        with open(splits_json) as f:
            splits = json.load(f)
        cases = splits.get("test") or splits.get("Test") or []
        cases = [str(c) for c in cases]
        if cases:
            return cases
    # fall back to subdirectories that actually contain a prediction
    cases = []
    for d in sorted(glob.glob(os.path.join(pred_root, "*"))):
        if os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "seg_mask.nii.gz"))
            or os.path.exists(os.path.join(d, "seg_prob.nii.gz"))
        ):
            cases.append(os.path.basename(d))
    return cases


def main():
    ap = argparse.ArgumentParser(description="Compute per-case clDice.")
    ap.add_argument("--pred-root", required=True,
                    help="Folder with per-case subdirs, each holding seg_mask.nii.gz / seg_prob.nii.gz")
    ap.add_argument("--gt-glob", required=True,
                    help="GT path template with a {case} placeholder, "
                         "e.g. 'Data/all/{case}/{case}.label.nii.gz'")
    ap.add_argument("--splits-json", default=None,
                    help="Optional splits.json; if given, uses its 'test' list for case order")
    ap.add_argument("--from-prob", action="store_true",
                    help="Threshold seg_prob.nii.gz instead of using seg_mask.nii.gz")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="cldice_per_case.csv")
    args = ap.parse_args()

    cases = resolve_cases(args.pred_root, args.splits_json)
    if not cases:
        sys.exit(f"No cases found under {args.pred_root}")

    rows = []
    for i, case in enumerate(cases, 1):
        case_dir = os.path.join(args.pred_root, case)
        gt_path = args.gt_glob.format(case=case)
        try:
            pred_img, pred = load_pred(case_dir, args.from_prob, args.threshold)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(gt_path)
            gt = gt_on_pred_grid(gt_path, pred_img)
            cl, tp, ts = cldice(pred, gt)
            rows.append({"case_id": case, "cldice": cl, "tprec": tp, "tsens": ts,
                         "pred_vox": int(pred.sum()), "gt_vox": int(gt.sum())})
            print(f"[{i}/{len(cases)}] {case}: clDice={cl:.4f}  Tprec={tp:.4f}  Tsens={ts:.4f}")
        except Exception as e:  # keep going; one bad case shouldn't kill the run
            rows.append({"case_id": case, "cldice": np.nan, "tprec": np.nan,
                         "tsens": np.nan, "pred_vox": np.nan, "gt_vox": np.nan,
                         "error": str(e)})
            print(f"[{i}/{len(cases)}] {case}: ERROR {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    valid = df["cldice"].dropna()
    print("\n=== clDice summary ===")
    print(f"cases: {len(df)}  valid: {len(valid)}")
    if len(valid):
        print(f"mean   : {valid.mean():.4f}")
        print(f"median : {valid.median():.4f}")
        print(f"std    : {valid.std(ddof=1):.4f}")
        print(f"min/max: {valid.min():.4f} / {valid.max():.4f}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
