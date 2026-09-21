"""Experiment orchestration: per-case execution and cohort assembly."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import components as comp_mod
from . import geometry as geom_mod
from . import mesh_metrics as mesh_mod
from . import seg_metrics as seg_mod
from . import strategies as strat_mod
from .io_utils import (Volume, ensure_dir, load_binary_volume, resample_gt_to,
                       save_binary_volume, sha256_file)

STRENGTH_KEY = {
    "absolute_volume_filter": "min_volume_mm3",
    "relative_volume_filter": "min_fraction_of_largest",
    "gap_bridge": "max_gap_mm",
    "morphological_closing": "radius_mm",
}


def _first_existing(root: Path, candidates: List[Any]) -> Tuple[Optional[Path], Optional[str]]:
    """Return the first candidate path that exists, with the source that supplied it."""
    for source, cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        p = p if p.is_absolute() else (root / p)
        if "*" in str(p):
            matches = sorted(root.glob(str(p.relative_to(root))))
            if matches:
                return matches[0], source
            continue
        if p.exists():
            return p, source
    return None, None


def resolve_case_inputs(
    cfg: Dict[str, Any],
    root: Path,
    case: str,
    split_entry: Optional[Dict[str, Any]] = None,
    authoritative: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Locate the frozen prediction, ground truth and CTA for one case.

    The ImageCAS layout is not consistent across the repository's own records:
    ``dataset_splits.json`` stores ``Data/all/1000.label.nii.gz`` while
    ``outputs/final_test_250/per_case_metrics.csv`` - written by the run that
    produced the reported numbers - stores ``Data/all/801-1000/1000.label.nii.gz``.
    Candidates are therefore tried in order of authority, and the source that
    resolved each path is recorded so the manifest shows which layout was used.
    """
    paths = cfg["paths"]
    pred_root = str((root / paths["pred_root"]) if not Path(paths["pred_root"]).is_absolute()
                    else Path(paths["pred_root"]))
    mask = Path(paths["pred_mask_template"].format(pred_root=pred_root, case=case))
    prob = Path(paths["pred_prob_template"].format(pred_root=pred_root, case=case))

    auth_row = None
    if authoritative is not None:
        m = authoritative[authoritative["case_id"].astype(str) == str(case)]
        if not m.empty:
            auth_row = m.iloc[0]

    gt_candidates = [
        ("authoritative_csv", auth_row["label"] if auth_row is not None and "label" in auth_row else None),
        ("config_template", paths["gt_template"].format(case=case)),
        ("split_entry", (split_entry or {}).get("label")),
        ("glob_range_subdir", f"Data/all/*/{case}.label.nii.gz"),
    ]
    ct_candidates = [
        ("authoritative_csv", auth_row["image"] if auth_row is not None and "image" in auth_row else None),
        ("config_template", paths["ct_template"].format(case=case)),
        ("split_entry", (split_entry or {}).get("image")),
        ("glob_range_subdir", f"Data/all/*/{case}.img.nii.gz"),
    ]
    gt, gt_source = _first_existing(root, gt_candidates)
    ct, ct_source = _first_existing(root, ct_candidates)

    use_prob = cfg["segmentation"].get("binarise_from", "mask") == "prob"
    pred_path = prob if use_prob else mask

    missing = []
    if not pred_path.exists():
        missing.append("prediction")
    if gt is None:
        missing.append("ground_truth")

    rec: Dict[str, Any] = {
        "case_id": case,
        "prediction_path": str(pred_path),
        "prediction_kind": "probability" if use_prob else "binary_mask",
        "ground_truth_path": str(gt) if gt else None,
        "ground_truth_source": gt_source,
        "ct_path": str(ct) if ct else None,
        "ct_source": ct_source,
        "candidates_tried": {"ground_truth": [c for _, c in gt_candidates if c],
                             "ct": [c for _, c in ct_candidates if c]},
        "complete": not missing,
        "missing": missing,
    }
    if not missing:
        rec["prediction_sha256"] = sha256_file(pred_path)
        rec["ground_truth_sha256"] = sha256_file(gt)  # type: ignore[arg-type]
    return rec


def check_provenance(
    case: str, dice_recomputed: float, authoritative: Optional[pd.DataFrame], tol: float
) -> Dict[str, Any]:
    """Verify the loaded prediction is the one that produced the archived metrics."""
    if authoritative is None:
        return {"provenance_checked": False, "provenance_ok": None,
                "authoritative_dice": np.nan, "dice_abs_delta": np.nan}
    row = authoritative[authoritative["case_id"].astype(str) == str(case)]
    if row.empty:
        return {"provenance_checked": True, "provenance_ok": False,
                "authoritative_dice": np.nan, "dice_abs_delta": np.nan,
                "provenance_note": "case absent from authoritative per_case_metrics.csv"}
    ref = float(pd.to_numeric(row.iloc[0]["dice@0.5"], errors="coerce"))
    delta = abs(dice_recomputed - ref)
    return {"provenance_checked": True, "provenance_ok": bool(delta <= tol),
            "authoritative_dice": ref, "dice_abs_delta": float(delta)}


def run_case(
    cfg: Dict[str, Any],
    root: Path,
    case: str,
    inputs: Dict[str, Any],
    variants: List[Dict[str, Any]],
    out_root: Path,
    authoritative: Optional[pd.DataFrame] = None,
    do_mesh: bool = True,
    do_geometry: bool = True,
    mesh_variants: str = "primary",
    logger=None,
) -> Dict[str, Any]:
    """Execute every strategy variant for one case. Returns per-case records."""
    result: Dict[str, Any] = {"case_id": case, "status": "ok", "rows": [], "components": [], "error": None}
    seed = int(cfg["experiment"]["seed"])
    rank = int(cfg["connectivity"]["structure_rank"])
    mcfg = cfg["metrics"]
    sec = cfg["segmentation"]

    try:
        threshold = float(sec["threshold"]) if inputs["prediction_kind"] == "probability" else None
        vol = load_binary_volume(inputs["prediction_path"], threshold=threshold)
        original = vol.array

        expected = np.asarray(sec.get("expected_spacing_mm", vol.spacing), dtype=float)
        spacing_ok = bool(np.all(np.abs(np.asarray(vol.spacing) - expected)
                                 <= float(sec.get("spacing_tolerance_mm", 0.05))))

        gt = resample_gt_to(inputs["prediction_path"], inputs["ground_truth_path"])

        # ---------------- Stage A: characterise before any modification -------
        records = comp_mod.characterise(vol, original, case, structure_rank=rank)
        result["components"] = [r.as_dict() for r in records]
        stage_a = comp_mod.summarise(records)

        case_dir = ensure_dir(out_root / "case_outputs" / str(case))
        gt_mesh_path: Optional[str] = None
        original_mesh_path: Optional[str] = None
        base_metrics: Optional[Dict[str, Any]] = None

        # Ground-truth mesh, built with the identical procedure, for EVALUATION ONLY.
        if do_mesh and do_geometry and mcfg["geometry"].get("build_ground_truth_meshes", False):
            gt_mask_path = case_dir / "ground_truth_mask.nii.gz"
            save_binary_volume(vol, gt, gt_mask_path)
            gt_row = mesh_mod.reconstruct(
                f"{case}__groundtruth", gt_mask_path, out_root / "mesh_qc" / "ground_truth",
                target_wall_thickness_mm=float(mcfg["mesh"]["target_wall_thickness_mm"]),
                minimal=bool(mcfg["mesh"]["minimal"]),
                slicability_planes=int(mcfg["mesh"]["slicability_planes"]),
            )
            gt_mesh_path = gt_row.get("repaired_stl") if gt_row.get("status") == "ok" else None

        for variant in variants:
            row: Dict[str, Any] = {
                "case_id": case,
                "strategy": variant["strategy"],
                "variant_id": variant["variant_id"],
                "kind": variant["kind"],
                "primary_variant": bool(variant["primary"]),
                "spacing_x_mm": vol.spacing[0], "spacing_y_mm": vol.spacing[1],
                "spacing_z_mm": vol.spacing[2], "spacing_as_expected": spacing_ok,
                "segmentation_threshold_frozen": sec["threshold"],
            }
            row["strength"] = variant["params"].get(STRENGTH_KEY.get(variant["kind"], ""), np.nan)
            row.update({f"stage_a_{k}": v for k, v in stage_a.items()})

            corrected, info = strat_mod.apply_strategy(
                original, vol.spacing, variant["kind"], variant["params"], structure_rank=rank
            )
            row["correction_components_before"] = info.get("components_before")
            row["correction_components_after"] = info.get("components_after")
            row["correction_bridges_added"] = info.get("bridges_added", 0)
            row["correction_added_volume_mm3"] = info.get("added_volume_mm3", 0.0)
            row["correction_removed_volume_mm3"] = info.get("removed_volume_mm3", 0.0)

            metrics = seg_mod.evaluate_mask(
                corrected, gt, vol.spacing,
                reference_pred=None if variant["kind"] == "identity" else original,
                structure_rank=rank,
                compute_hd95=bool(mcfg["segmentation"]["compute_hd95"]),
                compute_cldice_flag=bool(mcfg["segmentation"]["compute_cldice"]),
                compute_skeleton=bool(mcfg["segmentation"]["compute_skeleton_stats"]),
            )
            row.update(metrics)

            if variant["kind"] == "identity":
                base_metrics = metrics
                row.update(check_provenance(case, float(metrics.get("dice", np.nan)),
                                            authoritative,
                                            float(cfg["provenance"]["dice_abs_tolerance"])))

            # ---------------- Phase B reconstruction --------------------------
            run_mesh = do_mesh and (mesh_variants == "all" or variant["primary"])
            row["mesh_attempted"] = bool(run_mesh)
            if run_mesh:
                mask_path = case_dir / f"mask__{variant['variant_id']}.nii.gz"
                save_binary_volume(vol, corrected, mask_path)
                row["corrected_mask_path"] = str(mask_path)
                qc = mesh_mod.reconstruct(
                    f"{case}__{variant['variant_id']}", mask_path,
                    out_root / "mesh_qc" / variant["variant_id"],
                    target_wall_thickness_mm=float(mcfg["mesh"]["target_wall_thickness_mm"]),
                    minimal=bool(mcfg["mesh"]["minimal"]),
                    slicability_planes=int(mcfg["mesh"]["slicability_planes"]),
                )
                for k, v in qc.items():
                    if k in ("case_id",):
                        continue
                    row[f"mesh_{k}"] = v
                row["mesh_integrity_pass"] = mesh_mod.mesh_integrity_pass(qc)
                stl = qc.get("repaired_stl") if qc.get("status") == "ok" else None
                if variant["kind"] == "identity":
                    original_mesh_path = stl
                row["repaired_stl"] = stl

                # ---------------- Geometry preservation -----------------------
                if do_geometry and stl and original_mesh_path and variant["kind"] != "identity":
                    dev = geom_mod.surface_deviation(
                        original_mesh_path, stl,
                        n_points=int(mcfg["geometry"]["surface_sample_points"]), seed=seed)
                    row.update({f"geom_vs_original_{k}": v for k, v in dev.items()})
                if do_geometry and stl and gt_mesh_path:
                    devg = geom_mod.surface_deviation(
                        gt_mesh_path, stl,
                        n_points=int(mcfg["geometry"]["surface_sample_points"]), seed=seed)
                    row.update({f"geom_vs_gt_{k}": v for k, v in devg.items()})

            result["rows"].append(row)

    except Exception as exc:  # one bad case must not kill the cohort
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        if logger:
            logger.error("case %s failed: %s", case, result["error"])
    return result


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------
def build_paired_table(per_case: pd.DataFrame) -> pd.DataFrame:
    """Join every corrected variant against the S0 control, paired by case."""
    control = per_case[per_case["kind"] == "identity"].copy()
    corrected = per_case[per_case["kind"] != "identity"].copy()
    if control.empty or corrected.empty:
        return pd.DataFrame()

    carry = ["dice", "cldice", "hd95", "connected_components", "precision", "recall",
             "pred_volume_mm3", "assd", "centerline_mean_mm", "skeleton_endpoints",
             "skeleton_branchpoints", "mesh_integrity_pass", "topology_precision",
             "topology_sensitivity", "gt_volume_mm3", "mesh_slicability_closed_fraction",
             "mesh_to_mask_centroid_mm", "mesh_mesh_to_mask_centroid_mm",
             "mesh_non_manifold_edge_count", "mesh_watertight", "mesh_volume_mm3"]
    carry = [c for c in carry if c in control.columns]
    ctrl = control[["case_id"] + carry].rename(columns={c: f"{c}_original" for c in carry})

    merged = corrected.merge(ctrl, on="case_id", how="inner", validate="many_to_one")
    rename = {c: f"{c}_corrected" for c in carry if c in merged.columns}
    merged = merged.rename(columns=rename)

    merged["components_original"] = merged.get("connected_components_original")
    merged["components_corrected"] = merged.get("connected_components_corrected")
    for base in ("dice", "cldice", "hd95", "assd", "precision", "recall",
                 "pred_volume_mm3", "centerline_mean_mm"):
        o, c = f"{base}_original", f"{base}_corrected"
        if o in merged.columns and c in merged.columns:
            merged[f"delta_{base}"] = (pd.to_numeric(merged[c], errors="coerce")
                                       - pd.to_numeric(merged[o], errors="coerce"))
    if "components_original" in merged.columns:
        merged["delta_components"] = (pd.to_numeric(merged["components_corrected"], errors="coerce")
                                      - pd.to_numeric(merged["components_original"], errors="coerce"))
    return merged


def cohort_summary(paired: pd.DataFrame, per_case: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy variant with cohort-level magnitudes."""
    if paired.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for vid, sub in paired.groupby("variant_id", sort=True):
        r: Dict[str, Any] = {
            "variant_id": vid,
            "strategy": sub["strategy"].iloc[0],
            "kind": sub["kind"].iloc[0],
            "primary_variant": bool(sub["primary_variant"].iloc[0]),
            "strength": pd.to_numeric(sub["strength"], errors="coerce").iloc[0],
            "n_cases": int(len(sub)),
        }
        for base in ("dice", "cldice", "hd95", "components", "assd", "centerline_mean_mm"):
            o, c, d = f"{base}_original", f"{base}_corrected", f"delta_{base}"
            if o in sub.columns:
                r[f"{base}_original_mean"] = float(pd.to_numeric(sub[o], errors="coerce").mean())
            if c in sub.columns:
                r[f"{base}_corrected_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
                r[f"{base}_corrected_median"] = float(pd.to_numeric(sub[c], errors="coerce").median())
            if d in sub.columns:
                dd = pd.to_numeric(sub[d], errors="coerce")
                r[f"{d}_mean"] = float(dd.mean())
                r[f"{d}_median"] = float(dd.median())
        for col, out in (("geom_vs_original_symmetric_mean_surface_distance_mm", "surface_deviation_mean_mm"),
                         ("geom_vs_original_symmetric_p95_surface_distance_mm", "surface_deviation_p95_mm"),
                         ("geom_vs_original_hausdorff_mm", "surface_deviation_hausdorff_mm"),
                         ("geom_vs_original_chamfer_mm", "chamfer_mm"),
                         ("geom_vs_original_centroid_displacement_mm", "centroid_displacement_mm"),
                         ("geom_vs_original_mesh_volume_change_relative", "mesh_volume_change_relative"),
                         ("geom_vs_original_bbox_extent_change_norm_mm", "bbox_extent_change_mm")):
            if col in sub.columns:
                r[out] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        for col, out in (("mesh_integrity_pass_original", "mesh_integrity_rate_original"),
                         ("mesh_integrity_pass_corrected", "mesh_integrity_rate_corrected")):
            if col in sub.columns:
                vals = sub[col].map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})
                r[out] = float(vals.mean())
                r[out.replace("rate", "count")] = int(vals.sum())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["strategy", "strength"], na_position="first")
