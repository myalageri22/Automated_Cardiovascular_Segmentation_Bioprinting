#!/usr/bin/env python3
"""Cohort mesh-QC hooks for Phase B fabrication-oriented evidence.

The module computes computational mesh quality signals from segmentation masks.
It does not claim physical printing or biological validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _optional_imports() -> Dict[str, Any]:
    imports: Dict[str, Any] = {}
    for name in ("nibabel", "scipy.ndimage", "skimage.measure", "trimesh"):
        try:
            if name == "nibabel":
                import nibabel as nib  # type: ignore

                imports["nib"] = nib
            elif name == "scipy.ndimage":
                from scipy import ndimage as ndi  # type: ignore

                imports["ndi"] = ndi
            elif name == "skimage.measure":
                from skimage import measure  # type: ignore

                imports["measure"] = measure
            elif name == "trimesh":
                import trimesh  # type: ignore

                imports["trimesh"] = trimesh
        except Exception as exc:
            imports[f"missing_{name}"] = str(exc)
    return imports


def append_csv(path: Path, row: Dict[str, Any]) -> None:
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mesh_edge_lengths(mesh: Any) -> np.ndarray:
    if mesh.edges_unique is None or len(mesh.edges_unique) == 0:
        return np.asarray([], dtype=float)
    vertices = mesh.vertices
    edges = mesh.edges_unique
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def non_manifold_edge_count(mesh: Any) -> int:
    if len(mesh.edges_unique_inverse) == 0:
        return 0
    counts = np.bincount(mesh.edges_unique_inverse)
    return int(np.sum(counts != 2))


def _mesh_summary(mesh: Any, prefix: str = "") -> Dict[str, Any]:
    lengths = mesh_edge_lengths(mesh)
    bounds = mesh.bounds if getattr(mesh, "vertices", np.asarray([])).size else np.zeros((2, 3))
    extents = bounds[1] - bounds[0]
    return {
        f"{prefix}watertight": bool(mesh.is_watertight),
        f"{prefix}non_manifold_edge_count": non_manifold_edge_count(mesh),
        f"{prefix}connected_component_count": int(len(mesh.split(only_watertight=False))) if len(mesh.faces) else 0,
        f"{prefix}surface_area_mm2": float(mesh.area) if len(mesh.faces) else 0.0,
        f"{prefix}volume_mm3": float(mesh.volume) if mesh.is_watertight else "not_watertight",
        f"{prefix}bbox_x_mm": float(extents[0]) if len(extents) else 0.0,
        f"{prefix}bbox_y_mm": float(extents[1]) if len(extents) else 0.0,
        f"{prefix}bbox_z_mm": float(extents[2]) if len(extents) else 0.0,
        f"{prefix}edge_length_min_mm": float(np.min(lengths)) if lengths.size else "missing_dependency",
        f"{prefix}edge_length_median_mm": float(np.median(lengths)) if lengths.size else "missing_dependency",
        f"{prefix}edge_length_mean_mm": float(np.mean(lengths)) if lengths.size else "missing_dependency",
    }


def _roughness_proxy(mesh: Any) -> Dict[str, float]:
    try:
        normals = mesh.face_normals
        if len(normals) < 2:
            return {"mean": 0.0, "max": 0.0}
        dots = np.clip(np.sum(normals[:-1] * normals[1:], axis=1), -1.0, 1.0)
        angles = np.arccos(dots)
        return {"mean": float(np.mean(angles)), "max": float(np.max(angles))}
    except Exception:
        return {"mean": float("nan"), "max": float("nan")}


def run_phaseb_for_case(
    case_id: str,
    mask_path: str | Path,
    output_root: str | Path = "outputs/phase_b_mesh_qc",
    target_wall_thickness_mm: float = 3.0,
    minimal: bool = False,
) -> Dict[str, Any]:
    imports = _optional_imports()
    output_root = Path(output_root)
    case_dir = output_root / "case_outputs" / str(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    row: Dict[str, Any] = {
        "case_id": case_id,
        "mask_path": str(mask_path),
        "status": "ok",
        "target_wall_thickness_mm": target_wall_thickness_mm,
    }
    missing = {k: v for k, v in imports.items() if k.startswith("missing_")}
    if missing:
        row.update({"status": "missing_dependency", "missing_dependency": json.dumps(missing, sort_keys=True)})
        append_csv(output_root / "per_case_mesh_qc.csv", row)
        return row
    nib = imports["nib"]
    ndi = imports["ndi"]
    measure = imports["measure"]
    trimesh = imports["trimesh"]
    try:
        img = nib.load(str(mask_path))
        arr = np.asarray(img.get_fdata() > 0, dtype=np.uint8)
        spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
        labels, n_comp = ndi.label(arr)
        row["connected_component_count"] = int(n_comp)
        if arr.sum() == 0:
            raise ValueError("empty mask")
        # marching_cubes returns voxel indices in nibabel's x,y,z array order.
        # Apply the complete affine so the STL retains the mask's RAS world
        # frame, including origin, orientation/reflection, and voxel spacing.
        verts_vox, faces, _, _ = measure.marching_cubes(
            arr.astype(np.float32), level=0.5
        )
        verts_world = nib.affines.apply_affine(img.affine, verts_vox)
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
        raw_path = case_dir / "segmentation_raw.stl"
        if not minimal:
            mesh.export(raw_path)
        rough_pre = _roughness_proxy(mesh) if not minimal else None
        repaired = mesh.copy()
        repair_success = True
        try:
            trimesh.repair.fix_normals(repaired)
            trimesh.repair.fill_holes(repaired)
        except Exception:
            repair_success = False
        repaired_path = case_dir / "segmentation_repaired.stl"
        repaired.export(repaired_path)
        row.update(_mesh_summary(repaired))
        if minimal:
            row.update(
                {
                    "raw_stl": "not_exported_minimal_qc",
                    "repaired_stl": str(repaired_path),
                    "smoothed_stl": "not_computed_minimal_qc",
                    "repair_success": bool(repair_success),
                    "smoothing_volume_change_relative": "not_computed_minimal_qc",
                    "wall_thickness_compliance_fraction": "not_computed_minimal_qc",
                    "slicer_compatibility_proxy": bool(
                        repaired.faces.size and non_manifold_edge_count(repaired) == 0
                    ),
                }
            )
        else:
            smoothed = repaired.copy()
            try:
                trimesh.smoothing.filter_taubin(
                    smoothed, lamb=0.5, nu=0.5, iterations=10
                )
            except Exception:
                pass
            smooth_path = case_dir / "segmentation_smoothed.stl"
            smoothed.export(smooth_path)
            rough_post = _roughness_proxy(smoothed)
            volume_pre = float(repaired.volume) if repaired.is_watertight else np.nan
            volume_post = float(smoothed.volume) if smoothed.is_watertight else np.nan
            dt = ndi.distance_transform_edt(arr, sampling=spacing)
            diameters = dt[arr > 0] * 2.0
            compliance = (
                float(np.mean(diameters >= target_wall_thickness_mm))
                if diameters.size
                else "missing_dependency"
            )
            row.update(
                {
                    "raw_stl": str(raw_path),
                    "repaired_stl": str(repaired_path),
                    "smoothed_stl": str(smooth_path),
                    "repair_success": bool(repair_success),
                    "surface_roughness_pre_mean": rough_pre["mean"],
                    "surface_roughness_pre_max": rough_pre["max"],
                    "surface_roughness_post_mean": rough_post["mean"],
                    "surface_roughness_post_max": rough_post["max"],
                    "smoothing_volume_change_relative": (
                        float((volume_post - volume_pre) / volume_pre)
                        if np.isfinite(volume_pre) and volume_pre
                        else "not_watertight"
                    ),
                    "wall_thickness_compliance_fraction": compliance,
                    "slicer_compatibility_proxy": bool(
                        repaired.faces.size and non_manifold_edge_count(repaired) == 0
                    ),
                }
            )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        append_csv(output_root / "failed_cases.csv", row)
    append_csv(output_root / "per_case_mesh_qc.csv", row)
    return row


def summarize_mesh_qc(output_root: str | Path = "outputs/phase_b_mesh_qc") -> Dict[str, Any]:
    output_root = Path(output_root)
    rows: List[Dict[str, Any]] = []
    path = output_root / "per_case_mesh_qc.csv"
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    numeric_keys = set()
    for row in rows:
        for key, value in row.items():
            try:
                float(value)
                numeric_keys.add(key)
            except Exception:
                pass
    summary_rows = []
    summary_json: Dict[str, Any] = {"n_cases": len(rows), "metrics": {}}
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
        stats = {
            "metric": key,
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.nanmedian(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
        summary_rows.append(stats)
        summary_json["metrics"][key] = {k: v for k, v in stats.items() if k != "metric"}
    write_csv(output_root / "summary_mesh_qc.csv", summary_rows)
    (output_root / "summary_mesh_qc.json").write_text(json.dumps(summary_json, indent=2))
    linkage = [
        {
            "case_id": row.get("case_id"),
            "repaired_stl": row.get("repaired_stl"),
            "mesh_qc_status": row.get("status"),
            "watertight": row.get("watertight"),
            "non_manifold_edge_count": row.get("non_manifold_edge_count"),
        }
        for row in rows
    ]
    write_csv(output_root / "segmentation_mesh_linkage.csv", linkage)
    (output_root / "segmentation_mesh_correlation_summary.json").write_text(
        json.dumps({"n_cases": len(rows), "note": "Correlation plots are generated by generate_reviewer_outputs.py when segmentation metrics are available."}, indent=2)
    )
    return summary_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal Phase B mesh QC from mask files")
    parser.add_argument("--mask", help="Single mask NIfTI")
    parser.add_argument("--case_id", default="case")
    parser.add_argument("--output_root", default="outputs/phase_b_mesh_qc")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mask:
        run_phaseb_for_case(args.case_id, args.mask, args.output_root)
    if args.summarize:
        summarize_mesh_qc(args.output_root)
    return 0
def slicability_plane_check(mesh, n_planes: int = 50, inset_frac: float = 0.02):
    """
    Multi-plane cross-sectional continuity (slicer-readiness proxy).
    Slices `mesh` with `n_planes` evenly spaced planes along its longest
    bounding-box axis; a slice passes if its cross-section is composed
    entirely of closed contours. Reports per-case QC fields.
    """
    import numpy as np
    out = {
        "slicability_planes_total": int(n_planes),
        "slicability_planes_intersecting": 0,
        "slicability_planes_closed": 0,
        "slicability_closed_fraction": float("nan"),
        "slicable": False,
    }
    if mesh is None or getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        return out
    extents = mesh.bounds[1] - mesh.bounds[0]
    axis = int(np.argmax(extents))                      # 50 planes ⟂ longest axis
    normal = np.zeros(3); normal[axis] = 1.0
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    span = hi - lo
    if span <= 0:
        return out
    lo += inset_frac * span; hi -= inset_frac * span    # inset off the caps
    inter = closed = 0
    for level in np.linspace(lo, hi, n_planes):
        origin = mesh.bounds[0].copy(); origin[axis] = level
        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
        except Exception:
            continue
        if section is None:
            continue
        inter += 1
        try:
            planar = section.to_2D()[0] if hasattr(section, "to_2D") else section.to_planar()[0]
        except Exception:
            continue                                    # exists but won't close → fail
        ents = getattr(planar, "entities", [])
        polys = getattr(planar, "polygons_full", [])
        all_closed = len(ents) > 0 and all(bool(getattr(e, "closed", False)) for e in ents)
        if len(polys) > 0 and all_closed:
            closed += 1
    out["slicability_planes_intersecting"] = int(inter)
    out["slicability_planes_closed"] = int(closed)
    if inter:
        out["slicability_closed_fraction"] = float(closed / inter)
        out["slicable"] = bool(closed == inter)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
