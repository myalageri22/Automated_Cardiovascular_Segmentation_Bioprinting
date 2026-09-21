#!/usr/bin/env python3
"""Computational fabrication-readiness checks for generated STL meshes.

These checks are mesh/slicer-readiness proxies only. They are not physical
printing, perfusion, cell viability, endothelialization, or biological
validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _trimesh_import():
    try:
        import trimesh  # type: ignore

        return trimesh, None
    except Exception as exc:
        return None, str(exc)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
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
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def edge_lengths(mesh: Any) -> np.ndarray:
    if len(mesh.edges_unique) == 0:
        return np.asarray([], dtype=float)
    v = mesh.vertices
    e = mesh.edges_unique
    return np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)


def non_manifold_edges(mesh: Any) -> int:
    if len(mesh.edges_unique_inverse) == 0:
        return 0
    counts = np.bincount(mesh.edges_unique_inverse)
    return int(np.sum(counts != 2))


def validate_stl(
    stl_path: str | Path,
    case_id: Optional[str] = None,
    output_root: str | Path = "outputs/fabrication_readiness",
    target_wall_thickness_mm: float = 3.0,
    wall_thickness_compliance_fraction: Any = "not_computed",
    smoothing_volume_change_relative: Any = "not_computed",
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stl_path = Path(stl_path)
    row: Dict[str, Any] = {
        "case_id": case_id or stl_path.stem,
        "stl_path": str(stl_path),
        "stl_exists": stl_path.exists(),
        "target_wall_thickness_mm": target_wall_thickness_mm,
        "wall_thickness_compliance_fraction": wall_thickness_compliance_fraction,
        "smoothing_volume_change_relative": smoothing_volume_change_relative,
    }
    trimesh, import_error = _trimesh_import()
    if import_error:
        row.update({"status": "missing_dependency", "missing_dependency": f"trimesh: {import_error}"})
        append_csv(output_root / "per_case_fabrication_readiness.csv", row)
        return row
    try:
        mesh = trimesh.load_mesh(stl_path, force="mesh")
        lengths = edge_lengths(mesh)
        bbox = mesh.bounds if mesh.vertices.size else np.zeros((2, 3))
        extents = bbox[1] - bbox[0]
        repaired = mesh.copy()
        repair_success = True
        try:
            trimesh.repair.fix_normals(repaired)
            trimesh.repair.fill_holes(repaired)
        except Exception:
            repair_success = False
        defects = non_manifold_edges(repaired)
        row.update(
            {
                "status": "ok",
                "stl_loadable": True,
                "watertight": bool(mesh.is_watertight),
                "watertight_after_repair": bool(repaired.is_watertight),
                "non_manifold_edge_count": non_manifold_edges(mesh),
                "non_manifold_edge_count_after_repair": defects,
                "connected_component_count": int(len(mesh.split(only_watertight=False))) if len(mesh.faces) else 0,
                "mesh_volume_mm3": float(mesh.volume) if mesh.is_watertight else "not_watertight",
                "surface_area_mm2": float(mesh.area) if len(mesh.faces) else 0.0,
                "bbox_x_mm": float(extents[0]),
                "bbox_y_mm": float(extents[1]),
                "bbox_z_mm": float(extents[2]),
                "edge_length_min_mm": float(np.min(lengths)) if lengths.size else "not_computed",
                "edge_length_median_mm": float(np.median(lengths)) if lengths.size else "not_computed",
                "edge_length_mean_mm": float(np.mean(lengths)) if lengths.size else "not_computed",
                "estimated_minimum_feature_size_mm": float(np.min(lengths)) if lengths.size else "not_computed",
                "repair_success": bool(repair_success),
                "slicer_compatibility_proxy": bool(repaired.faces.size and defects == 0),
                "slicer_proxy_notes": "loadable STL and repaired mesh has no non-manifold edges" if defects == 0 else "geometry defects may break slicing",
            }
        )
    except Exception as exc:
        row.update({"status": "failed", "stl_loadable": False, "error": repr(exc)})
        append_csv(output_root / "failed_cases.csv", row)
    append_csv(output_root / "per_case_fabrication_readiness.csv", row)
    return row


def summarize(output_root: str | Path = "outputs/fabrication_readiness") -> Dict[str, Any]:
    output_root = Path(output_root)
    rows: List[Dict[str, Any]] = []
    path = output_root / "per_case_fabrication_readiness.csv"
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
        if vals:
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
    write_csv(output_root / "summary_fabrication_readiness.csv", summary_rows)
    (output_root / "summary_fabrication_readiness.json").write_text(json.dumps(summary_json, indent=2))
    tables = Path("outputs/tables_for_paper")
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "fabrication_readiness_summary.csv", summary_rows)
    (output_root / "README_fabrication_readiness.md").write_text(
        "# Computational Fabrication-Readiness Checks\n\n"
        "These outputs summarize STL loadability and mesh-quality proxies. They do not demonstrate physical bioprinting, "
        "perfusion, cell viability, endothelialization, or biological validation.\n"
    )
    return summary_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated STL fabrication-readiness proxies")
    parser.add_argument("--stl", action="append", default=[])
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--output_root", default="outputs/fabrication_readiness")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for stl in args.stl:
        validate_stl(stl, args.case_id, args.output_root)
    if args.summarize:
        summarize(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
