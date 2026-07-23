#!/usr/bin/env python3
"""
run_missing_checks.py

Executes the three affine-aware Phase B checks used by the corrected cohort
analysis: cross-sectional slicability, mesh-to-mask centroid displacement, and
world-coordinate bounding-box alignment.

Checks:
  1. Slicability  — 50-plane cross-sectional continuity, exactly as Section 3.7
                    Reports both the fraction of intersecting planes whose
                    contours are all closed (primary definition) and the raw
                    closed/intersecting counts needed for a strict 50/50 check.
  2. Centre of mass — world-coordinate centroid of the exported mesh versus the
                    world-coordinate centroid of the source binary mask.
  3. Spatial alignment — world-coordinate bounding box of the mesh versus the
                    world-coordinate bounding box of the mask foreground, plus a
                    spacing round-trip check.

Usage:
    python3 run_missing_checks.py \
        --stl-glob  "outputs/**/vessels_repaired.stl" \
        --mask-glob "outputs/**/seg_mask.nii.gz" \
        --out-csv   outputs/phase_b_mesh_qc/missing_checks_per_case.csv

Case IDs are extracted from the path with --case-regex (default: first 3-4 digit
run in the path). Run with --dry-run first to confirm pairing before the full
250-case pass.

Dependencies: trimesh, nibabel, numpy, pandas. No shapely/rtree needed.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import trimesh
    import nibabel as nib
except ImportError as e:
    sys.exit(f"missing dependency: {e}. pip install trimesh nibabel")


def case_id_from_path(path, pattern):
    m = re.search(pattern, path)
    return m.group(0) if m else None


def slicability_50_planes(mesh, n_planes=50, axis=None):
    """Slice along the longest bounding-box axis and score closed contours."""
    lo, hi = mesh.bounds
    extents = hi - lo
    ax = int(np.argmax(extents)) if axis is None else axis
    normal = np.zeros(3)
    normal[ax] = 1.0

    span = extents[ax]
    positions = np.linspace(lo[ax] + 0.01 * span, hi[ax] - 0.01 * span, n_planes)

    hit = 0
    closed = 0
    for p in positions:
        origin = mesh.bounds.mean(axis=0).copy()
        origin[ax] = p
        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
        except Exception:
            section = None
        if section is None:
            continue
        hit += 1
        ents = section.entities
        if len(ents) > 0 and all(bool(getattr(e, "closed", False)) for e in ents):
            closed += 1

    frac = closed / hit if hit else 0.0
    return hit, closed, frac


def mask_world_centroid(img, world_space="ras"):
    """Return the centroid and bounds in the requested RAS or LPS world space."""
    data = np.asanyarray(img.dataobj)
    idx = np.argwhere(data > 0)
    if len(idx) == 0:
        return None, None
    centroid_vox = idx.mean(axis=0)
    aff = img.affine
    centroid_world = aff[:3, :3] @ centroid_vox + aff[:3, 3]
    corners_vox = np.array([idx.min(axis=0), idx.max(axis=0)])
    corners_world = (aff[:3, :3] @ corners_vox.T).T + aff[:3, 3]
    if world_space == "lps":
        ras_to_lps = np.array([-1.0, -1.0, 1.0])
        centroid_world = centroid_world * ras_to_lps
        corners_world = corners_world * ras_to_lps
    bbox = np.array([corners_world.min(axis=0), corners_world.max(axis=0)])
    return centroid_world, bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl-glob", required=True)
    ap.add_argument("--mask-glob", required=True)
    ap.add_argument("--out-csv", default="missing_checks_per_case.csv")
    ap.add_argument("--case-regex", default=r"\d{3,4}")
    ap.add_argument("--n-planes", type=int, default=50)
    ap.add_argument(
        "--mesh-space",
        choices=("ras", "lps"),
        default="ras",
        help=(
            "world-coordinate convention used by the STL; use lps for meshes "
            "exported by this project's SimpleITK Phase B pipeline"
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the first 5 pairings and exit"
    )
    args = ap.parse_args()

    stls = sorted(glob.glob(args.stl_glob, recursive=True))
    masks = sorted(glob.glob(args.mask_glob, recursive=True))
    if not stls:
        sys.exit(f"no STL files matched {args.stl_glob}")

    mask_by_case = {}
    for p in masks:
        cid = case_id_from_path(p, args.case_regex)
        if cid:
            mask_by_case.setdefault(cid, p)

    pairs = []
    for s in stls:
        cid = case_id_from_path(s, args.case_regex)
        pairs.append((cid, s, mask_by_case.get(cid)))

    print(
        f"{len(stls)} STL files, {len(masks)} masks, "
        f"{sum(1 for _, _, m in pairs if m)} paired\n"
    )
    for cid, s, m in pairs[:5]:
        print(f"  case {cid}\n    stl  {s}\n    mask {m}")
    if args.dry_run:
        print("\ndry run — stopping here. Check the pairing, then rerun without --dry-run.")
        return

    rows = []
    for i, (cid, stl_path, mask_path) in enumerate(pairs, 1):
        row = {"case_id": cid, "stl": stl_path, "mask": mask_path}
        try:
            # STL stores independent triangles rather than shared topology.
            # Processing reconnects coincident vertices so watertightness and
            # volume centre-of-mass calculations are meaningful. It does not
            # change the mesh's coordinate frame.
            mesh = trimesh.load(stl_path, process=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

            hit, closed, frac = slicability_50_planes(mesh, args.n_planes)
            row.update(
                planes_with_intersection=hit,
                planes_all_contours_closed=closed,
                slicability_fraction=frac,
                slicability_pass=bool(frac == 1.0),
            )

            mesh_centroid = mesh.center_mass if mesh.is_watertight else mesh.centroid
            row["mesh_bbox_diagonal_mm"] = float(
                np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
            )

            if mask_path and os.path.exists(mask_path):
                img = nib.load(mask_path)
                mask_centroid, mask_bbox = mask_world_centroid(img, args.mesh_space)
                if mask_centroid is not None:
                    d = float(np.linalg.norm(np.asarray(mesh_centroid) - mask_centroid))
                    row["centroid_offset_mm"] = d
                    row["centroid_offset_pct_of_diagonal"] = (
                        100.0 * d / row["mesh_bbox_diagonal_mm"]
                    )
                    bbox_err = np.abs(np.asarray(mesh.bounds) - mask_bbox)
                    row["bbox_max_error_mm"] = float(bbox_err.max())
                    sp = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
                    row["mask_spacing_mm"] = " ".join(f"{v:.4f}" for v in sp)
                    row["alignment_pass"] = bool(
                        row["bbox_max_error_mm"] <= 2.0 * sp.max()
                    )
            row["status"] = "ok"
        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
        rows.append(row)
        if i % 25 == 0:
            print(f"  ...{i}/{len(pairs)}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    ok = df[df.status == "ok"]

    print("\n" + "=" * 70)
    print("COHORT SUMMARY")
    print("=" * 70)
    print(f"cases processed: {len(ok)}/{len(df)}")

    if "slicability_fraction" in ok:
        f = ok.slicability_fraction.dropna()
        n_full = int((f == 1.0).sum())
        print(f"\nSlicability ({args.n_planes} planes per mesh)")
        print(
            "  primary pass (closed contours on every intersecting plane): "
            f"{n_full}/{len(f)} ({100*n_full/len(f):.1f}%)"
        )
        strict = (
            (ok.planes_with_intersection == args.n_planes)
            & (ok.planes_all_contours_closed == args.n_planes)
        )
        print(
            f"  strict {args.n_planes}/{args.n_planes} pass: "
            f"{int(strict.sum())}/{len(f)} ({100*strict.mean():.1f}%)"
        )
        print(
            "  mean fraction of planes with closed contours: "
            f"{100*f.mean():.1f}%  (median {100*f.median():.1f}%)"
        )

    if "centroid_offset_mm" in ok:
        d = ok.centroid_offset_mm.dropna()
        if len(d):
            print("\nCentre-of-mass agreement (mesh versus source mask)")
            print(
                f"  mean offset {d.mean():.3f} mm, median {d.median():.3f} mm, "
                f"max {d.max():.3f} mm"
            )
            print(
                "  as a share of mesh bounding-box diagonal: mean "
                f"{ok.centroid_offset_pct_of_diagonal.mean():.3f}%"
            )

    if "bbox_max_error_mm" in ok:
        b = ok.bbox_max_error_mm.dropna()
        if len(b):
            n_pass = int(ok.alignment_pass.sum())
            print("\nSpatial alignment (world bounding box)")
            print(
                f"  within two voxels: {n_pass}/{len(b)} "
                f"({100*n_pass/len(b):.1f}%)"
            )
            print(
                f"  max bounding-box error: mean {b.mean():.3f} mm, "
                f"max {b.max():.3f} mm"
            )

    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} cases failed — see the status column in {args.out_csv}")
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
