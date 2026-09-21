"""Geometry-preservation metrics between two reconstructed surfaces.

Used to demonstrate that any reduction in fragmentation is not bought by
distorting patient anatomy. All distances are in millimetres in the shared world
(RAS) frame that Phase B writes into the STL vertices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _load(path: str | Path):
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass
    # STL stores a face soup: every triangle carries its own copy of each vertex,
    # so a freshly loaded STL is never `is_watertight` and `mesh.volume` would be
    # NaN for every case. Merging identical vertices restores the face adjacency
    # that Phase B had in memory. It welds duplicate coordinates only and moves
    # no vertex, so the surface itself is unchanged.
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    return mesh


def _sample(mesh: Any, n_points: int, seed: int) -> np.ndarray:
    """Deterministic surface sampling. Falls back to vertices for tiny meshes."""
    verts = np.asarray(mesh.vertices, dtype=float)
    if not len(getattr(mesh, "faces", [])):
        return verts
    try:
        import trimesh

        rng = np.random.default_rng(int(seed))
        samples, _ = trimesh.sample.sample_surface(mesh, int(n_points), seed=int(seed))
        pts = np.asarray(samples, dtype=float)
        if pts.size == 0:
            return verts
        # Include vertices so extreme points are never missed by sampling.
        return np.vstack([pts, verts])
    except Exception:
        return verts


def _point_to_surface(points: np.ndarray, mesh: Any) -> "tuple[np.ndarray, str]":
    """Exact distance from each point to the nearest point ON the mesh surface.

    A point-cloud-to-point-cloud nearest-neighbour distance (the naive Chamfer)
    carries a positive floor set by the sample spacing: two IDENTICAL surfaces
    sampled independently score a non-zero deviation. That floor is of the same
    order as the sub-millimetre distortions this experiment must detect, so the
    exact point-to-triangle distance is used instead. The KD-tree form is kept
    only as a fallback and is labelled in the output when it is used.
    """
    import trimesh

    # Preferred: exact unsigned point-to-triangle distance, accelerated by rtree.
    # Works for non-watertight surfaces too, unlike signed_distance.
    try:
        _, d, _ = trimesh.proximity.closest_point(mesh, points)
        return np.asarray(d, dtype=float), "exact_point_to_surface"
    except Exception:
        pass
    # Exact but unaccelerated. Correct, just slower; used when rtree is absent.
    try:
        _, d, _ = trimesh.proximity.closest_point_naive(mesh, points)
        return np.asarray(d, dtype=float), "exact_point_to_surface_naive"
    except Exception:
        pass
    # Last resort. Carries a sampling floor and is labelled as approximate so a
    # reader can tell that these rows are not directly comparable to exact ones.
    from scipy.spatial import cKDTree

    verts = np.asarray(mesh.vertices, dtype=float)
    d, _ = cKDTree(verts).query(points, k=1)
    return np.asarray(d, dtype=float), "approximate_point_to_vertex"


def surface_deviation(
    mesh_a_path: str | Path,
    mesh_b_path: str | Path,
    n_points: int = 50000,
    seed: int = 20260826,
) -> Dict[str, float]:
    """Symmetric surface deviation of B relative to A.

    Reports Chamfer distance, symmetric mean surface distance, the 95th
    percentile symmetric distance, the Hausdorff distance, relative mesh-volume
    change, centroid displacement and per-axis bounding-box extent change.
    """
    out: Dict[str, float] = {}
    for key, path in (("a", mesh_a_path), ("b", mesh_b_path)):
        if not path or not Path(str(path)).exists():
            return {"geometry_status": f"missing_mesh_{key}"}

    a, b = _load(mesh_a_path), _load(mesh_b_path)
    if not len(getattr(a, "faces", [])) or not len(getattr(b, "faces", [])):
        return {"geometry_status": "empty_mesh"}

    pa, pb = _sample(a, n_points, seed), _sample(b, n_points, seed + 1)
    d_ab, method_ab = _point_to_surface(pa, b)
    d_ba, method_ba = _point_to_surface(pb, a)
    both = np.concatenate([d_ab, d_ba])

    out["geometry_status"] = "ok"
    out["distance_method"] = method_ab if method_ab == method_ba else f"{method_ab}|{method_ba}"
    out["n_sample_points_a"] = int(pa.shape[0])
    out["n_sample_points_b"] = int(pb.shape[0])
    out["chamfer_mm"] = float(d_ab.mean() + d_ba.mean())
    out["symmetric_mean_surface_distance_mm"] = float(both.mean())
    out["symmetric_p95_surface_distance_mm"] = float(np.percentile(both, 95))
    out["hausdorff_mm"] = float(both.max())
    out["directed_mean_a_to_b_mm"] = float(d_ab.mean())
    out["directed_mean_b_to_a_mm"] = float(d_ba.mean())

    va = float(a.volume) if a.is_watertight else float("nan")
    vb = float(b.volume) if b.is_watertight else float("nan")
    out["mesh_volume_a_mm3"] = va
    out["mesh_volume_b_mm3"] = vb
    out["mesh_volume_change_mm3"] = vb - va
    out["mesh_volume_change_relative"] = (vb - va) / va if np.isfinite(va) and va else float("nan")

    ca = np.asarray(a.vertices, dtype=float).mean(axis=0)
    cb = np.asarray(b.vertices, dtype=float).mean(axis=0)
    out["centroid_displacement_mm"] = float(np.linalg.norm(cb - ca))

    ea = a.bounds[1] - a.bounds[0]
    eb = b.bounds[1] - b.bounds[0]
    for i, ax in enumerate("xyz"):
        out[f"bbox_extent_change_{ax}_mm"] = float(eb[i] - ea[i])
    out["bbox_extent_change_norm_mm"] = float(np.linalg.norm(eb - ea))
    return out
