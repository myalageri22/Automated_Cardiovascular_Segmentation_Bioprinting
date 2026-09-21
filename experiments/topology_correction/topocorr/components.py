"""Stage A: connected-component characterisation of a coronary mask.

Produces a complete audit record for every component BEFORE any modification.
No anatomical assumption is made about how many components are correct: the left
and right coronary systems may legitimately form separate major components.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .io_utils import Volume


def label_components(mask: np.ndarray, structure_rank: int = 1) -> Tuple[np.ndarray, int]:
    """Label 3D connected components.

    ``structure_rank=1`` reproduces ``scipy.ndimage.generate_binary_structure(3, 1)``,
    the 6-neighbour connectivity used by ``phaseb.postprocess`` and by
    ``phaseb_mesh_qc.run_phaseb_for_case``.
    """
    structure = ndimage.generate_binary_structure(3, int(structure_rank))
    labeled, num = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labeled, int(num)


@dataclass
class ComponentRecord:
    case_id: str
    component_label: int
    volume_rank: int              # 1 == largest by physical volume
    voxel_count: int
    volume_mm3: float
    volume_fraction_of_largest: float
    volume_fraction_of_total: float
    centroid_vox_x: float
    centroid_vox_y: float
    centroid_vox_z: float
    centroid_world_x: float
    centroid_world_y: float
    centroid_world_z: float
    bbox_min_x: int
    bbox_min_y: int
    bbox_min_z: int
    bbox_max_x: int
    bbox_max_y: int
    bbox_max_z: int
    bbox_extent_x_mm: float
    bbox_extent_y_mm: float
    bbox_extent_z_mm: float
    max_extent_mm: float                     # bbox diagonal, an upper bound on length
    max_inscribed_diameter_mm: float         # 2 * max distance transform inside component
    distance_to_nearest_larger_mm: float     # inf when this is the largest component
    nearest_larger_label: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _surface_voxels(component: np.ndarray) -> np.ndarray:
    """Indices of voxels on the component boundary (erosion complement)."""
    eroded = ndimage.binary_erosion(component, ndimage.generate_binary_structure(3, 1))
    surface = component & ~eroded
    if not surface.any():
        surface = component
    return np.argwhere(surface)


def min_surface_distance_mm(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    spacing: Tuple[float, float, float],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Minimum physical distance between two voxel-index sets, with the closest pair.

    Distances are computed in millimetres by scaling indices with ``spacing``.
    """
    from scipy.spatial import cKDTree

    sp = np.asarray(spacing, dtype=float)
    a_mm = idx_a.astype(float) * sp
    b_mm = idx_b.astype(float) * sp
    tree = cKDTree(b_mm)
    dists, nearest = tree.query(a_mm, k=1)
    i = int(np.argmin(dists))
    j = int(nearest[i])
    return float(dists[i]), idx_a[i], idx_b[j]


def characterise(
    vol: Volume,
    mask: np.ndarray,
    case_id: str,
    structure_rank: int = 1,
    compute_nearest_larger: bool = True,
    max_components_for_pairwise: int = 400,
) -> List[ComponentRecord]:
    """Full Stage-A audit of every connected component in ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    labeled, num = label_components(mask, structure_rank)
    if num == 0:
        return []

    spacing = np.asarray(vol.spacing, dtype=float)
    voxel_volume = float(np.prod(spacing))

    counts = np.asarray(ndimage.sum(mask, labeled, index=range(1, num + 1)), dtype=float)
    volumes = counts * voxel_volume
    order = np.argsort(-volumes)                    # descending physical volume
    rank_of_label = {int(order[r]) + 1: r + 1 for r in range(num)}
    largest_volume = float(volumes.max())
    total_volume = float(volumes.sum())

    objects = ndimage.find_objects(labeled)
    centroids = ndimage.center_of_mass(mask, labeled, index=range(1, num + 1))

    # Distance transform per component gives the maximum inscribed radius.
    surfaces: Dict[int, np.ndarray] = {}
    records: List[ComponentRecord] = []

    for lab in range(1, num + 1):
        sl = objects[lab - 1]
        sub = labeled[sl] == lab
        offset = np.array([s.start for s in sl], dtype=int)

        dt = ndimage.distance_transform_edt(
            np.pad(sub, 1, mode="constant", constant_values=False), sampling=spacing
        )
        max_inscribed_radius = float(dt.max())

        bbox_min = offset
        bbox_max = offset + np.array(sub.shape, dtype=int) - 1
        extent_vox = np.array(sub.shape, dtype=float)
        extent_mm = extent_vox * spacing

        cen = np.asarray(centroids[lab - 1], dtype=float)
        world = vol.voxel_to_world(cen)

        if compute_nearest_larger and num <= max_components_for_pairwise:
            surfaces[lab] = _surface_voxels(sub) + offset

        records.append(
            ComponentRecord(
                case_id=str(case_id),
                component_label=int(lab),
                volume_rank=int(rank_of_label[lab]),
                voxel_count=int(counts[lab - 1]),
                volume_mm3=float(volumes[lab - 1]),
                volume_fraction_of_largest=float(volumes[lab - 1] / largest_volume) if largest_volume else 0.0,
                volume_fraction_of_total=float(volumes[lab - 1] / total_volume) if total_volume else 0.0,
                centroid_vox_x=float(cen[0]), centroid_vox_y=float(cen[1]), centroid_vox_z=float(cen[2]),
                centroid_world_x=float(world[0]), centroid_world_y=float(world[1]), centroid_world_z=float(world[2]),
                bbox_min_x=int(bbox_min[0]), bbox_min_y=int(bbox_min[1]), bbox_min_z=int(bbox_min[2]),
                bbox_max_x=int(bbox_max[0]), bbox_max_y=int(bbox_max[1]), bbox_max_z=int(bbox_max[2]),
                bbox_extent_x_mm=float(extent_mm[0]), bbox_extent_y_mm=float(extent_mm[1]),
                bbox_extent_z_mm=float(extent_mm[2]),
                max_extent_mm=float(np.linalg.norm(extent_mm)),
                max_inscribed_diameter_mm=float(2.0 * max_inscribed_radius),
                distance_to_nearest_larger_mm=float("inf"),
                nearest_larger_label=-1,
            )
        )

    if compute_nearest_larger and surfaces:
        by_rank = sorted(records, key=lambda r: r.volume_rank)
        for rec in by_rank:
            if rec.volume_rank == 1:
                continue
            larger = [r for r in by_rank if r.volume_rank < rec.volume_rank]
            best_d, best_label = float("inf"), -1
            for other in larger:
                d, _, _ = min_surface_distance_mm(
                    surfaces[rec.component_label], surfaces[other.component_label], vol.spacing
                )
                if d < best_d:
                    best_d, best_label = d, other.component_label
            rec.distance_to_nearest_larger_mm = float(best_d)
            rec.nearest_larger_label = int(best_label)

    records.sort(key=lambda r: r.volume_rank)
    return records


def summarise(records: List[ComponentRecord]) -> Dict[str, Any]:
    """Cohort-friendly per-case summary of the Stage-A audit."""
    if not records:
        return {
            "component_count": 0,
            "total_volume_mm3": 0.0,
            "largest_component_mm3": 0.0,
            "largest_component_ratio": 0.0,
            "second_component_ratio": 0.0,
            "n_components_ge_5mm3": 0,
            "n_components_lt_5mm3": 0,
            "median_component_volume_mm3": 0.0,
            "median_gap_to_larger_mm": float("nan"),
            "min_gap_to_larger_mm": float("nan"),
            "n_components_within_1p2mm": 0,
        }
    vols = np.array([r.volume_mm3 for r in records], dtype=float)
    gaps = np.array(
        [r.distance_to_nearest_larger_mm for r in records if np.isfinite(r.distance_to_nearest_larger_mm)],
        dtype=float,
    )
    return {
        "component_count": len(records),
        "total_volume_mm3": float(vols.sum()),
        "largest_component_mm3": float(vols.max()),
        "largest_component_ratio": float(vols.max() / vols.sum()) if vols.sum() else 0.0,
        "second_component_ratio": float(np.sort(vols)[-2] / vols.max()) if len(vols) > 1 else 0.0,
        "n_components_ge_5mm3": int((vols >= 5.0).sum()),
        "n_components_lt_5mm3": int((vols < 5.0).sum()),
        "median_component_volume_mm3": float(np.median(vols)),
        "median_gap_to_larger_mm": float(np.median(gaps)) if gaps.size else float("nan"),
        "min_gap_to_larger_mm": float(gaps.min()) if gaps.size else float("nan"),
        "n_components_within_1p2mm": int((gaps <= 1.2).sum()) if gaps.size else 0,
    }
