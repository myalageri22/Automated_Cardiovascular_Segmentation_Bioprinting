"""Quality control and printability metrics."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage

try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d
except Exception:  # pragma: no cover - compatibility fallback
    _skeletonize_3d = None
try:  # skimage moved exports across versions
    from skimage.morphology import skeletonize as _skeletonize
except Exception:  # pragma: no cover
    _skeletonize = None

from . import mesh as mesh_utils
from . import postprocess
from .repair import repair_mesh


def _skeletonize_safe(mask: np.ndarray) -> np.ndarray:
    """Compatibility helper across scikit-image versions."""
    if _skeletonize_3d is not None:
        try:
            return _skeletonize_3d(mask)
        except Exception:
            pass
    if _skeletonize is not None:
        try:
            return _skeletonize(mask)
        except Exception:
            try:
                # fallback: slice-wise skeletonization
                out = np.zeros_like(mask, dtype=bool)
                for z in range(mask.shape[0]):
                    out[z] = _skeletonize(mask[z])
                return out
            except Exception:
                pass
    return np.zeros_like(mask, dtype=bool)


def mask_metrics(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Dict:
    base = postprocess.component_metrics(mask, spacing)
    base["voxel_count"] = int(mask.sum())
    base["spacing"] = tuple(float(s) for s in spacing)
    return base


def skeleton_endpoints(mask: np.ndarray) -> int:
    if mask.sum() == 0:
        return 0
    skeleton = _skeletonize_safe(mask.astype(np.uint8))
    structure = ndimage.generate_binary_structure(3, 1)
    neighbor_counts = ndimage.convolve(skeleton.astype(np.uint8), structure, mode="constant", cval=0)
    endpoints = (skeleton > 0) & (neighbor_counts == 2)  # center counted once
    return int(endpoints.sum())


def printability_radius_metrics(mask: np.ndarray, spacing: Tuple[float, float, float], min_radius_mm: float) -> Dict:
    if mask.sum() == 0:
        return {
            "min_radius_mm": None,
            "p5_radius_mm": None,
            "p50_radius_mm": None,
            "percent_below_threshold": None,
        }
    dtm = ndimage.distance_transform_edt(mask, sampling=spacing)
    radii = dtm[mask]
    metrics = {
        "min_radius_mm": float(np.min(radii)),
        "p5_radius_mm": float(np.percentile(radii, 5)),
        "p50_radius_mm": float(np.percentile(radii, 50)),
        "percent_below_threshold": float((radii < min_radius_mm).mean() * 100.0),
    }
    return metrics


def mesh_metrics(mesh: trimesh.Trimesh) -> Dict:
    bounds = mesh.bounds if mesh.vertices.size else np.zeros((2, 3))
    components = mesh.split(only_watertight=False)
    metrics = {
        "surface_area_mm2": float(mesh.area) if mesh.faces.size else 0.0,
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "bounding_box_mm": bounds.tolist(),
        "component_count": len(components),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": mesh.euler_number if mesh.faces.size else None,
    }
    return metrics


def robustness_checks(
    prob_array: Optional[np.ndarray],
    reference_image: sitk.Image,
    base_threshold: float,
    spacing: Tuple[float, float, float],
    post_cfg: Dict,
    base_volume_mm3: float,
    base_components: int,
    base_watertight: bool,
) -> Optional[Dict]:
    if prob_array is None:
        return None
    thresholds = [max(0.0, base_threshold - 0.05), min(1.0, base_threshold + 0.05)]
    results: Dict[str, Dict] = {}
    for name, thr in zip(["lower", "upper"], thresholds):
        mask = prob_array >= thr
        cleaned = postprocess.clean_mask(
            mask,
            spacing=spacing,
            closing_mm=post_cfg.get("closing_mm", 1.0),
            fill_holes_first=post_cfg.get("fill_holes", True),
            min_component_mm3=post_cfg.get("min_component_mm3", 0.0),
            mode=post_cfg.get("keep_mode", "largest"),
            seed_index_zyx=None,
        )
        comp_stats = postprocess.component_metrics(cleaned, spacing)
        try:
            mesh_obj, _, _ = mesh_utils.mask_to_mesh(cleaned, reference_image)
            repaired, _ = repair_mesh(mesh_obj, use_pymeshfix=post_cfg.get("use_pymeshfix", True))
            watertight = bool(repaired.is_watertight)
        except Exception:
            watertight = False
        results[name] = {
            "threshold": thr,
            "volume_mm3": comp_stats["volume_mm3"],
            "components": comp_stats["component_count"],
            "watertight": watertight,
        }

    def delta_pct(new: float, base: float) -> float:
        if base <= 0:
            return 0.0
        return float((new - base) / base * 100.0)

    summary = {
        "lower": results["lower"],
        "upper": results["upper"],
        "lower_delta_volume_pct": delta_pct(results["lower"]["volume_mm3"], base_volume_mm3),
        "upper_delta_volume_pct": delta_pct(results["upper"]["volume_mm3"], base_volume_mm3),
        "lower_component_delta": results["lower"]["components"] - base_components,
        "upper_component_delta": results["upper"]["components"] - base_components,
        "lower_watertight_change": results["lower"]["watertight"] != base_watertight,
        "upper_watertight_change": results["upper"]["watertight"] != base_watertight,
    }
    return summary


def assemble_qc_report(
    case_id: str,
    ct_path: str,
    seg_path: str,
    seg_type: str,
    ct_image: sitk.Image,
    probability_array: Optional[np.ndarray],
    cleaned_mask: np.ndarray,
    raw_mesh: trimesh.Trimesh,
    repaired_mesh: trimesh.Trimesh,
    threshold: float,
    post_cfg: Dict,
    min_radius_mm: float,
    sweep_results: Optional[List[Dict]],
    robustness: Optional[Dict],
    used_config: Dict,
) -> Dict:
    spacing = tuple(float(s) for s in ct_image.GetSpacing())
    ct_metadata = {
        "spacing": spacing,
        "origin": tuple(float(o) for o in ct_image.GetOrigin()),
        "direction": tuple(float(d) for d in ct_image.GetDirection()),
    }

    mask_stats = mask_metrics(cleaned_mask, spacing)
    topology_stats = {
        "component_count_after_cleanup": mask_stats.get("component_count", 0),
        "skeleton_endpoints": skeleton_endpoints(cleaned_mask),
    }

    geom_raw = mesh_metrics(raw_mesh)
    geom_repaired = mesh_metrics(repaired_mesh)

    printability = printability_radius_metrics(cleaned_mask, spacing, min_radius_mm)

    report = {
        "case_id": case_id,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "input_paths": {"ct": ct_path, "seg": seg_path},
        "segmentation": {"type": seg_type, "threshold": threshold},
        "ct_metadata": ct_metadata,
        "mask_metrics": mask_stats,
        "topology_metrics": topology_stats,
        "mesh_raw": geom_raw,
        "mesh_repaired": geom_repaired,
        "printability": printability,
        "robustness": robustness,
        "threshold_sweep": sweep_results,
        "used_config": used_config,
    }
    return report
