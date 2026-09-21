"""Voxel-domain segmentation fidelity and topology metrics.

clDice is delegated to the repository's existing ``compute_cldice.cldice`` so the
definition is identical to the authoritative evaluation. Everything else is
computed here in physical units using the volume spacing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .components import label_components


# ---------------------------------------------------------------------------
# clDice: import the repository implementation rather than re-deriving it.
# ---------------------------------------------------------------------------
def _load_repo_cldice():
    from .io_utils import repo_root

    path = repo_root() / "compute_cldice.py"
    spec = importlib.util.spec_from_file_location("_repo_compute_cldice", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_repo_compute_cldice"] = mod
    spec.loader.exec_module(mod)
    return mod


_CLDICE_MOD = None
CLDICE_SOURCE = "compute_cldice.cldice (repository implementation)"


def cldice(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    """(clDice, topology precision, topology sensitivity) via the repo implementation."""
    global _CLDICE_MOD
    if _CLDICE_MOD is None:
        _CLDICE_MOD = _load_repo_cldice()
    return _CLDICE_MOD.cldice(pred, gt)


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------
def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    denom = 2.0 * tp + fp + fn
    return {
        "dice": float(2.0 * tp / denom) if denom > 0 else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "jaccard": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
        "tp_voxels": int(tp),
        "fp_voxels": int(fp),
        "fn_voxels": int(fn),
    }


# ---------------------------------------------------------------------------
# Surface distances
# ---------------------------------------------------------------------------
def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, ndimage.generate_binary_structure(3, 1))
    return mask & ~eroded


def directed_surface_distances(
    a: np.ndarray, b: np.ndarray, spacing: Sequence[float]
) -> np.ndarray:
    """Distances in mm from every surface voxel of ``a`` to the nearest surface of ``b``."""
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return np.array([], dtype=float)
    dt = ndimage.distance_transform_edt(~sb, sampling=np.asarray(spacing, dtype=float))
    return dt[sa]


def surface_distance_metrics(
    pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]
) -> Dict[str, float]:
    """Symmetric surface distance summary, including HD95."""
    d_pg = directed_surface_distances(pred, gt, spacing)
    d_gp = directed_surface_distances(gt, pred, spacing)
    if d_pg.size == 0 or d_gp.size == 0:
        return {"hd95": float("nan"), "hd": float("nan"), "assd": float("nan"),
                "masd_pred_to_gt": float("nan"), "masd_gt_to_pred": float("nan")}
    both = np.concatenate([d_pg, d_gp])
    return {
        "hd95": float(np.percentile(both, 95)),
        "hd": float(both.max()),
        "assd": float(both.mean()),
        "masd_pred_to_gt": float(d_pg.mean()),
        "masd_gt_to_pred": float(d_gp.mean()),
    }


# ---------------------------------------------------------------------------
# Skeleton / centreline topology
# ---------------------------------------------------------------------------
def skeleton_stats(mask: np.ndarray, spacing: Sequence[float]) -> Dict[str, float]:
    """Skeleton size, endpoint and branch-point counts.

    Endpoints and branch points are derived from 26-neighbour degree on the
    skeleton. No branch labels are invented: the dataset carries no branch
    annotation, so these are unnamed topological counts only.
    """
    from skimage.morphology import skeletonize

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"skeleton_voxels": 0, "skeleton_length_mm_approx": 0.0,
                "skeleton_endpoints": 0, "skeleton_branchpoints": 0,
                "skeleton_components": 0}
    skel = skeletonize(mask)
    n = int(skel.sum())
    if n == 0:
        return {"skeleton_voxels": 0, "skeleton_length_mm_approx": 0.0,
                "skeleton_endpoints": 0, "skeleton_branchpoints": 0,
                "skeleton_components": 0}
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    degree = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    deg_on_skel = degree[skel]
    _, skel_components = label_components(skel, structure_rank=3)
    mean_voxel_edge = float(np.mean(np.asarray(spacing, dtype=float)))
    return {
        "skeleton_voxels": n,
        "skeleton_length_mm_approx": float(n * mean_voxel_edge),
        "skeleton_endpoints": int(np.count_nonzero(deg_on_skel == 1)),
        "skeleton_branchpoints": int(np.count_nonzero(deg_on_skel >= 3)),
        "skeleton_components": int(skel_components),
    }


def centerline_distance(
    pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]
) -> Dict[str, float]:
    """Symmetric distance between predicted and ground-truth skeletons, in mm."""
    from skimage.morphology import skeletonize

    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() or not gt.any():
        return {"centerline_mean_mm": float("nan"), "centerline_p95_mm": float("nan")}
    sp_arr = np.asarray(spacing, dtype=float)
    sk_p, sk_g = skeletonize(pred), skeletonize(gt)
    if not sk_p.any() or not sk_g.any():
        return {"centerline_mean_mm": float("nan"), "centerline_p95_mm": float("nan")}
    d_pg = ndimage.distance_transform_edt(~sk_g, sampling=sp_arr)[sk_p]
    d_gp = ndimage.distance_transform_edt(~sk_p, sampling=sp_arr)[sk_g]
    both = np.concatenate([d_pg, d_gp])
    return {
        "centerline_mean_mm": float(both.mean()),
        "centerline_p95_mm": float(np.percentile(both, 95)),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def evaluate_mask(
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    spacing: Sequence[float],
    reference_pred: Optional[np.ndarray] = None,
    structure_rank: int = 1,
    compute_hd95: bool = True,
    compute_cldice_flag: bool = True,
    compute_skeleton: bool = True,
) -> Dict[str, Any]:
    """Full fidelity + topology evaluation of one mask.

    ``reference_pred`` is the ORIGINAL uncorrected prediction; supplying it adds
    volume-difference-from-original fields. Ground truth is used for scoring only.
    """
    pred = np.asarray(pred, dtype=bool)
    sp = np.asarray(spacing, dtype=float)
    voxel_mm3 = float(np.prod(sp))
    out: Dict[str, Any] = {}

    _, n_comp = label_components(pred, structure_rank)
    out["connected_components"] = int(n_comp)
    out["pred_voxels"] = int(pred.sum())
    out["pred_volume_mm3"] = float(pred.sum() * voxel_mm3)

    if reference_pred is not None:
        ref = np.asarray(reference_pred, dtype=bool)
        ref_vol = float(ref.sum() * voxel_mm3)
        out["reference_volume_mm3"] = ref_vol
        out["volume_diff_from_original_mm3"] = out["pred_volume_mm3"] - ref_vol
        out["volume_diff_from_original_relative"] = (
            (out["pred_volume_mm3"] - ref_vol) / ref_vol if ref_vol > 0 else float("nan")
        )
        inter = float(np.count_nonzero(pred & ref))
        denom = float(pred.sum() + ref.sum())
        out["dice_vs_original_prediction"] = float(2.0 * inter / denom) if denom > 0 else 1.0

    if gt is None:
        return out

    gt = np.asarray(gt, dtype=bool)
    out.update(overlap_metrics(pred, gt))
    gt_vol = float(gt.sum() * voxel_mm3)
    out["gt_volume_mm3"] = gt_vol
    out["volume_diff_from_gt_mm3"] = out["pred_volume_mm3"] - gt_vol
    out["volume_diff_from_gt_relative"] = (
        (out["pred_volume_mm3"] - gt_vol) / gt_vol if gt_vol > 0 else float("nan")
    )

    if compute_cldice_flag:
        cl, tprec, tsens = cldice(pred, gt)
        out["cldice"] = float(cl)
        out["topology_precision"] = float(tprec)
        out["topology_sensitivity"] = float(tsens)

    if compute_hd95:
        out.update(surface_distance_metrics(pred, gt, sp))

    if compute_skeleton:
        out.update(skeleton_stats(pred, sp))
        out.update(centerline_distance(pred, gt, sp))

    return out


# ---------------------------------------------------------------------------
# Environment self-test
# ---------------------------------------------------------------------------
def verify_skeletonisation_backend() -> Dict[str, Any]:
    """Check that the installed skeletonisation backend behaves correctly.

    A topology-preserving thinning must never delete a whole connected
    component. scikit-image 0.25.2 violates this for perfectly symmetric
    even-sided solid blocks (a 10x10x10 cube thins to zero voxels), an artifact
    of tie-breaking in the 3D Lee implementation. Tube-like objects - the shape
    that actually occurs in coronary masks - are unaffected, which is why clDice
    on real data is unaffected. The check below uses a vessel-like cylinder as
    the gate and reports the symmetric-block artifact separately so that it is
    recorded rather than silently inherited.
    """
    from skimage.morphology import skeletonize

    result: Dict[str, Any] = {"backend": "skimage.morphology.skeletonize"}
    try:
        import skimage
        result["skimage_version"] = skimage.__version__
    except Exception:
        result["skimage_version"] = None

    # Vessel-like cylinder: radius 2 voxels, length 20. This is the gate.
    vol = np.zeros((40, 40, 40), dtype=bool)
    yy, zz = np.mgrid[-6:7, -6:7]
    disk = (yy ** 2 + zz ** 2) <= 4
    for x in range(10, 30):
        vol[x, 14:27, 14:27] = disk
    skel = skeletonize(vol)
    _, n_in = label_components(vol, structure_rank=3)
    _, n_skel = label_components(skel, structure_rank=3)
    result["cylinder_input_voxels"] = int(vol.sum())
    result["cylinder_skeleton_voxels"] = int(skel.sum())
    result["cylinder_components_preserved"] = bool(n_in == n_skel == 1)
    result["cylinder_length_plausible"] = bool(10 <= int(skel.sum()) <= 40)
    result["gate_passed"] = bool(result["cylinder_components_preserved"]
                                 and result["cylinder_length_plausible"])

    # Informational: the symmetric even-sided block artifact.
    cube = np.zeros((40, 40, 40), dtype=bool)
    cube[15:25, 15:25, 15:25] = True
    result["symmetric_block_skeleton_voxels"] = int(skeletonize(cube).sum())
    result["symmetric_block_artifact_present"] = bool(result["symmetric_block_skeleton_voxels"] == 0)
    if result["symmetric_block_artifact_present"]:
        result["note"] = ("Installed scikit-image empties perfectly symmetric even-sided solid "
                          "blocks. Does not affect tube-like coronary anatomy; affects synthetic "
                          "box phantoms only. See KNOWN_ISSUES.md.")
    return result
