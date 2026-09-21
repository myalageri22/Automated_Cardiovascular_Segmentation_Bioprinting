"""Deterministic topology-correction strategies.

Every strategy is a pure function of (mask, spacing, parameters). No randomness,
no per-case tuning, no use of ground truth. All physical thresholds are in mm or
mm^3 and account for the 0.6 mm isotropic representation via ``spacing``.

Strategies implemented (all prespecified in config/experiment_config.yaml):

  s0_original          identity - the control
  s1_absolute_volume   remove components below an absolute physical volume
  s2_relative_volume   remove components below a fraction of the largest component
  s3_gap_bridge        prefilter, then bridge component pairs separated by a short
                       physical distance using a minimal-radius straight connector
  s3c_closing          prefilter, then binary closing at a small physical radius
                       (reuses phaseb.postprocess.binary_closing_mm unmodified)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .components import label_components, min_surface_distance_mm, _surface_voxels

# ---------------------------------------------------------------------------
# Reuse the repository's own closing implementation so the morphological
# semantics are identical to Phase B rather than a re-implementation.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import path depends on how the repo is laid out
    from phaseb.src.phaseb.postprocess import binary_closing_mm as _repo_binary_closing_mm
    _CLOSING_SOURCE = "phaseb.src.phaseb.postprocess.binary_closing_mm"
except Exception:  # pragma: no cover
    _repo_binary_closing_mm = None
    _CLOSING_SOURCE = "local_fallback"


def binary_closing_mm(mask: np.ndarray, spacing: Sequence[float], radius_mm: float) -> np.ndarray:
    """Physical-radius binary closing, identical in behaviour to Phase B."""
    if radius_mm <= 0:
        return np.asarray(mask, dtype=bool)
    if _repo_binary_closing_mm is not None:
        return np.asarray(_repo_binary_closing_mm(np.asarray(mask, dtype=bool), tuple(spacing), float(radius_mm)),
                          dtype=bool)
    from skimage.morphology import ball

    radius_vox = max(1, int(np.ceil(float(radius_mm) / float(max(spacing)))))
    return ndimage.binary_closing(np.asarray(mask, dtype=bool), structure=ball(radius_vox))


# ---------------------------------------------------------------------------
# Component filters
# ---------------------------------------------------------------------------
def _component_volumes(mask: np.ndarray, spacing: Sequence[float], structure_rank: int = 1):
    labeled, num = label_components(mask, structure_rank)
    if num == 0:
        return labeled, num, np.array([], dtype=float)
    counts = np.asarray(ndimage.sum(mask, labeled, index=range(1, num + 1)), dtype=float)
    return labeled, num, counts * float(np.prod(np.asarray(spacing, dtype=float)))


def filter_absolute_volume(
    mask: np.ndarray,
    spacing: Sequence[float],
    min_volume_mm3: float,
    structure_rank: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove components whose physical volume is below ``min_volume_mm3``.

    The largest component is never removed, so the output is never empty for a
    non-empty input. Multiple major components are preserved by design.
    """
    mask = np.asarray(mask, dtype=bool)
    labeled, num, volumes = _component_volumes(mask, spacing, structure_rank)
    if num == 0:
        return mask, {"components_before": 0, "components_after": 0, "components_removed": 0,
                      "removed_volume_mm3": 0.0}
    keep = np.flatnonzero(volumes >= float(min_volume_mm3)) + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(volumes)) + 1])
    out = np.isin(labeled, keep)
    removed = float(volumes.sum() - volumes[keep - 1].sum())
    return out, {
        "components_before": int(num),
        "components_after": int(keep.size),
        "components_removed": int(num - keep.size),
        "removed_volume_mm3": removed,
        "min_volume_mm3": float(min_volume_mm3),
    }


def filter_relative_volume(
    mask: np.ndarray,
    spacing: Sequence[float],
    min_fraction_of_largest: float,
    structure_rank: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove components below a fraction of the largest component's volume."""
    mask = np.asarray(mask, dtype=bool)
    labeled, num, volumes = _component_volumes(mask, spacing, structure_rank)
    if num == 0:
        return mask, {"components_before": 0, "components_after": 0, "components_removed": 0,
                      "removed_volume_mm3": 0.0}
    largest = float(volumes.max())
    cutoff = largest * float(min_fraction_of_largest)
    keep = np.flatnonzero(volumes >= cutoff) + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(volumes)) + 1])
    out = np.isin(labeled, keep)
    return out, {
        "components_before": int(num),
        "components_after": int(keep.size),
        "components_removed": int(num - keep.size),
        "removed_volume_mm3": float(volumes.sum() - volumes[keep - 1].sum()),
        "min_fraction_of_largest": float(min_fraction_of_largest),
        "effective_cutoff_mm3": float(cutoff),
    }


# ---------------------------------------------------------------------------
# Short-gap reconnection
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n + 1))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def _physical_ball_offsets(spacing: Sequence[float], radius_mm: float) -> np.ndarray:
    """Voxel offsets inside a sphere of ``radius_mm``, honouring anisotropic spacing."""
    sp = np.asarray(spacing, dtype=float)
    rad_vox = np.maximum(np.floor(float(radius_mm) / sp).astype(int), 0)
    ranges = [np.arange(-r, r + 1) for r in rad_vox]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    dist = np.linalg.norm(grid.astype(float) * sp, axis=1)
    return grid[dist <= float(radius_mm) + 1e-9]


def _line_voxels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Deterministic discrete straight segment between two voxel indices."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(np.ceil(np.abs(b - a).max())) + 1
    ts = np.linspace(0.0, 1.0, max(n, 2))
    pts = np.rint(a[None, :] + ts[:, None] * (b - a)[None, :]).astype(int)
    return np.unique(pts, axis=0)


def bridge_short_gaps(
    mask: np.ndarray,
    spacing: Sequence[float],
    max_gap_mm: float,
    bridge_radius_mm: float = 0.3,
    min_component_mm3: float = 5.0,
    max_bridges: int = 50,
    structure_rank: int = 1,
    max_components_pairwise: int = 60,
    tolerance_mm: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Conservatively reconnect components separated by a short physical distance.

    Deterministic procedure:
      1. label components and discard from consideration any below
         ``min_component_mm3`` (speckle is never bridged);
      2. compute the minimum surface-to-surface distance in mm for every eligible
         pair, along with the closest voxel pair;
      3. sort candidate pairs by (gap, -smaller component volume, labels) so the
         ordering is total and independent of labelling order;
      4. walk the sorted list with a union-find, adding a bridge only when the two
         components are not already connected and the gap is within tolerance;
      5. each bridge is the straight discrete segment between the closest voxel
         pair, dilated to ``bridge_radius_mm`` in physical units.

    The default radius of 0.3 mm at 0.6 mm isotropic spacing yields a
    single-voxel-thick connector: the least invasive bridge representable on the
    grid.
    """
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    labeled, num, volumes = _component_volumes(mask, spacing, structure_rank)
    info: Dict[str, Any] = {
        "components_before": int(num),
        "bridges_added": 0,
        "bridge_voxels_added": 0,
        "bridged_gaps_mm": [],
        "eligible_components": 0,
        "candidate_pairs_evaluated": 0,
        "max_gap_mm": float(max_gap_mm),
        "bridge_radius_mm": float(bridge_radius_mm),
        "pairwise_truncated": False,
        "tolerance_mm": float(tolerance_mm),
    }
    if num <= 1:
        info["components_after"] = int(num)
        return out, info

    eligible = [int(i + 1) for i, v in enumerate(volumes) if v >= float(min_component_mm3)]
    info["eligible_components"] = len(eligible)
    if len(eligible) <= 1:
        info["components_after"] = int(num)
        return out, info
    if len(eligible) > max_components_pairwise:
        # Deterministic truncation: keep the largest components only.
        eligible = sorted(eligible, key=lambda l: (-volumes[l - 1], l))[:max_components_pairwise]
        info["pairwise_truncated"] = True

    objects = ndimage.find_objects(labeled)
    surfaces: Dict[int, np.ndarray] = {}
    for lab in eligible:
        sl = objects[lab - 1]
        sub = labeled[sl] == lab
        surfaces[lab] = _surface_voxels(sub) + np.array([s.start for s in sl], dtype=int)

    candidates: List[Tuple[float, float, int, int, np.ndarray, np.ndarray]] = []
    for ii in range(len(eligible)):
        for jj in range(ii + 1, len(eligible)):
            la, lb = eligible[ii], eligible[jj]
            gap, pa, pb = min_surface_distance_mm(surfaces[la], surfaces[lb], spacing)
            info["candidate_pairs_evaluated"] += 1
            # Tolerance guards the grid-exact case: a gap of exactly N voxels
            # evaluates to max_gap_mm +/- 1 ulp, which would otherwise make the
            # threshold silently exclusive at every round parameter value.
            if gap <= float(max_gap_mm) + float(tolerance_mm):
                smaller = float(min(volumes[la - 1], volumes[lb - 1]))
                candidates.append((float(gap), -smaller, int(la), int(lb), pa, pb))

    candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    uf = _UnionFind(num)
    offsets = _physical_ball_offsets(spacing, bridge_radius_mm)
    shape = np.array(mask.shape, dtype=int)
    added_voxels = 0

    for gap, _neg_vol, la, lb, pa, pb in candidates:
        if info["bridges_added"] >= int(max_bridges):
            break
        if not uf.union(la, lb):
            continue
        line = _line_voxels(pa, pb)
        pts = (line[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        inside = np.all((pts >= 0) & (pts < shape[None, :]), axis=1)
        pts = pts[inside]
        if pts.size == 0:
            continue
        before = int(out.sum())
        out[pts[:, 0], pts[:, 1], pts[:, 2]] = True
        added_voxels += int(out.sum()) - before
        info["bridges_added"] += 1
        info["bridged_gaps_mm"].append(float(gap))

    info["bridge_voxels_added"] = int(added_voxels)
    info["added_volume_mm3"] = float(added_voxels * np.prod(np.asarray(spacing, dtype=float)))
    _, after = label_components(out, structure_rank)
    info["components_after"] = int(after)
    return out, info


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------
def apply_strategy(
    mask: np.ndarray,
    spacing: Sequence[float],
    kind: str,
    params: Dict[str, Any],
    structure_rank: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply one named strategy. Returns (corrected_mask, provenance_info)."""
    mask = np.asarray(mask, dtype=bool)

    if kind == "identity":
        _, n = label_components(mask, structure_rank)
        return mask.copy(), {"kind": kind, "components_before": int(n), "components_after": int(n)}

    if kind == "absolute_volume_filter":
        out, info = filter_absolute_volume(mask, spacing, float(params["min_volume_mm3"]), structure_rank)
        info["kind"] = kind
        return out, info

    if kind == "relative_volume_filter":
        out, info = filter_relative_volume(mask, spacing, float(params["min_fraction_of_largest"]), structure_rank)
        info["kind"] = kind
        return out, info

    if kind == "gap_bridge":
        pre, pre_info = filter_absolute_volume(mask, spacing, float(params.get("prefilter_mm3", 0.0)), structure_rank)
        out, info = bridge_short_gaps(
            pre,
            spacing,
            max_gap_mm=float(params["max_gap_mm"]),
            bridge_radius_mm=float(params.get("bridge_radius_mm", 0.3)),
            min_component_mm3=float(params.get("min_component_mm3_to_bridge", 5.0)),
            max_bridges=int(params.get("max_bridges_per_case", 50)),
            structure_rank=structure_rank,
        )
        info["kind"] = kind
        info["prefilter"] = pre_info
        info["components_before"] = pre_info["components_before"]
        return out, info

    if kind == "morphological_closing":
        pre, pre_info = filter_absolute_volume(mask, spacing, float(params.get("prefilter_mm3", 0.0)), structure_rank)
        out = binary_closing_mm(pre, spacing, float(params["radius_mm"]))
        _, after = label_components(out, structure_rank)
        added = int(out.sum()) - int(pre.sum())
        return out, {
            "kind": kind,
            "closing_source": _CLOSING_SOURCE,
            "radius_mm": float(params["radius_mm"]),
            "prefilter": pre_info,
            "components_before": pre_info["components_before"],
            "components_after": int(after),
            "added_voxels": added,
            "added_volume_mm3": float(added * np.prod(np.asarray(spacing, dtype=float))),
        }

    raise ValueError(f"Unknown strategy kind: {kind}")


def expand_variants(cfg_strategies: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the config into the full, explicitly documented variant list.

    Every variant is reported. ``primary=True`` marks the single prespecified
    variant of each strategy used for the headline comparison; the remainder form
    the sensitivity analysis.
    """
    variants: List[Dict[str, Any]] = []
    for name, spec in cfg_strategies.items():
        if not spec.get("enabled", True):
            continue
        kind = spec["kind"]

        if kind == "identity":
            variants.append({"strategy": name, "variant_id": name, "kind": kind, "params": {}, "primary": True})

        elif kind == "absolute_volume_filter":
            for thr in spec["grid_mm3"]:
                variants.append({
                    "strategy": name,
                    "variant_id": f"{name}__mm3_{thr:g}",
                    "kind": kind,
                    "params": {"min_volume_mm3": float(thr)},
                    "primary": float(thr) == float(spec["primary_threshold_mm3"]),
                })

        elif kind == "relative_volume_filter":
            for frac in spec["grid_fraction"]:
                variants.append({
                    "strategy": name,
                    "variant_id": f"{name}__frac_{frac:g}",
                    "kind": kind,
                    "params": {"min_fraction_of_largest": float(frac)},
                    "primary": float(frac) == float(spec["primary_fraction"]),
                })

        elif kind == "gap_bridge":
            for gap in spec["grid_max_gap_mm"]:
                variants.append({
                    "strategy": name,
                    "variant_id": f"{name}__gap_{gap:g}mm",
                    "kind": kind,
                    "params": {
                        "max_gap_mm": float(gap),
                        "prefilter_mm3": float(spec.get("prefilter_mm3", 0.0)),
                        "bridge_radius_mm": float(spec.get("bridge_radius_mm", 0.3)),
                        "min_component_mm3_to_bridge": float(spec.get("min_component_mm3_to_bridge", 5.0)),
                        "max_bridges_per_case": int(spec.get("max_bridges_per_case", 50)),
                    },
                    "primary": float(gap) == float(spec["primary_max_gap_mm"]),
                })

        elif kind == "morphological_closing":
            for rad in spec["grid_radius_mm"]:
                variants.append({
                    "strategy": name,
                    "variant_id": f"{name}__r_{rad:g}mm",
                    "kind": kind,
                    "params": {"radius_mm": float(rad), "prefilter_mm3": float(spec.get("prefilter_mm3", 0.0))},
                    "primary": float(rad) == float(spec["primary_radius_mm"]),
                })
        else:
            raise ValueError(f"Unknown strategy kind in config: {kind}")
    return variants
