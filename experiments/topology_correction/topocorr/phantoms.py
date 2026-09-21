"""Synthetic vessel-tree phantoms.

These exist to validate the experiment code end to end WITHOUT any cohort data.
Phantom results are never mixed with, or reported as, held-out test results.

Each phantom provides a ground truth tree and a degraded "prediction" with a
known, controlled defect, so the expected behaviour of every strategy is known
in advance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Phantom:
    case_id: str
    ground_truth: np.ndarray
    prediction: np.ndarray
    spacing: Tuple[float, float, float]
    affine: np.ndarray
    description: str
    expected: Dict[str, object]


def _draw_tube(vol: np.ndarray, p0: Sequence[float], p1: Sequence[float],
               radius_vox: float) -> None:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n = int(np.ceil(np.linalg.norm(p1 - p0))) * 3 + 2
    r = int(np.ceil(radius_vox))
    offs = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    offs = offs[np.linalg.norm(offs, axis=1) <= radius_vox + 1e-9]
    shape = np.array(vol.shape)
    for t in np.linspace(0.0, 1.0, n):
        c = np.rint(p0 + t * (p1 - p0)).astype(int)
        pts = c[None, :] + offs
        ok = np.all((pts >= 0) & (pts < shape[None, :]), axis=1)
        pts = pts[ok]
        vol[pts[:, 0], pts[:, 1], pts[:, 2]] = True


def _tree(shape: Tuple[int, int, int]) -> np.ndarray:
    """A small two-system coronary-like tree: one large trunk plus a separate system."""
    v = np.zeros(shape, dtype=bool)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    # left system: trunk with two branches
    _draw_tube(v, (cx - 18, cy, cz), (cx + 4, cy, cz), 2.2)
    _draw_tube(v, (cx + 4, cy, cz), (cx + 16, cy + 12, cz + 4), 1.6)
    _draw_tube(v, (cx + 4, cy, cz), (cx + 14, cy - 11, cz - 5), 1.6)
    # right system: legitimately separate second major component
    _draw_tube(v, (cx - 14, cy + 16, cz + 10), (cx + 8, cy + 20, cz + 12), 1.8)
    return v


def _affine(spacing: Sequence[float]) -> np.ndarray:
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = float(spacing[0]), float(spacing[1]), float(spacing[2])
    a[:3, 3] = [-30.0, -25.0, -20.0]
    return a


def make_phantoms(
    shape: Tuple[int, int, int] = (64, 64, 64),
    spacing: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    seed: int = 20260826,
) -> List[Phantom]:
    """Six phantoms exercising every code path and every expected outcome."""
    rng = np.random.default_rng(int(seed))
    affine = _affine(spacing)
    gt = _tree(shape)
    phantoms: List[Phantom] = []

    # 1. clean: prediction == ground truth. Every strategy must be near-identity.
    phantoms.append(Phantom(
        "phantom_clean", gt.copy(), gt.copy(), spacing, affine,
        "Prediction equals ground truth.",
        {"strategies_should_not_reduce_dice": True},
    ))

    # 2. speckle: many tiny isolated islands. S1/S2 must remove them and RAISE Dice.
    spk = gt.copy()
    n_speckle = 40
    coords = rng.integers(3, min(shape) - 3, size=(n_speckle, 3))
    for c in coords:
        if not gt[max(0, c[0] - 4):c[0] + 5, max(0, c[1] - 4):c[1] + 5, max(0, c[2] - 4):c[2] + 5].any():
            spk[c[0], c[1], c[2]] = True
    phantoms.append(Phantom(
        "phantom_speckle", gt.copy(), spk, spacing, affine,
        "Ground truth plus isolated single-voxel islands.",
        {"s1_should_remove_islands": True, "s1_should_not_decrease_dice": True},
    ))

    # 3. short gap: removing k voxels leaves a surface-to-surface gap of
    #    (k + 1) * spacing. One voxel removed gives exactly 1.2 mm, the primary
    #    bridging tolerance, so this phantom pins the primary setting.
    gap = gt.copy()
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    gap[cx - 8:cx - 7, :, :] = False
    phantoms.append(Phantom(
        "phantom_short_gap", gt.copy(), gap, spacing, affine,
        "Trunk severed by a one-voxel cut, a 1.2 mm surface-to-surface gap.",
        {"s3_should_reduce_components_at_1.2mm": True, "gap_mm": 1.2},
    ))

    # 4. wide gap: six voxels removed -> a 4.2 mm gap, wider than every tested
    #    tolerance. No strategy may bridge it.
    wide = gt.copy()
    wide[cx - 10:cx - 4, :, :] = False
    phantoms.append(Phantom(
        "phantom_wide_gap", gt.copy(), wide, spacing, affine,
        "Trunk severed by a six-voxel cut, a 4.2 mm gap, wider than any tested tolerance.",
        {"s3_should_not_bridge": True, "gap_mm": 4.2},
    ))

    # 5. two legitimate systems only: correction must NOT merge them.
    two = gt.copy()
    phantoms.append(Phantom(
        "phantom_two_systems", gt.copy(), two, spacing, affine,
        "Two anatomically separate coronary systems, no defect.",
        {"must_not_collapse_to_one_component": True},
    ))

    # 6. mixed: speckle plus a short gap plus an over-segmented blob.
    mixed = spk.copy()
    mixed[cx - 8:cx - 7, :, :] = False
    _draw_tube(mixed, (5, 5, 5), (9, 9, 9), 2.0)     # false-positive blob far from the tree
    phantoms.append(Phantom(
        "phantom_mixed", gt.copy(), mixed, spacing, affine,
        "Speckle, a short gap and a large false-positive blob.",
        {"s1_should_keep_large_false_positive": True},
    ))
    return phantoms


def write_phantom_case(ph: Phantom, root, threshold_name: str = "seg_mask_0.5.nii.gz"):
    """Materialise a phantom on disk in the layout the runner expects."""
    import nibabel as nib
    from pathlib import Path

    root = Path(root)
    pred_dir = root / "pred" / ph.case_id
    gt_dir = root / "gt" / ph.case_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pred_img = nib.Nifti1Image(ph.prediction.astype(np.uint8), ph.affine)
    pred_img.header.set_zooms(tuple(ph.spacing))
    nib.save(pred_img, str(pred_dir / threshold_name))

    gt_img = nib.Nifti1Image(ph.ground_truth.astype(np.uint8), ph.affine)
    gt_img.header.set_zooms(tuple(ph.spacing))
    nib.save(gt_img, str(gt_dir / f"{ph.case_id}.label.nii.gz"))

    # A synthetic CT-like volume so failure-analysis rendering has an image channel.
    rng = np.random.default_rng(7)
    ct = (rng.normal(0.0, 20.0, ph.ground_truth.shape) + ph.ground_truth * 300.0).astype(np.float32)
    ct_img = nib.Nifti1Image(ct, ph.affine)
    ct_img.header.set_zooms(tuple(ph.spacing))
    nib.save(ct_img, str(gt_dir / f"{ph.case_id}.img.nii.gz"))
    return pred_dir, gt_dir
