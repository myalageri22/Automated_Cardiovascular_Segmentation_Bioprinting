"""Topology-aware postprocessing for vascular masks."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.morphology import ball


def _closing_structure(radius_vox: int) -> np.ndarray:
    radius_vox = max(1, int(radius_vox))
    return ball(radius_vox)


def binary_closing_mm(mask: np.ndarray, spacing: Tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if radius_mm <= 0:
        return mask
    max_spacing = float(max(spacing))
    radius_vox = int(np.ceil(radius_mm / max_spacing))
    structure = _closing_structure(radius_vox)
    return ndimage.binary_closing(mask, structure=structure)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask)


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, num = ndimage.label(mask, structure=structure)
    return labeled, int(num)


def component_metrics(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float]:
    labeled, num = _label_components(mask)
    if num == 0:
        return {
            "component_count": 0,
            "volume_mm3": 0.0,
            "largest_component_mm3": 0.0,
            "largest_component_ratio": 0.0,
        }
    voxel_volume = float(np.prod(spacing))
    component_sizes = ndimage.sum(mask, labeled, index=range(1, num + 1))
    component_sizes = np.asarray(component_sizes, dtype=float)
    volumes_mm3 = component_sizes * voxel_volume
    total_volume = float(volumes_mm3.sum())
    largest_volume = float(volumes_mm3.max()) if volumes_mm3.size else 0.0
    largest_ratio = float(largest_volume / total_volume) if total_volume > 0 else 0.0
    return {
        "component_count": num,
        "volume_mm3": total_volume,
        "largest_component_mm3": largest_volume,
        "largest_component_ratio": largest_ratio,
    }


def remove_small_components(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    min_component_mm3: float,
    mode: str = "largest",
    seed_index_zyx: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    labeled, num = _label_components(mask)
    if num == 0:
        return mask
    voxel_volume = float(np.prod(spacing))
    component_sizes = ndimage.sum(mask, labeled, index=range(1, num + 1))
    component_sizes = np.asarray(component_sizes, dtype=float)
    volumes_mm3 = component_sizes * voxel_volume

    valid_labels = [i + 1 for i, vol in enumerate(volumes_mm3) if vol >= min_component_mm3]
    if not valid_labels:
        valid_labels = [int(np.argmax(volumes_mm3)) + 1]

    if mode == "seed" and seed_index_zyx is not None:
        z, y, x = seed_index_zyx
        if 0 <= z < labeled.shape[0] and 0 <= y < labeled.shape[1] and 0 <= x < labeled.shape[2]:
            seed_label = int(labeled[z, y, x])
            if seed_label in valid_labels:
                valid_labels = [seed_label]
            elif seed_label > 0:
                valid_labels = [seed_label]
    elif mode == "largest":
        largest_label = int(np.argmax(component_sizes)) + 1
        if largest_label in valid_labels:
            valid_labels = [largest_label]

    mask_out = np.isin(labeled, valid_labels)
    return mask_out


def clean_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    closing_mm: float,
    fill_holes_first: bool,
    min_component_mm3: float,
    mode: str = "largest",
    seed_index_zyx: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    working = mask.astype(bool)
    if fill_holes_first:
        working = fill_holes(working)
    working = binary_closing_mm(working, spacing, closing_mm)
    working = remove_small_components(
        working, spacing=spacing, min_component_mm3=min_component_mm3, mode=mode, seed_index_zyx=seed_index_zyx
    )
    working = fill_holes(working) if fill_holes_first else working
    return working
