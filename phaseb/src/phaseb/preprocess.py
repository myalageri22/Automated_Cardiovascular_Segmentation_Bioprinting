"""Segmentation thresholding and sweep utilities."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import SegmentationType
from . import postprocess


def apply_threshold(seg_array: np.ndarray, seg_type: str, threshold: float) -> np.ndarray:
    if seg_type == SegmentationType.PROBABILITY:
        return seg_array >= threshold
    return seg_array > 0


def run_threshold_sweep(
    prob_array: np.ndarray,
    spacing: Tuple[float, float, float],
    thresholds: List[float],
    post_cfg: Dict,
    seed_index_zyx: Optional[Tuple[int, int, int]] = None,
) -> (float, List[Dict]):
    results: List[Dict] = []
    best_score = -np.inf
    best_threshold = thresholds[0]
    for thr in thresholds:
        mask = apply_threshold(prob_array, SegmentationType.PROBABILITY, thr)
        cleaned = postprocess.clean_mask(
            mask,
            spacing=spacing,
            closing_mm=post_cfg.get("closing_mm", 1.0),
            fill_holes_first=post_cfg.get("fill_holes", True),
            min_component_mm3=post_cfg.get("min_component_mm3", 0.0),
            mode=post_cfg.get("keep_mode", "largest"),
            seed_index_zyx=seed_index_zyx,
        )
        comp_stats = postprocess.component_metrics(cleaned, spacing)
        score = comp_stats["largest_component_ratio"] * comp_stats["volume_mm3"] - 100.0 * max(
            comp_stats["component_count"] - 1, 0
        )
        results.append({
            "threshold": float(thr),
            "score": float(score),
            "component_count": comp_stats["component_count"],
            "volume_mm3": comp_stats["volume_mm3"],
            "largest_component_ratio": comp_stats["largest_component_ratio"],
        })
        if score > best_score:
            best_score = score
            best_threshold = thr
    return float(best_threshold), results


def threshold_and_postprocess(
    seg_array: np.ndarray,
    seg_type: str,
    threshold: float,
    spacing: Tuple[float, float, float],
    post_cfg: Dict,
    seed_index_zyx: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    mask = apply_threshold(seg_array, seg_type, threshold)
    cleaned = postprocess.clean_mask(
        mask,
        spacing=spacing,
        closing_mm=post_cfg.get("closing_mm", 1.0),
        fill_holes_first=post_cfg.get("fill_holes", True),
        min_component_mm3=post_cfg.get("min_component_mm3", 0.0),
        mode=post_cfg.get("keep_mode", "largest"),
        seed_index_zyx=seed_index_zyx,
    )
    return cleaned
