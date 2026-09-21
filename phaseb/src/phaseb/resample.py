"""Resampling utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk


def resample_image(image: sitk.Image, target_spacing: float, is_label: bool = False) -> Tuple[sitk.Image, np.ndarray]:
    """Resample an image to isotropic spacing in mm.

    Args:
        image: Input SimpleITK image.
        target_spacing: Desired isotropic spacing in mm.
        is_label: Whether to use nearest-neighbor interpolation.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    spacing = (float(target_spacing),) * 3

    new_size = [
        int(round(osz * ospc / target_spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    new_size = [max(1, n) for n in new_size]

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    resampled = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )
    return resampled, sitk.GetArrayFromImage(resampled)
