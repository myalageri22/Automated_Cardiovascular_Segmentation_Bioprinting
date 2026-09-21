import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from phaseb import postprocess  # noqa: E402


def test_keep_largest_with_min_component():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    mask[0, 0, 0] = True  # tiny blob
    spacing = (1.0, 1.0, 1.0)

    cleaned = postprocess.clean_mask(
        mask,
        spacing=spacing,
        closing_mm=0.0,
        fill_holes_first=True,
        min_component_mm3=20.0,
        mode="largest",
        seed_index_zyx=None,
    )
    metrics = postprocess.component_metrics(cleaned, spacing)
    assert metrics["component_count"] == 1
    assert cleaned[1, 1, 1]


def test_seed_component_preserved_despite_volume_filter():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    mask[0, 0, 0] = True
    spacing = (1.0, 1.0, 1.0)

    cleaned = postprocess.clean_mask(
        mask,
        spacing=spacing,
        closing_mm=0.0,
        fill_holes_first=True,
        min_component_mm3=20.0,
        mode="seed",
        seed_index_zyx=(0, 0, 0),
    )
    assert cleaned.sum() >= 1
    assert cleaned[0, 0, 0]
