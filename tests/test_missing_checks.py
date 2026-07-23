import numpy as np
import pytest

import run_missing_checks


def test_slicability_reports_all_closed_sections_for_box():
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(10.0, 8.0, 6.0))

    hit, closed, fraction = run_missing_checks.slicability_50_planes(
        mesh, n_planes=10
    )

    assert hit == 10
    assert closed == 10
    assert fraction == 1.0


def test_mask_centroid_uses_complete_nifti_affine():
    nib = pytest.importorskip("nibabel")
    data = np.zeros((5, 6, 7), dtype=np.uint8)
    data[1:4, 2:5, 3:6] = 1
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 40.0],
            [0.0, 3.0, 0.0, -20.0],
            [0.0, 0.0, 4.0, 100.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    centroid, bounds = run_missing_checks.mask_world_centroid(
        nib.Nifti1Image(data, affine), world_space="ras"
    )

    np.testing.assert_allclose(centroid, [36.0, -11.0, 116.0])
    np.testing.assert_allclose(bounds, [[34.0, -14.0, 112.0], [38.0, -8.0, 120.0]])
