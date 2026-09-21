import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage import measure

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from phaseb import mesh as mesh_utils  # noqa: E402


def test_bounding_box_physical_scaling():
    spacing = (0.5, 1.0, 2.0)
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    image = sitk.GetImageFromArray(arr.astype(np.uint8))
    image.SetSpacing(spacing)

    mesh, _, _ = mesh_utils.mask_to_mesh(arr, image)
    bbox_mm = mesh.bounds[1] - mesh.bounds[0]

    verts_vox, _, _, _ = measure.marching_cubes(arr.astype(float), level=0.5)
    bbox_vox = verts_vox.max(axis=0) - verts_vox.min(axis=0)  # z, y, x order
    expected = np.array([bbox_vox[2] * spacing[0], bbox_vox[1] * spacing[1], bbox_vox[0] * spacing[2]])
    assert np.allclose(bbox_mm, expected, atol=1e-3)
