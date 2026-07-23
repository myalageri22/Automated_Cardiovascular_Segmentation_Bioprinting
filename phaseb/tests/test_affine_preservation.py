import numpy as np
import SimpleITK as sitk

from phaseb import mesh, repair


def test_mesh_export_and_repair_preserve_full_physical_frame():
    mask = np.zeros((8, 9, 10), dtype=np.uint8)
    mask[2:6, 2:7, 3:8] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.7, 1.1, 1.3))
    image.SetOrigin((40.0, -25.0, 80.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    raw, _, _ = mesh.mask_to_mesh(mask, image)
    repaired, _ = repair.repair_mesh(raw, use_pymeshfix=False)

    np.testing.assert_allclose(repaired.bounds, raw.bounds, atol=1e-7)
    assert raw.bounds[0, 0] > 30.0
    assert raw.bounds[0, 1] < 0.0
    assert raw.bounds[0, 2] > 80.0
