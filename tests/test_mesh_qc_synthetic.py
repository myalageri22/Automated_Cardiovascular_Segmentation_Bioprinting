from pathlib import Path

import numpy as np
import pytest

import phaseb_mesh_qc


def test_synthetic_mesh_qc_sphere(tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    grid = np.indices((24, 24, 24))
    center = np.array([12, 12, 12])[:, None, None, None]
    mask = (np.sum((grid - center) ** 2, axis=0) < 64).astype(np.uint8)
    mask_path = tmp_path / "sphere.nii.gz"
    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    row = phaseb_mesh_qc.run_phaseb_for_case("sphere", mask_path, tmp_path / "mesh_qc")
    assert row["status"] == "ok"
    assert Path(row["repaired_stl"]).exists()
