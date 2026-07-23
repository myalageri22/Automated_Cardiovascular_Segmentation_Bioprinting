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
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 50.0],
            [0.0, 3.0, 0.0, -20.0],
            [0.0, 0.0, 4.0, 100.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    nib.Nifti1Image(mask, affine).to_filename(mask_path)
    row = phaseb_mesh_qc.run_phaseb_for_case("sphere", mask_path, tmp_path / "mesh_qc")
    assert row["status"] == "ok"
    assert Path(row["repaired_stl"]).exists()
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(row["repaired_stl"], process=True)
    np.testing.assert_allclose(
        mesh.bounds,
        np.array([[11.0, -6.5, 118.0], [41.0, 38.5, 178.0]]),
        atol=1e-6,
    )


def test_minimal_mesh_qc_skips_unrelated_expensive_outputs(tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:9, 3:9] = 1
    mask_path = tmp_path / "cube.nii.gz"
    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)

    row = phaseb_mesh_qc.run_phaseb_for_case(
        "cube", mask_path, tmp_path / "minimal_mesh_qc", minimal=True
    )

    assert row["status"] == "ok"
    assert Path(row["repaired_stl"]).exists()
    assert row["wall_thickness_compliance_fraction"] == "not_computed_minimal_qc"
    assert not (
        tmp_path / "minimal_mesh_qc/case_outputs/cube/segmentation_raw.stl"
    ).exists()
    assert not (
        tmp_path / "minimal_mesh_qc/case_outputs/cube/segmentation_smoothed.stl"
    ).exists()
