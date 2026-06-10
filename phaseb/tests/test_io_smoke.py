import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from phaseb import pipeline  # noqa: E402


def _make_image(arr: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(spacing))
    return img


def test_pipeline_smoke(tmp_path):
    ct_arr = np.zeros((16, 16, 16), dtype=np.float32)
    ct_arr[8, 8, 8] = 100.0
    seg_arr = np.zeros_like(ct_arr)
    zz, yy, xx = np.ogrid[:16, :16, :16]
    mask = (xx - 8) ** 2 + (yy - 8) ** 2 + (zz - 8) ** 2 < 9
    seg_arr[mask] = 0.9

    ct_img = _make_image(ct_arr)
    seg_img = _make_image(seg_arr)

    ct_path = tmp_path / "ct.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"
    sitk.WriteImage(ct_img, str(ct_path))
    sitk.WriteImage(seg_img, str(seg_path))

    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    config = pipeline.load_config(str(config_path))

    outdir = tmp_path / "out"
    report = pipeline.run_case(
        ct_path=str(ct_path),
        seg_path=str(seg_path),
        seg_type="prob",
        case_id="smoke",
        outdir=outdir,
        config=config,
        threshold_override=0.5,
        threshold_sweep=False,
        generate_previews=False,
    )

    case_dir = outdir / "smoke"
    assert (case_dir / "vessels_raw.stl").exists()
    assert (case_dir / "vessels_repaired.stl").exists()
    assert report["mask_metrics"]["voxel_count"] > 0
    assert "mesh_repaired" in report
