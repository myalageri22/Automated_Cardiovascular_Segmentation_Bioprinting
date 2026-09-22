"""Regression test: HD95 must use the resampled-grid spacing, not the original header pixdim."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import evaluate_full_imagecas_test_mps as ev  # noqa: E402


class _Tensorish:
    def __init__(self, affine):
        self.affine = affine


def test_spacing_prefers_resampled_affine_over_header_pixdim():
    affine = np.diag([0.6, 0.6, 0.6, 1.0])
    batch = {
        "label": _Tensorish(torch.as_tensor(affine)[None]),
        "label_meta_dict": {"pixdim": torch.tensor([[1.0, 0.35, 0.35, 0.4, 1, 1, 1, 1]])},
    }
    assert ev.spacing_from_batch(batch, (9.0, 9.0, 9.0)) == pytest.approx((0.6, 0.6, 0.6))


def test_spacing_falls_back_to_configured_grid():
    assert ev.spacing_from_batch({}, (0.6, 0.6, 0.6)) == (0.6, 0.6, 0.6)


def test_hd95_single_voxel_shift_on_06mm_grid():
    a = np.zeros((10, 10, 10), bool); a[3:6, 3:6, 3:6] = True
    b = np.roll(a, 1, axis=0)
    assert ev.hd95_cpu(a, b, (0.6, 0.6, 0.6)) == pytest.approx(0.6)
