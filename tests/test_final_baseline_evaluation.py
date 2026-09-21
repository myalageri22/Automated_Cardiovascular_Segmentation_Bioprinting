import numpy as np
import pytest

from scripts.evaluation.audit_exact_test_cohort import audit_rows
from scripts.evaluation.compare_attention_vs_plain import METRICS, compare, fdr_bh
from scripts.evaluation.compute_cldice_3d import cldice_components, skeletonize_volume


def test_skeletonize_volume_is_shape_preserving_boolean_3d():
    mask = np.zeros((12, 13, 14), dtype=np.uint8)
    mask[2:10, 6, 7] = 1
    skeleton = skeletonize_volume(mask)
    assert skeleton.shape == mask.shape
    assert skeleton.dtype == np.bool_
    assert skeleton.any()


def test_cldice_perfect_and_empty_behavior():
    empty = np.zeros((16, 16, 16), dtype=bool)
    vessel = empty.copy()
    vessel[3:13, 7, 7] = True
    assert cldice_components(vessel, vessel)["cldice"] == 1.0
    assert cldice_components(empty, empty)["cldice"] == 1.0
    assert cldice_components(vessel, empty)["cldice"] == 0.0


def test_cldice_decreases_for_broken_branch():
    label = np.zeros((24, 24, 24), dtype=bool)
    label[3:21, 12, 12] = True
    label[12, 12, 12:21] = True
    broken = label.copy()
    broken[12, 12, 15:21] = False
    assert 0.0 < cldice_components(broken, label)["cldice"] < 1.0


def test_cohort_audit_identifies_only_duplicates():
    test_ids = [str(index) for index in range(250)]
    rows = [{"case_id": case_id, "status": "ok"} for case_id in test_ids]
    rows.extend([{"case_id": "0", "status": "ok"}, {"case_id": "249", "status": "ok"}])
    audit = audit_rows(rows, test_ids)
    assert audit["rows"] == 252
    assert audit["unique_ids"] == 250
    assert audit["duplicate_ids"] == ["0", "249"]
    assert not audit["missing_ids"]
    assert not audit["extra_ids"]


def _model_rows(offset: float):
    return [
        {"case_id": str(index), **{key: index / 1000 + offset for key in METRICS}}
        for index in range(1, 6)
    ]


def test_paired_bootstrap_is_fixed_and_attention_minus_plain():
    first = compare(_model_rows(0.0), _model_rows(0.05), 5, 1_000, 42)
    second = compare(_model_rows(0.0), _model_rows(0.05), 5, 1_000, 42)
    assert first == second
    assert all(row["mean_paired_difference"] == pytest.approx(0.05) for row in first[2])
    assert all(row["bootstrap_seed"] == 42 for row in first[2])


def test_benjamini_hochberg_is_monotone_in_rank_order():
    p_values = [0.01, 0.04, 0.03, 0.2, 0.5]
    q_values = fdr_bh(p_values)
    ranked = sorted(zip(p_values, q_values))
    assert all(ranked[index][1] <= ranked[index + 1][1] for index in range(len(ranked) - 1))
    assert all(p <= q <= 1.0 for p, q in zip(p_values, q_values))
