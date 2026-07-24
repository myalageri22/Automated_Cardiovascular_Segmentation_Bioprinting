from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import production_slicer_validation as psv
import verify_production_slicer_cohort as verifier


def test_positive_extrusion_detection_requires_spatial_motion() -> None:
    parsed = psv.parse_gcode_text("M83\nG1 E1\nG1 X1 E0.2\n")
    assert parsed.extrusion_move_count == 1


def test_absolute_extrusion_parsing() -> None:
    parsed = psv.parse_gcode_text("M82\nG92 E0\nG1 Z0.2\nG1 X1 E0.5\nG1 X2 E0.4\nG1 X3 E0.8\n")
    assert parsed.extrusion_move_count == 2


def test_relative_extrusion_parsing() -> None:
    parsed = psv.parse_gcode_text("M83\nG1 Z0.2\nG1 X1 E0.1\nG1 X2 E0.2\n")
    assert parsed.extrusion_move_count == 2


def test_retraction_is_not_positive_extrusion() -> None:
    parsed = psv.parse_gcode_text("M83\nG1 Z0.2\nG1 X1 E-0.8\nG1 X2 E0\n")
    assert parsed.extrusion_move_count == 0


def test_layer_marker_detection() -> None:
    parsed = psv.parse_gcode_text(";LAYER_CHANGE\nG1 Z0.2\nG1 X1 E0.1\n;LAYER_CHANGE\nG1 Z0.4\nG1 X2 E0.2\n")
    assert parsed.layer_count == 2
    assert parsed.layer_detection_method == "prusa_layer_markers"


def test_z_height_fallback() -> None:
    parsed = psv.parse_gcode_text("M83\nG1 Z0.2\nG1 X1 E0.1\nG1 Z0.4\nG1 X2 E0.1\n")
    assert parsed.layer_count == 2
    assert parsed.layer_detection_method == "print_z_changes"


def test_empty_gcode(tmp_path: Path) -> None:
    output = tmp_path / "empty.gcode"
    output.write_text("")
    success, analysis, reason = psv.assess_toolpath(0, output)
    assert not success
    assert not analysis.is_textual
    assert "empty" in reason


def test_nonzero_slicer_exit(tmp_path: Path) -> None:
    output = tmp_path / "valid.gcode"
    output.write_text(";LAYER_CHANGE\nM83\nG1 Z0.2\nG1 X1 E0.1\n")
    success, _, reason = psv.assess_toolpath(1, output)
    assert not success
    assert "exit code" in reason


def test_missing_output_file(tmp_path: Path) -> None:
    success, _, reason = psv.assess_toolpath(0, tmp_path / "missing.gcode")
    assert not success
    assert "missing" in reason


def test_binary_gcode_rejection(tmp_path: Path) -> None:
    output = tmp_path / "binary.bgcode"
    output.write_bytes(psv.GCODE_BINARY_MAGIC + b"\x00\x01")
    success, analysis, reason = psv.assess_toolpath(0, output)
    assert not success
    assert not analysis.is_textual
    assert "binary" in reason


def test_summary_aggregation() -> None:
    rows = [
        {
            "case_id": "720",
            "status": "success",
            "toolpath_generation_success": True,
            "runtime_seconds": 2.0,
            "layer_count": 10,
            "estimated_print_time_seconds": 120.0,
            "filament_length_mm": 50.0,
            "filament_volume_cm3": None,
            "filament_mass_g": 1.5,
            "failure_reason": None,
        },
        {
            "case_id": "721",
            "status": "failed",
            "toolpath_generation_success": False,
            "runtime_seconds": 1.0,
            "layer_count": 0,
            "estimated_print_time_seconds": None,
            "filament_length_mm": None,
            "filament_volume_cm3": None,
            "filament_mass_g": None,
            "failure_reason": "failed",
        },
    ]
    summary = psv.aggregate_summary(rows, 2, "2.9.0", "abc", 250, {"git_commit_hash": "def", "git_dirty_working_tree": True})
    assert summary["number_attempted"] == 2
    assert summary["number_successful"] == 1
    assert summary["number_failed"] == 1
    assert summary["success_percentage"] is None
    assert summary["success_fraction"] == "1/2 available cases"
    assert summary["layer_count_summary"]["median"] == 10
    assert summary["result_scope"] == "available_subset"
    assert summary["warning_free_cases"] == 1
    assert summary["warned_cases_count"] == 0


def test_case_id_extraction() -> None:
    assert psv.extract_case_id(Path("outputs/phase_b_mesh_qc/case_outputs/720/segmentation_repaired.stl")) == "720"
    assert psv.extract_case_id(Path("exports/case-890_repaired.stl")) == "890"


def test_sha256_calculation(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"fixture")
    assert psv.sha256_file(path) == hashlib.sha256(b"fixture").hexdigest()


def test_support_and_object_extrusion_classification() -> None:
    parsed = psv.parse_gcode_text(
        ";LAYER_CHANGE\nM83\nG1 Z0.2\n;TYPE:Perimeter\nG1 X1 E0.2\n"
        ";TYPE:Support material\nG1 X2 E0.3\n;TYPE:Support material interface\nG1 X3 E0.4\n"
    )
    assert parsed.object_extrusion_move_count == 1
    assert parsed.support_extrusion_move_count == 1
    assert parsed.support_interface_extrusion_move_count == 1
    assert parsed.support_layer_count == 1
    assert parsed.support_generation_result == "generated"
    assert abs(float(parsed.support_filament_length_mm) - 0.7) < 1e-12


def test_missing_support_comments_report_unavailable() -> None:
    parsed = psv.parse_gcode_text(";LAYER_CHANGE\nM83\nG1 Z0.2\nG1 X1 E0.2\n")
    assert parsed.object_extrusion_move_count is None
    assert parsed.support_extrusion_move_count is None
    assert parsed.support_generation_result == "unavailable_no_extrusion_role_comments"


def test_representative_scope_has_explicit_fraction_without_percentage() -> None:
    rows = [{
        "case_id": "720", "status": "success", "toolpath_generation_success": True,
        "runtime_seconds": 1.0, "layer_count": 2, "estimated_print_time_seconds": None,
        "filament_length_mm": None, "filament_volume_cm3": None, "filament_mass_g": None,
        "failure_reason": None,
    }]
    summary = psv.aggregate_summary(rows, 1, "PrusaSlicer 2.9.6", "abc", 250, {})
    assert summary["result_scope"] == "representative_case"
    assert summary["success_fraction"] == "1/1 available representative case"
    assert summary["success_denominator"] == 1
    assert summary["success_percentage"] is None


def test_full_cohort_scope_requires_all_attempted() -> None:
    row = {
        "case_id": "x", "status": "success", "toolpath_generation_success": True,
        "runtime_seconds": 1.0, "layer_count": 2, "estimated_print_time_seconds": None,
        "filament_length_mm": None, "filament_volume_cm3": None, "filament_mass_g": None,
        "failure_reason": None,
    }
    summary = psv.aggregate_summary([row], 250, "PrusaSlicer 2.9.6", "abc", 250, {})
    assert summary["result_scope"] == "available_subset"


def test_warning_and_cohort_evidence_aggregation() -> None:
    row = {
        "case_id": "803",
        "status": "success",
        "toolpath_generation_success": True,
        "runtime_seconds": 1.0,
        "layer_count": 2,
        "extrusion_move_count": 3,
        "estimated_print_time_seconds": 4.0,
        "filament_length_mm": None,
        "filament_volume_cm3": None,
        "filament_mass_g": None,
        "complete_stl_processed": True,
        "build_volume_fit": True,
        "support_generation_result": "generated",
        "slicer_warnings_json": '["print warning: Empty layer"]',
        "failure_reason": None,
    }
    summary = psv.aggregate_summary([row], 1, "PrusaSlicer 2.9.6", "abc", 1, {})
    assert summary["warned_cases_count"] == 1
    assert summary["warning_free_cases"] == 0
    assert summary["warned_cases"][0]["case_id"] == "803"
    assert summary["complete_stl_processed_count"] == 1
    assert summary["build_volume_fit_count"] == 1
    assert summary["support_generation_count"] == 1
    assert summary["positive_extrusion_move_count_summary"]["median"] == 3


def test_duplicate_case_id_detection() -> None:
    paths = [
        Path("outputs/phase_b_mesh_qc/case_outputs/803/segmentation_repaired.stl"),
        Path("other/case_803_repaired.stl"),
        Path("outputs/phase_b_mesh_qc/case_outputs/804/segmentation_repaired.stl"),
    ]
    assert psv.duplicate_case_ids(paths) == ["803"]


def test_profile_diff_generation(tmp_path: Path) -> None:
    source = tmp_path / "source.ini"
    derived = tmp_path / "derived.ini"
    source.write_text("support_material_auto = 0\nunchanged = yes\n")
    derived.write_text("support_material_auto = 1\nunchanged = yes\n")
    assert psv.profile_field_diff(source, derived) == {
        "support_material_auto": {"source": "0", "derived": "1"}
    }


def test_stl_bounding_box_and_scale_recording(tmp_path: Path) -> None:
    path = tmp_path / "triangle.stl"
    header = b"fixture".ljust(80, b" ")
    facet = __import__("struct").pack("<12fH", 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 4, 0)
    path.write_bytes(header + __import__("struct").pack("<I", 1) + facet)
    geometry = psv.stl_geometry(path)
    assert geometry["extents"] == (2.0, 3.0, 4.0)
    assert geometry["triangle_count"] == 1
    assert geometry["binary_complete"] is True


def _retained_cohort_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    cohort = root / "outputs/production_slicer_validation/cohort_250_real"
    return {
        "per_case_csv": cohort / "per_case_slicer_results.csv",
        "summary_json": cohort / "summary_slicer_results.json",
        "manifest_json": cohort / "validation_manifest.json",
        "phase_b_csv": root / "outputs/phase_b_mesh_qc/per_case_mesh_qc.csv",
        "test_metrics_csv": root / "outputs/final_test_250/per_case_metrics.csv",
        "profile_path": root / "outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini",
    }


def test_retained_full_cohort_artifacts_are_consistent() -> None:
    report = verifier.verify_cohort(**_retained_cohort_paths())
    assert report["verdict"] == "verified"
    assert report["cohort"]["unique_case_ids"] == 250
    assert report["cohort"]["warning_free"] == 247
    assert report["cohort"]["warned"] == 3
    assert report["cohort"]["mesh_integrity_failures_succeeding"] == 22


def test_retained_cohort_verifier_rejects_duplicate_case(tmp_path: Path) -> None:
    paths = _retained_cohort_paths()
    rows = paths["per_case_csv"].read_text(encoding="utf-8").splitlines()
    duplicate_csv = tmp_path / "duplicates.csv"
    duplicate_csv.write_text("\n".join([rows[0], rows[1], rows[1], *rows[3:]]) + "\n", encoding="utf-8")
    paths["per_case_csv"] = duplicate_csv
    with pytest.raises(verifier.VerificationError, match="duplicate case IDs"):
        verifier.verify_cohort(**paths)


def test_retained_cohort_verifier_rejects_missing_case(tmp_path: Path) -> None:
    paths = _retained_cohort_paths()
    rows = paths["per_case_csv"].read_text(encoding="utf-8").splitlines()
    missing_csv = tmp_path / "missing.csv"
    missing_csv.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    paths["per_case_csv"] = missing_csv
    with pytest.raises(verifier.VerificationError, match="per-case row count"):
        verifier.verify_cohort(**paths)
