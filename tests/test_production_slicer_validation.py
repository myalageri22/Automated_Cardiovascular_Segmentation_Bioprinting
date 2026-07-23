from __future__ import annotations

import hashlib
from pathlib import Path

import production_slicer_validation as psv


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
