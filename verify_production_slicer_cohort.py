#!/usr/bin/env python3
"""Verify the retained 250-case production-slicer publication artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUIRED_SUCCESS_FIELDS = (
    "case_id",
    "process_exit_code",
    "gcode_exists",
    "gcode_is_textual",
    "layer_count",
    "extrusion_move_count",
    "complete_stl_processed",
    "build_volume_fit",
    "support_generation_result",
    "toolpath_generation_success",
    "status",
)
EXPECTED_WARNING_CASES = {"803", "821", "822"}


class VerificationError(ValueError):
    """Raised when retained publication artifacts are inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise VerificationError(f"{field} contains a non-boolean value: {value!r}")
    return normalized == "true"


def parse_number(value: Any, field: str) -> float:
    if value is None or str(value).strip() in {"", "null", "None", "NaN", "nan"}:
        raise VerificationError(f"{field} contains a missing or placeholder value")
    number = float(value)
    if not math.isfinite(number):
        raise VerificationError(f"{field} contains a non-finite value: {value!r}")
    return number


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = sorted(float(value) for value in values)
    if not data:
        raise VerificationError("cannot summarize an empty numeric series")
    if len(data) == 1:
        q1 = q3 = data[0]
    else:
        q1, _, q3 = statistics.quantiles(data, n=4, method="inclusive")
    return {
        "n": len(data),
        "min": data[0],
        "q1": q1,
        "median": statistics.median(data),
        "q3": q3,
        "max": data[-1],
    }


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise VerificationError(f"{message}: expected {expected!r}, found {actual!r}")


def _resolve_recorded_path(recorded: str, repo: Path) -> Path:
    path = Path(recorded)
    if path.is_absolute():
        return path
    return repo / path


def verify_cohort(
    *,
    per_case_csv: Path,
    summary_json: Path,
    manifest_json: Path,
    phase_b_csv: Path,
    test_metrics_csv: Path,
    profile_path: Path,
    expected_size: int = 250,
    verify_payload_hashes: bool = False,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parent
    rows = read_csv(per_case_csv)
    phase_rows = read_csv(phase_b_csv)
    test_rows = read_csv(test_metrics_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    _assert_equal(len(rows), expected_size, "per-case row count")
    case_ids = [row["case_id"] for row in rows]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    _assert_equal(duplicates, [], "duplicate case IDs")
    _assert_equal(len(set(case_ids)), expected_size, "unique case count")

    phase_ids = {row["case_id"] for row in phase_rows}
    test_ids = {row["case_id"] for row in test_rows}
    manifest_inputs = manifest.get("discovered_inputs")
    if not isinstance(manifest_inputs, list):
        raise VerificationError("validation manifest has no discovered_inputs list")
    manifest_ids = {str(item["case_id"]) for item in manifest_inputs}
    _assert_equal(set(case_ids), phase_ids, "slicer/Phase B case-ID sets")
    _assert_equal(set(case_ids), test_ids, "slicer/held-out-test case-ID sets")
    _assert_equal(set(case_ids), manifest_ids, "slicer/manifest case-ID sets")
    if "720" in case_ids:
        raise VerificationError("validation-partition case 720 is present in the held-out slicer cohort")

    missing_required = {
        field: sum(row.get(field) in {None, "", "null", "None", "NaN", "nan"} for row in rows)
        for field in REQUIRED_SUCCESS_FIELDS
    }
    _assert_equal({key: value for key, value in missing_required.items() if value}, {}, "missing required values")

    _assert_equal(sum(row["status"] == "success" for row in rows), expected_size, "successful statuses")
    _assert_equal(sum(parse_bool(row["toolpath_generation_success"], "toolpath_generation_success") for row in rows), expected_size, "successful toolpaths")
    _assert_equal(sum(parse_number(row["process_exit_code"], "process_exit_code") == 0 for row in rows), expected_size, "zero exit statuses")
    _assert_equal(sum(parse_bool(row["gcode_exists"], "gcode_exists") for row in rows), expected_size, "nonempty G-code outputs")
    _assert_equal(sum(parse_bool(row["gcode_is_textual"], "gcode_is_textual") for row in rows), expected_size, "textual G-code outputs")
    _assert_equal(sum(parse_number(row["layer_count"], "layer_count") >= 1 for row in rows), expected_size, "rows with printed layers")
    _assert_equal(sum(parse_number(row["extrusion_move_count"], "extrusion_move_count") >= 1 for row in rows), expected_size, "rows with positive-extrusion movement")
    _assert_equal(sum(parse_bool(row["complete_stl_processed"], "complete_stl_processed") for row in rows), expected_size, "complete STLs processed")
    _assert_equal(sum(parse_bool(row["build_volume_fit"], "build_volume_fit") for row in rows), expected_size, "build-volume fits")
    _assert_equal(sum(row["support_generation_result"] == "generated" for row in rows), expected_size, "support-generation successes")

    warning_details: dict[str, list[str]] = {}
    for row in rows:
        warnings = json.loads(row["slicer_warnings_json"])
        if warnings:
            warning_details[row["case_id"]] = warnings
    _assert_equal(set(warning_details), EXPECTED_WARNING_CASES, "warning case IDs")
    if any("empty layer" not in " ".join(warnings).lower() for warnings in warning_details.values()):
        raise VerificationError("one or more retained warnings is not an empty-layer warning")

    layer_summary = numeric_summary(parse_number(row["layer_count"], "layer_count") for row in rows)
    extrusion_summary = numeric_summary(parse_number(row["extrusion_move_count"], "extrusion_move_count") for row in rows)
    time_summary = numeric_summary(parse_number(row["estimated_print_time_seconds"], "estimated_print_time_seconds") for row in rows)
    _assert_equal(summary.get("layer_count_summary"), layer_summary, "summary JSON layer statistics")
    _assert_equal(summary.get("positive_extrusion_move_count_summary"), extrusion_summary, "summary JSON extrusion statistics")
    _assert_equal(summary.get("estimated_print_time_seconds_summary"), time_summary, "summary JSON time statistics")
    for key, expected in {
        "expected_cohort_size": expected_size,
        "available_cases": expected_size,
        "attempted_cases": expected_size,
        "successful_cases": expected_size,
        "failed_cases_count": 0,
        "number_attempted": expected_size,
        "number_successful": expected_size,
        "number_failed": 0,
        "warning_free_cases": 247,
        "warned_cases_count": 3,
        "complete_stl_processed_count": expected_size,
        "build_volume_fit_count": expected_size,
        "support_generation_count": expected_size,
        "success_percentage": 100.0,
        "result_scope": "full_cohort",
        "prusaslicer_version": "PrusaSlicer 2.9.6",
    }.items():
        _assert_equal(summary.get(key), expected, f"summary JSON {key}")

    profile_hash = sha256_file(profile_path)
    _assert_equal(summary.get("printer_profile_sha256"), profile_hash, "summary/profile hash")
    _assert_equal(manifest.get("printer_profile_sha256"), profile_hash, "manifest/profile hash")
    _assert_equal(manifest.get("prusaslicer_version"), "PrusaSlicer 2.9.6", "manifest slicer version")
    _assert_equal(manifest.get("dry_run"), False, "manifest dry-run state")

    phase_by_id = {row["case_id"]: row for row in phase_rows}
    mesh_integrity_failures = sorted(
        case_id
        for case_id, row in phase_by_id.items()
        if not (
            parse_bool(row["watertight"], "watertight")
            and parse_number(row["non_manifold_edge_count"], "non_manifold_edge_count") == 0
        )
    )
    _assert_equal(len(mesh_integrity_failures), 22, "mesh-integrity failure count")
    slicer_by_id = {row["case_id"]: row for row in rows}
    if not all(parse_bool(slicer_by_id[case_id]["toolpath_generation_success"], "toolpath_generation_success") for case_id in mesh_integrity_failures):
        raise VerificationError("at least one mesh-integrity failure did not meet the slicer criteria")

    payload_hashes_checked = 0
    if verify_payload_hashes:
        manifest_by_id = {str(item["case_id"]): item for item in manifest_inputs}
        for row in rows:
            case_id = row["case_id"]
            _assert_equal(row["stl_sha256"], manifest_by_id[case_id]["sha256"], f"case {case_id} STL hash in CSV/manifest")
            for path_field, hash_field in (("stl_path", "stl_sha256"), ("gcode_path", "gcode_sha256")):
                payload_path = _resolve_recorded_path(row[path_field], repo)
                if not payload_path.is_file():
                    raise VerificationError(f"case {case_id} payload is missing: {payload_path}")
                _assert_equal(sha256_file(payload_path), row[hash_field], f"case {case_id} {path_field} hash")
                payload_hashes_checked += 1

    return {
        "verdict": "verified",
        "quartile_convention": "Python statistics.quantiles(n=4, method='inclusive')",
        "inputs": {
            str(path.relative_to(repo)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (per_case_csv, summary_json, manifest_json, phase_b_csv, test_metrics_csv, profile_path)
        },
        "cohort": {
            "rows": len(rows),
            "unique_case_ids": len(set(case_ids)),
            "case_id_min": min(map(int, case_ids)),
            "case_id_max": max(map(int, case_ids)),
            "case_720_present": False,
            "attempted": expected_size,
            "successful": expected_size,
            "failed": 0,
            "warning_free": expected_size - len(warning_details),
            "warned": len(warning_details),
            "warning_details": warning_details,
            "complete_stl_processed": expected_size,
            "build_volume_fit": expected_size,
            "organic_support_paths_generated": expected_size,
            "mesh_integrity_failures_succeeding": len(mesh_integrity_failures),
            "mesh_integrity_failure_case_ids": mesh_integrity_failures,
        },
        "statistics": {
            "layers": layer_summary,
            "positive_extrusion_movements": extrusion_summary,
            "estimated_time_seconds": time_summary,
        },
        "software": {
            "prusaslicer_version": manifest["prusaslicer_version"],
            "profile_sha256": profile_hash,
        },
        "consistency": {
            "summary_matches_per_case_csv": True,
            "manifest_matches_per_case_csv": True,
            "held_out_case_ids_match": True,
            "phase_b_case_ids_match": True,
            "required_null_or_placeholder_values": 0,
            "payload_hashes_checked": payload_hashes_checked,
        },
    }


def render_markdown(report: dict[str, Any], command: str) -> str:
    cohort = report["cohort"]
    stats = report["statistics"]
    warning_lines = "\n".join(
        f"- Case {case_id}: {'; '.join(warnings)}"
        for case_id, warnings in cohort["warning_details"].items()
    )
    input_lines = "\n".join(
        f"- `{path}` — SHA-256 `{metadata['sha256']}`, {metadata['size_bytes']} bytes"
        for path, metadata in report["inputs"].items()
    )
    return f"""# Production-slicer cohort verification

## Verdict

Verified with no discrepancies. The retained artifacts describe exactly 250 unique held-out test cases, all of which met the predefined software-level PrusaSlicer toolpath-generation criteria. Case 720 is absent from this cohort.

## Command

```bash
{command}
```

Quartiles use `{report['quartile_convention']}`.

## Inputs

{input_lines}

## Recomputed results

- Attempts / successes / failures: {cohort['attempted']} / {cohort['successful']} / {cohort['failed']}
- Warning-free / warned cases: {cohort['warning_free']} / {cohort['warned']}
- Complete STLs processed: {cohort['complete_stl_processed']}/250
- Native-scale reference build-volume fits: {cohort['build_volume_fit']}/250
- Organic support paths generated: {cohort['organic_support_paths_generated']}/250
- Layers: median {stats['layers']['median']:.0f}, IQR {stats['layers']['q1']:.0f}-{stats['layers']['q3']:.0f}, range {stats['layers']['min']:.0f}-{stats['layers']['max']:.0f}
- Positive-extrusion movements: median {stats['positive_extrusion_movements']['median']:.0f}, IQR {stats['positive_extrusion_movements']['q1']:.2f}-{stats['positive_extrusion_movements']['q3']:.2f}, range {stats['positive_extrusion_movements']['min']:.0f}-{stats['positive_extrusion_movements']['max']:.0f}
- Estimated time: median {stats['estimated_time_seconds']['median']:.1f} s, IQR {stats['estimated_time_seconds']['q1']:.2f}-{stats['estimated_time_seconds']['q3']:.1f} s, range {stats['estimated_time_seconds']['min']:.0f}-{stats['estimated_time_seconds']['max']:.0f} s
- Earlier combined mesh-integrity failures that met slicer criteria: {cohort['mesh_integrity_failures_succeeding']}/22

## Warnings

{warning_lines}

## Integrity and consistency

- The per-case CSV, summary JSON, validation manifest, held-out Phase A cohort, and corrected Phase B cohort use the same 250 case IDs.
- Every required success field is populated with a finite, non-placeholder value.
- The profile hash and PrusaSlicer version agree across the retained artifacts.
- Recorded STL and G-code hashes were checked for {report['consistency']['payload_hashes_checked']} payloads during final verification.
- No discrepancy was found between recomputed statistics and the committed summary.

No physical fabrication was performed. This report verifies software-level production-slicer execution under one fixed computational reference profile only.
"""


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    cohort = root / "outputs/production_slicer_validation/cohort_250_real"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case-csv", type=Path, default=cohort / "per_case_slicer_results.csv")
    parser.add_argument("--summary-json", type=Path, default=cohort / "summary_slicer_results.json")
    parser.add_argument("--manifest-json", type=Path, default=cohort / "validation_manifest.json")
    parser.add_argument("--phase-b-csv", type=Path, default=root / "outputs/phase_b_mesh_qc/per_case_mesh_qc.csv")
    parser.add_argument("--test-metrics-csv", type=Path, default=root / "outputs/final_test_250/per_case_metrics.csv")
    parser.add_argument("--profile", type=Path, default=root / "outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini")
    parser.add_argument("--expected-size", type=int, default=250)
    parser.add_argument("--verify-payload-hashes", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = verify_cohort(
        per_case_csv=args.per_case_csv.resolve(),
        summary_json=args.summary_json.resolve(),
        manifest_json=args.manifest_json.resolve(),
        phase_b_csv=args.phase_b_csv.resolve(),
        test_metrics_csv=args.test_metrics_csv.resolve(),
        profile_path=args.profile.resolve(),
        expected_size=args.expected_size,
        verify_payload_hashes=args.verify_payload_hashes,
    )
    command = (
        "python verify_production_slicer_cohort.py "
        "--verify-payload-hashes "
        "--json-output outputs/production_slicer_validation/cohort_250_real/verification_report.json "
        "--markdown-output outputs/production_slicer_validation/cohort_250_real/VERIFICATION_REPORT.md"
    )
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(render_markdown(report, command), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
