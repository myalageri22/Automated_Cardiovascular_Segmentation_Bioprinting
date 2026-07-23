#!/usr/bin/env python3
"""Reproducible PrusaSlicer toolpath-generation validation for repaired STLs.

This module complements, but does not replace, the repository's geometric
slicer-compatibility proxies.  A successful result requires a real slicer
process, textual G-code, at least one printed layer, and at least one motion
with positive extrusion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import re
import shlex
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = "1.0"
DEFAULT_INPUT_GLOB = "outputs/phase_b_mesh_qc/case_outputs/*/segmentation_repaired.stl"
GCODE_BINARY_MAGIC = b"GCDE"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PARAM_RE = re.compile(rf"([A-Za-z])\s*({FLOAT_PATTERN})")
LAYER_MARKER_RE = re.compile(r"^\s*;\s*(?:LAYER_CHANGE\b|LAYER\s*:\s*\d+\b)", re.IGNORECASE)


CSV_FIELDS = [
    "case_id",
    "stl_path",
    "stl_sha256",
    "stl_size_bytes",
    "slicer_name",
    "slicer_version",
    "printer_profile_path",
    "printer_profile_sha256",
    "command_text",
    "command_json",
    "start_timestamp_utc",
    "completion_timestamp_utc",
    "runtime_seconds",
    "process_exit_code",
    "stdout_log",
    "stderr_log",
    "gcode_path",
    "gcode_exists",
    "gcode_is_textual",
    "gcode_sha256",
    "gcode_size_bytes",
    "layer_count",
    "layer_detection_method",
    "extrusion_move_count",
    "object_extrusion_move_count",
    "support_extrusion_move_count",
    "support_interface_extrusion_move_count",
    "unclassified_extrusion_move_count",
    "support_layer_count",
    "support_generation_result",
    "object_filament_length_mm",
    "support_filament_length_mm",
    "support_interface_filament_length_mm",
    "estimated_print_time_seconds",
    "filament_length_mm",
    "filament_volume_cm3",
    "filament_mass_g",
    "object_scale_percent",
    "stl_bbox_x_mm",
    "stl_bbox_y_mm",
    "stl_bbox_z_mm",
    "stl_triangle_count",
    "stl_binary_complete",
    "placed_bbox_min_x_mm",
    "placed_bbox_min_y_mm",
    "placed_bbox_max_x_mm",
    "placed_bbox_max_y_mm",
    "placement_translation_x_mm",
    "placement_translation_y_mm",
    "placement_translation_z_mm",
    "source_orientation_retained",
    "build_volume_fit",
    "complete_stl_processed",
    "slicer_warnings_json",
    "toolpath_generation_success",
    "status",
    "failure_reason",
]


@dataclass(frozen=True)
class GcodeAnalysis:
    is_textual: bool
    layer_count: int
    layer_detection_method: Optional[str]
    extrusion_move_count: int
    object_extrusion_move_count: Optional[int]
    support_extrusion_move_count: Optional[int]
    support_interface_extrusion_move_count: Optional[int]
    unclassified_extrusion_move_count: int
    support_layer_count: Optional[int]
    support_generation_result: str
    object_filament_length_mm: Optional[float]
    support_filament_length_mm: Optional[float]
    support_interface_filament_length_mm: Optional[float]
    estimated_print_time_seconds: Optional[float]
    filament_length_mm: Optional[float]
    filament_volume_cm3: Optional[float]
    filament_mass_g: Optional[float]
    warnings: list[str]
    placed_bbox_xy: Optional[tuple[float, float, float, float]] = None
    failure_reason: Optional[str] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_case_id(path: Path) -> str:
    """Extract a stable case identifier from common Phase B paths."""
    for part in reversed(path.parts[:-1]):
        match = re.fullmatch(r"(?:case[_-]?)?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    stem = path.stem
    for pattern in (r"(?:^|[_-])case[_-]?(\d+)(?:[_-]|$)", r"(?:^|[_-])(\d{3,})(?:[_-]|$)"):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")
    return cleaned or "case"


def safe_case_slug(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("_.-") or "case"


def discover_stls(explicit_stls: Sequence[Path], input_globs: Sequence[str], case_filter: Optional[str]) -> list[Path]:
    discovered: set[Path] = set()
    for path in explicit_stls:
        if path.is_file():
            discovered.add(path.resolve())
    for pattern in input_globs:
        pattern_path = Path(pattern).expanduser()
        if pattern_path.is_absolute():
            anchor = Path(pattern_path.anchor)
            relative_pattern = str(pattern_path.relative_to(anchor))
            matches = anchor.glob(relative_pattern)
        else:
            matches = Path.cwd().glob(pattern)
        for path in matches:
            if path.is_file():
                discovered.add(path.resolve())
    paths = sorted(discovered, key=lambda item: str(item))
    if case_filter is not None:
        paths = [path for path in paths if extract_case_id(path) == case_filter]
    return paths


def parse_duration_seconds(value: str) -> Optional[float]:
    total = 0.0
    found = False
    for number, unit in re.findall(rf"({FLOAT_PATTERN})\s*([dhms])", value, flags=re.IGNORECASE):
        factor = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}[unit.lower()]
        total += float(number) * factor
        found = True
    return total if found else None


def _parse_footer_value(text: str, label: str, unit: str) -> Optional[float]:
    match = re.search(rf"^\s*;\s*{re.escape(label)}\s*\[{re.escape(unit)}\]\s*=\s*({FLOAT_PATTERN})", text, re.MULTILINE | re.IGNORECASE)
    return float(match.group(1)) if match else None


def _command_and_params(code: str) -> tuple[Optional[str], dict[str, float]]:
    code = code.strip()
    if not code:
        return None, {}
    command_match = re.match(r"^([GMT]\d+(?:\.\d+)?)\b", code, flags=re.IGNORECASE)
    if not command_match:
        return None, {}
    command = command_match.group(1).upper()
    params = {letter.upper(): float(value) for letter, value in PARAM_RE.findall(code[command_match.end() :])}
    return command, params


def parse_gcode_text(text: str) -> GcodeAnalysis:
    if not text.strip():
        return GcodeAnalysis(
            is_textual=False,
            layer_count=0,
            layer_detection_method=None,
            extrusion_move_count=0,
            object_extrusion_move_count=None,
            support_extrusion_move_count=None,
            support_interface_extrusion_move_count=None,
            unclassified_extrusion_move_count=0,
            support_layer_count=None,
            support_generation_result="unavailable",
            object_filament_length_mm=None,
            support_filament_length_mm=None,
            support_interface_filament_length_mm=None,
            estimated_print_time_seconds=None,
            filament_length_mm=None,
            filament_volume_cm3=None,
            filament_mass_g=None,
            warnings=[],
            failure_reason="empty G-code",
        )

    marker_count = 0
    extrusion_moves = 0
    object_extrusion_moves = 0
    support_extrusion_moves = 0
    support_interface_extrusion_moves = 0
    unclassified_extrusion_moves = 0
    object_filament_length = 0.0
    support_filament_length = 0.0
    support_interface_filament_length = 0.0
    support_layers: set[int] = set()
    role_comments_seen = False
    current_role: Optional[str] = None
    extrusion_absolute = True
    xyz_absolute = True
    e_position = 0.0
    xyz_position: dict[str, Optional[float]] = {"X": None, "Y": None, "Z": None}
    extrusion_z_values: list[float] = []
    warnings: list[str] = []

    for raw_line in text.splitlines():
        if LAYER_MARKER_RE.match(raw_line):
            marker_count += 1
        role_match = re.match(r"^\s*;\s*TYPE\s*:\s*(.+?)\s*$", raw_line, flags=re.IGNORECASE)
        if role_match:
            current_role = role_match.group(1).strip().lower()
            role_comments_seen = True
        if "warning" in raw_line.lower():
            warning = raw_line.strip()
            if warning and warning not in warnings:
                warnings.append(warning)

        code = raw_line.split(";", 1)[0].strip()
        command, params = _command_and_params(code)
        if command is None:
            continue
        if command == "M82":
            extrusion_absolute = True
            continue
        if command == "M83":
            extrusion_absolute = False
            continue
        if command == "G90":
            xyz_absolute = True
            continue
        if command == "G91":
            xyz_absolute = False
            continue
        if command == "G92":
            if "E" in params:
                e_position = params["E"]
            for axis in ("X", "Y", "Z"):
                if axis in params:
                    xyz_position[axis] = params[axis]
            continue
        if command not in {"G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03"}:
            continue

        previous_xyz = xyz_position.copy()
        for axis in ("X", "Y", "Z"):
            if axis in params:
                if xyz_absolute or xyz_position[axis] is None:
                    xyz_position[axis] = params[axis]
                else:
                    xyz_position[axis] = float(xyz_position[axis]) + params[axis]

        if "E" not in params:
            continue
        if extrusion_absolute:
            extrusion_delta = params["E"] - e_position
            e_position = params["E"]
        else:
            extrusion_delta = params["E"]
            e_position += extrusion_delta

        spatial_motion = any(
            axis in params
            and (
                previous_xyz[axis] is None
                or xyz_position[axis] is None
                or not math.isclose(float(previous_xyz[axis]), float(xyz_position[axis]), abs_tol=1e-9)
            )
            for axis in ("X", "Y", "Z")
        )
        if extrusion_delta > 1e-9 and spatial_motion:
            extrusion_moves += 1
            if xyz_position["Z"] is not None:
                extrusion_z_values.append(float(xyz_position["Z"]))
            if current_role is None or current_role in {"custom", "wipe tower", "mixed"}:
                unclassified_extrusion_moves += 1
            elif "support material interface" in current_role or current_role == "support interface":
                support_interface_extrusion_moves += 1
                support_interface_filament_length += extrusion_delta
                support_layers.add(marker_count)
            elif "support material" in current_role or current_role == "support":
                support_extrusion_moves += 1
                support_filament_length += extrusion_delta
                support_layers.add(marker_count)
            else:
                object_extrusion_moves += 1
                object_filament_length += extrusion_delta

    if marker_count:
        layer_count = marker_count
        layer_method: Optional[str] = "prusa_layer_markers"
    elif extrusion_z_values:
        distinct_z: list[float] = []
        for z_value in extrusion_z_values:
            if not any(math.isclose(z_value, known, abs_tol=1e-4) for known in distinct_z):
                distinct_z.append(z_value)
        layer_count = len(distinct_z)
        layer_method = "print_z_changes"
    else:
        layer_count = 0
        layer_method = None

    estimated_seconds = None
    time_match = re.search(r"^\s*;\s*estimated printing time[^=]*=\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    if time_match:
        estimated_seconds = parse_duration_seconds(time_match.group(1))

    placed_bbox_xy: Optional[tuple[float, float, float, float]] = None
    object_match = re.search(r"^\s*;\s*objects_info\s*=\s*(\{.+\})\s*$", text, re.MULTILINE)
    if object_match:
        try:
            objects = json.loads(object_match.group(1)).get("objects", [])
            points = [point for item in objects for point in item.get("polygon", []) if len(point) >= 2]
            if points:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                placed_bbox_xy = (min(xs), min(ys), max(xs), max(ys))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    classified_support_moves = support_extrusion_moves + support_interface_extrusion_moves
    if not role_comments_seen:
        support_result = "unavailable_no_extrusion_role_comments"
    elif classified_support_moves:
        support_result = "generated"
    else:
        support_result = "not_generated"

    return GcodeAnalysis(
        is_textual=True,
        layer_count=layer_count,
        layer_detection_method=layer_method,
        extrusion_move_count=extrusion_moves,
        object_extrusion_move_count=object_extrusion_moves if role_comments_seen else None,
        support_extrusion_move_count=support_extrusion_moves if role_comments_seen else None,
        support_interface_extrusion_move_count=support_interface_extrusion_moves if role_comments_seen else None,
        unclassified_extrusion_move_count=unclassified_extrusion_moves,
        support_layer_count=len(support_layers) if role_comments_seen else None,
        support_generation_result=support_result,
        object_filament_length_mm=object_filament_length if role_comments_seen else None,
        support_filament_length_mm=(support_filament_length + support_interface_filament_length) if role_comments_seen else None,
        support_interface_filament_length_mm=support_interface_filament_length if role_comments_seen else None,
        estimated_print_time_seconds=estimated_seconds,
        filament_length_mm=_parse_footer_value(text, "filament used", "mm"),
        filament_volume_cm3=_parse_footer_value(text, "filament used", "cm3"),
        filament_mass_g=_parse_footer_value(text, "filament used", "g"),
        warnings=warnings,
        placed_bbox_xy=placed_bbox_xy,
    )


def analyze_gcode_file(path: Path) -> GcodeAnalysis:
    if not path.exists():
        return replace(parse_gcode_text(""), failure_reason="G-code output file is missing")
    if path.stat().st_size == 0:
        return replace(parse_gcode_text(""), failure_reason="G-code output file is empty")
    data = path.read_bytes()
    if data.startswith(GCODE_BINARY_MAGIC) or b"\x00" in data:
        return replace(parse_gcode_text(""), failure_reason="binary G-code is not accepted; disable binary G-code in the exported profile")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return replace(parse_gcode_text(""), failure_reason="G-code is not valid UTF-8 text; binary G-code may be enabled")
    return parse_gcode_text(text)


def assess_toolpath(exit_code: Optional[int], gcode_path: Path) -> tuple[bool, GcodeAnalysis, str]:
    analysis = analyze_gcode_file(gcode_path)
    failures: list[str] = []
    if exit_code != 0:
        failures.append(f"PrusaSlicer exit code was {exit_code!r}, expected 0")
    if not gcode_path.exists() or gcode_path.stat().st_size == 0:
        failures.append(analysis.failure_reason or "nonempty G-code was not generated")
    elif not analysis.is_textual:
        failures.append(analysis.failure_reason or "G-code was not textual")
    if analysis.layer_count < 1:
        failures.append("no printed layer was detected")
    if analysis.extrusion_move_count < 1:
        failures.append("no motion with positive extrusion was detected")
    return not failures, analysis, "; ".join(dict.fromkeys(failures))


def get_slicer_version(executable: Path) -> tuple[Optional[str], Optional[str]]:
    if not executable.is_file():
        return None, f"PrusaSlicer executable not found: {executable}"
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"could not query PrusaSlicer version: {exc}"
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    version_match = re.search(r"PrusaSlicer[- ](\d+(?:\.\d+){1,3})", output, flags=re.IGNORECASE)
    if version_match:
        return f"PrusaSlicer {version_match.group(1)}", None
    if completed.returncode != 0 or not output:
        return None, f"PrusaSlicer --version failed with exit code {completed.returncode}"
    return output.splitlines()[0].strip(), None


def git_state(repo: Path) -> dict[str, Any]:
    def run_git(arguments: list[str]) -> Optional[str]:
        try:
            completed = subprocess.run(["git", *arguments], cwd=repo, check=False, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run_git(["rev-parse", "HEAD"])
    porcelain = run_git(["status", "--porcelain"])
    return {
        "git_commit_hash": commit,
        "git_dirty_working_tree": bool(porcelain) if porcelain is not None else None,
        "git_status_porcelain": porcelain.splitlines() if porcelain else [],
    }


def extract_warnings(stdout: str, stderr: str, gcode_warnings: Iterable[str]) -> list[str]:
    warnings = list(gcode_warnings)
    for line in itertools.chain(stdout.splitlines(), stderr.splitlines()):
        if "warning" in line.lower() and line.strip() not in warnings:
            warnings.append(line.strip())
    return warnings


def build_slicer_command(executable: Path, profile: Path, output_gcode: Path, stl: Path) -> list[str]:
    return [
        str(executable),
        "--load",
        str(profile),
        "--export-gcode",
        "--output",
        str(output_gcode),
        str(stl),
    ]


def stl_geometry(path: Path) -> dict[str, Any]:
    """Read STL bounds and verify complete binary-facet payload when possible."""
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    triangle_count: Optional[int] = None
    binary_complete: Optional[bool] = None
    if len(data) >= 84:
        candidate_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + candidate_count * 50
        if expected_size == len(data):
            triangle_count = candidate_count
            binary_complete = True
            for index in range(candidate_count):
                values = struct.unpack_from("<12fH", data, 84 + index * 50)
                vertices.extend((values[3:6], values[6:9], values[9:12]))
    if not vertices:
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            text = ""
        for match in re.finditer(rf"^\s*vertex\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})", text, re.MULTILINE | re.IGNORECASE):
            vertices.append(tuple(float(match.group(i)) for i in range(1, 4)))
        if vertices:
            triangle_count = len(vertices) // 3
            binary_complete = None
    if not vertices:
        return {"bounds": None, "extents": None, "triangle_count": triangle_count, "binary_complete": binary_complete}
    mins = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maxs = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return {
        "bounds": (mins, maxs),
        "extents": tuple(maxs[axis] - mins[axis] for axis in range(3)),
        "triangle_count": triangle_count,
        "binary_complete": binary_complete,
    }


def build_volume_dimensions(profile_metadata: dict[str, Optional[str]]) -> Optional[tuple[float, float, float]]:
    bed_shape = profile_metadata.get("bed_shape")
    max_height = profile_metadata.get("max_print_height")
    if not bed_shape or not max_height:
        return None
    points = re.findall(rf"({FLOAT_PATTERN})x({FLOAT_PATTERN})", bed_shape)
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return max(xs) - min(xs), max(ys) - min(ys), float(max_height.split(";")[0])


def _base_row(
    case_id: str,
    stl: Path,
    executable: Path,
    slicer_version: Optional[str],
    profile: Path,
    profile_hash: Optional[str],
    command: list[str],
    gcode_path: Path,
    stdout_log: Path,
    stderr_log: Path,
    profile_metadata: dict[str, Optional[str]],
) -> dict[str, Any]:
    geometry = stl_geometry(stl)
    extents = geometry["extents"]
    build_volume = build_volume_dimensions(profile_metadata)
    build_fit = bool(extents and build_volume and all(extents[index] <= build_volume[index] + 1e-6 for index in range(3))) if build_volume else None
    return {
        "case_id": case_id,
        "stl_path": str(stl),
        "stl_sha256": sha256_file(stl),
        "stl_size_bytes": stl.stat().st_size,
        "slicer_name": "PrusaSlicer",
        "slicer_version": slicer_version,
        "printer_profile_path": str(profile),
        "printer_profile_sha256": profile_hash,
        "command_text": shlex.join(command),
        "command_json": json.dumps(command),
        "start_timestamp_utc": None,
        "completion_timestamp_utc": None,
        "runtime_seconds": None,
        "process_exit_code": None,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "gcode_path": str(gcode_path),
        "gcode_exists": False,
        "gcode_is_textual": False,
        "gcode_sha256": None,
        "gcode_size_bytes": None,
        "layer_count": 0,
        "layer_detection_method": None,
        "extrusion_move_count": 0,
        "object_extrusion_move_count": None,
        "support_extrusion_move_count": None,
        "support_interface_extrusion_move_count": None,
        "unclassified_extrusion_move_count": 0,
        "support_layer_count": None,
        "support_generation_result": "unavailable",
        "object_filament_length_mm": None,
        "support_filament_length_mm": None,
        "support_interface_filament_length_mm": None,
        "estimated_print_time_seconds": None,
        "filament_length_mm": None,
        "filament_volume_cm3": None,
        "filament_mass_g": None,
        "object_scale_percent": 100.0,
        "stl_bbox_x_mm": extents[0] if extents else None,
        "stl_bbox_y_mm": extents[1] if extents else None,
        "stl_bbox_z_mm": extents[2] if extents else None,
        "stl_triangle_count": geometry["triangle_count"],
        "stl_binary_complete": geometry["binary_complete"],
        "placed_bbox_min_x_mm": None,
        "placed_bbox_min_y_mm": None,
        "placed_bbox_max_x_mm": None,
        "placed_bbox_max_y_mm": None,
        "placement_translation_x_mm": None,
        "placement_translation_y_mm": None,
        "placement_translation_z_mm": 0.0,
        "source_orientation_retained": True,
        "build_volume_fit": build_fit,
        "complete_stl_processed": False,
        "slicer_warnings_json": "[]",
        "toolpath_generation_success": False,
        "status": "pending",
        "failure_reason": None,
    }


def run_case(
    stl: Path,
    case_id: str,
    executable: Path,
    slicer_version: Optional[str],
    profile: Path,
    profile_hash: Optional[str],
    output_dir: Path,
    dry_run: bool,
    timeout_seconds: Optional[float],
    profile_metadata: dict[str, Optional[str]],
) -> dict[str, Any]:
    slug = safe_case_slug(case_id)
    gcode_path = output_dir / "gcode" / f"case_{slug}.gcode"
    stdout_log = output_dir / "logs" / f"case_{slug}.stdout.log"
    stderr_log = output_dir / "logs" / f"case_{slug}.stderr.log"
    command = build_slicer_command(executable, profile, gcode_path, stl)
    row = _base_row(case_id, stl, executable, slicer_version, profile, profile_hash, command, gcode_path, stdout_log, stderr_log, profile_metadata)

    if dry_run:
        timestamp = utc_now()
        row.update(
            status="dry_run",
            failure_reason="dry run only; PrusaSlicer was not executed",
            start_timestamp_utc=timestamp,
            completion_timestamp_utc=timestamp,
            runtime_seconds=0.0,
        )
        stdout_log.write_text("DRY RUN: command not executed\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return row

    prerequisites: list[str] = []
    if not executable.is_file():
        prerequisites.append(f"PrusaSlicer executable not found: {executable}")
    if not profile.is_file():
        prerequisites.append(f"printer profile not found: {profile}")
    if gcode_path.exists():
        prerequisites.append(f"refusing to overwrite existing G-code: {gcode_path}")
    if prerequisites:
        reason = "; ".join(prerequisites)
        timestamp = utc_now()
        row.update(
            status="prerequisite_failed",
            failure_reason=reason,
            start_timestamp_utc=timestamp,
            completion_timestamp_utc=timestamp,
            runtime_seconds=0.0,
        )
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(reason + "\n", encoding="utf-8")
        return row

    row["start_timestamp_utc"] = utc_now()
    started = time.monotonic()
    stdout = ""
    stderr = ""
    exit_code: Optional[int] = None
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += f"\nPrusaSlicer timed out after {timeout_seconds} seconds.\n"
    except OSError as exc:
        stderr = f"Could not execute PrusaSlicer: {exc}\n"
    runtime = time.monotonic() - started
    row["completion_timestamp_utc"] = utc_now()
    row["runtime_seconds"] = round(runtime, 6)
    row["process_exit_code"] = exit_code
    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")

    success, analysis, reason = assess_toolpath(exit_code, gcode_path)
    warnings = extract_warnings(stdout, stderr, analysis.warnings)
    row.update(
        gcode_exists=gcode_path.exists(),
        gcode_is_textual=analysis.is_textual,
        gcode_sha256=sha256_file(gcode_path) if gcode_path.exists() and analysis.is_textual else None,
        gcode_size_bytes=gcode_path.stat().st_size if gcode_path.exists() else None,
        layer_count=analysis.layer_count,
        layer_detection_method=analysis.layer_detection_method,
        extrusion_move_count=analysis.extrusion_move_count,
        object_extrusion_move_count=analysis.object_extrusion_move_count,
        support_extrusion_move_count=analysis.support_extrusion_move_count,
        support_interface_extrusion_move_count=analysis.support_interface_extrusion_move_count,
        unclassified_extrusion_move_count=analysis.unclassified_extrusion_move_count,
        support_layer_count=analysis.support_layer_count,
        support_generation_result=analysis.support_generation_result,
        object_filament_length_mm=analysis.object_filament_length_mm,
        support_filament_length_mm=analysis.support_filament_length_mm,
        support_interface_filament_length_mm=analysis.support_interface_filament_length_mm,
        estimated_print_time_seconds=analysis.estimated_print_time_seconds,
        filament_length_mm=analysis.filament_length_mm,
        filament_volume_cm3=analysis.filament_volume_cm3,
        filament_mass_g=analysis.filament_mass_g,
        slicer_warnings_json=json.dumps(warnings),
        toolpath_generation_success=success,
        status="success" if success else "failed",
        failure_reason=None if success else reason,
    )
    if analysis.placed_bbox_xy is not None:
        min_x, min_y, max_x, max_y = analysis.placed_bbox_xy
        geometry = stl_geometry(stl)
        source_min = geometry["bounds"][0] if geometry["bounds"] else (0.0, 0.0, 0.0)
        row.update(
            placed_bbox_min_x_mm=min_x,
            placed_bbox_min_y_mm=min_y,
            placed_bbox_max_x_mm=max_x,
            placed_bbox_max_y_mm=max_y,
            placement_translation_x_mm=min_x - source_min[0],
            placement_translation_y_mm=min_y - source_min[1],
            complete_stl_processed=bool(geometry["triangle_count"] and analysis.is_textual),
        )
    return row


def numeric_summary(values: Iterable[Optional[float]]) -> Optional[dict[str, float | int]]:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return None
    data.sort()
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


def aggregate_summary(
    rows: Sequence[dict[str, Any]],
    discovered_count: int,
    slicer_version: Optional[str],
    profile_hash: Optional[str],
    expected_cohort_size: Optional[int],
    git_info: dict[str, Any],
) -> dict[str, Any]:
    attempted_rows = [row for row in rows if row["status"] not in {"dry_run"}]
    successful_rows = [row for row in attempted_rows if row["toolpath_generation_success"] is True]
    failed_rows = [row for row in attempted_rows if row["toolpath_generation_success"] is not True]
    aggregate_runtime = sum(float(row["runtime_seconds"] or 0.0) for row in attempted_rows)
    if expected_cohort_size is not None and discovered_count == expected_cohort_size and len(attempted_rows) == expected_cohort_size:
        scope = "full_cohort"
        success_denominator = expected_cohort_size
        success_fraction = f"{len(successful_rows)}/{expected_cohort_size} intended cohort cases"
        success_percentage: Optional[float] = 100.0 * len(successful_rows) / expected_cohort_size
    else:
        scope = "representative_case" if discovered_count == 1 else "available_subset"
        success_denominator = len(attempted_rows) if attempted_rows else None
        descriptor = "available representative case" if discovered_count == 1 else "available cases"
        success_fraction = f"{len(successful_rows)}/{len(attempted_rows)} {descriptor}" if attempted_rows else f"0/0 {descriptor}"
        success_percentage = None
    return {
        "schema_version": SCHEMA_VERSION,
        "result_scope": scope,
        "expected_cohort_size": expected_cohort_size,
        "available_cases": discovered_count,
        "attempted_cases": len(attempted_rows),
        "successful_cases": len(successful_rows),
        "failed_cases_count": len(failed_rows),
        "success_fraction": success_fraction,
        "success_denominator": success_denominator,
        "number_discovered_stls": discovered_count,
        "number_attempted": len(attempted_rows),
        "number_successful": len(successful_rows),
        "number_failed": len(failed_rows),
        "success_percentage": success_percentage,
        "prusaslicer_version": slicer_version,
        "printer_profile_sha256": profile_hash,
        "aggregate_runtime_seconds": aggregate_runtime,
        "layer_count_summary": numeric_summary(row["layer_count"] for row in successful_rows),
        "estimated_print_time_seconds_summary": numeric_summary(row["estimated_print_time_seconds"] for row in successful_rows),
        "filament_length_mm_summary": numeric_summary(row["filament_length_mm"] for row in successful_rows),
        "filament_volume_cm3_summary": numeric_summary(row["filament_volume_cm3"] for row in successful_rows),
        "filament_mass_g_summary": numeric_summary(row["filament_mass_g"] for row in successful_rows),
        "failed_cases": [
            {"case_id": row["case_id"], "reason": row["failure_reason"]}
            for row in failed_rows
        ],
        **git_info,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_prusa_profile(path: Path) -> dict[str, Optional[str]]:
    """Read only unambiguous flat key/value settings from an exported config."""
    wanted = {
        "printer_model",
        "printer_settings_id",
        "nozzle_diameter",
        "filament_type",
        "filament_settings_id",
        "layer_height",
        "perimeters",
        "fill_density",
        "brim_width",
        "support_material",
        "support_material_style",
        "binary_gcode",
        "support_material_auto",
        "support_material_buildplate_only",
        "print_settings_id",
        "bed_shape",
        "max_print_height",
    }
    values: dict[str, Optional[str]] = {key: None for key in sorted(wanted)}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in wanted and value:
            values[key] = value
    return values


def read_prusa_profile_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        fields[key] = value
    return fields


def profile_field_diff(source: Path, derived: Path) -> dict[str, dict[str, Optional[str]]]:
    source_fields = read_prusa_profile_fields(source)
    derived_fields = read_prusa_profile_fields(derived)
    return {
        key: {"source": source_fields.get(key), "derived": derived_fields.get(key)}
        for key in sorted(source_fields.keys() | derived_fields.keys())
        if source_fields.get(key) != derived_fields.get(key)
    }


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_seconds(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} s"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def generate_publication_artifacts(
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    profile_metadata: dict[str, Optional[str]],
    output_dir: Path,
) -> bool:
    successful = [row for row in rows if row["toolpath_generation_success"] is True]
    eligible = (
        bool(successful)
        and summary["number_failed"] == 0
        and bool(summary.get("prusaslicer_version"))
        and bool(summary.get("printer_profile_sha256"))
    )
    if not eligible:
        return False

    version = latex_escape(summary["prusaslicer_version"])
    profile_hash = latex_escape(summary["printer_profile_sha256"])
    configuration_parts: list[str] = []
    printer = profile_metadata.get("printer_model") or profile_metadata.get("printer_settings_id")
    if printer:
        printer_name = "Original Prusa MK4S" if printer == "MK4S" else printer
        configuration_parts.append(latex_escape(printer_name))
    nozzle = profile_metadata.get("nozzle_diameter")
    if nozzle:
        configuration_parts.append(f"{latex_escape(nozzle.split(';')[0])} mm nozzle")
    filament = profile_metadata.get("filament_type")
    if filament:
        configuration_parts.append(latex_escape(filament.split(';')[0]))
    configuration = ", ".join(configuration_parts)
    config_clause = f" ({configuration})" if configuration else ""
    support_clause = ""
    if profile_metadata.get("support_material_auto") == "1" and profile_metadata.get("support_material_style") == "organic":
        support_clause = " Automatic organic supports were permitted everywhere."
    methods = (
        f"A reference-profile production-slicer evaluation processed repaired Phase B STL outputs using {version} with a fixed exported printer configuration{config_clause} "
        f"(configuration SHA-256: \\texttt{{{profile_hash}}}). The Original Prusa MK4S served only as a standardized computational reference configuration; no physical printer or physical fabrication was used. Toolpath-generation success was predefined as a zero slicer exit "
        "status, generation of a nonempty textual G-code file, detection of at least one printed layer, and the presence of "
        f"positive extrusion movements. Native 100\\% scale and the source STL orientation were retained.{support_clause} The slicer version, configuration, hashes, execution logs, and per-case results were retained.\n"
    )

    if summary["result_scope"] == "full_cohort":
        layers = summary["layer_count_summary"]
        results = (
            f"Production-slicer validation evaluated {summary['number_attempted']} repaired STL files. "
            f"Executable textual toolpaths were generated for {summary['number_successful']} files "
            f"({summary['success_percentage']:.1f}\\%), with {summary['number_failed']} failures. "
            f"The median layer count was {layers['median']:.0f} (interquartile range {layers['q1']:.0f}--{layers['q3']:.0f})."
        )
        table_rows = [
            ("Repaired STLs evaluated", summary["number_attempted"]),
            ("Successful toolpaths", summary["number_successful"]),
            ("Success percentage", f"{summary['success_percentage']:.1f}\\%"),
            ("Failed toolpaths", summary["number_failed"]),
            ("Layers, median (IQR)", f"{layers['median']:.0f} ({layers['q1']:.0f}--{layers['q3']:.0f})"),
        ]
        time_summary = summary.get("estimated_print_time_seconds_summary")
        if time_summary:
            table_rows.append(("Estimated print time, median (IQR)", f"{_format_seconds(time_summary['median'])} ({_format_seconds(time_summary['q1'])}--{_format_seconds(time_summary['q3'])})"))
        filament_summary = summary.get("filament_length_mm_summary")
        if filament_summary:
            table_rows.append(("Filament length, median (IQR)", f"{filament_summary['median']:.1f} mm ({filament_summary['q1']:.1f}--{filament_summary['q3']:.1f})"))
        caption = "Full-cohort production-slicer validation of repaired Phase B STL outputs."
    else:
        row = successful[0]
        result_parts = [
            f"The representative case-{latex_escape(row['case_id'])} computational toolpath-generation assessment generated nonempty textual G-code with {row['layer_count']} detected layers and {row['object_extrusion_move_count']} positive object-extrusion movements."
        ]
        if row["estimated_print_time_seconds"] is not None:
            result_parts.append(f"PrusaSlicer reported an estimated print time of {_format_seconds(row['estimated_print_time_seconds'])}.")
        if row["filament_length_mm"] is not None:
            material = f"Reported filament usage was {row['filament_length_mm']:.1f} mm"
            if row["filament_volume_cm3"] is not None:
                material += f" ({row['filament_volume_cm3']:.2f} cm$^3$"
                if row["filament_mass_g"] is not None:
                    material += f", {row['filament_mass_g']:.2f} g"
                material += ")"
            result_parts.append(material + ".")
        if row["support_generation_result"] == "generated":
            result_parts.append(
                f"Organic support paths were detected on {row['support_layer_count']} layers, comprising "
                f"{row['support_extrusion_move_count']} support-material and {row['support_interface_extrusion_move_count']} support-interface extrusion movements."
            )
        results = " ".join(result_parts)
        table_rows = [
            ("Case ID", latex_escape(row["case_id"])),
            ("Slicer version", latex_escape(row["slicer_version"])),
            ("Reference printer", "Original Prusa MK4S" if profile_metadata.get("printer_model") == "MK4S" else latex_escape(profile_metadata.get("printer_model") or "unavailable")),
            ("Nozzle diameter", f"{latex_escape(profile_metadata.get('nozzle_diameter') or 'unavailable')} mm"),
            ("Layer height / preset", f"{latex_escape(profile_metadata.get('layer_height') or 'unavailable')} mm / {latex_escape(profile_metadata.get('print_settings_id') or 'unavailable')}"),
            ("Scale", f"{float(row['object_scale_percent']):.0f}\\%"),
            ("Bounding dimensions", f"{float(row['stl_bbox_x_mm']):.2f} $\\times$ {float(row['stl_bbox_y_mm']):.2f} $\\times$ {float(row['stl_bbox_z_mm']):.2f} mm"),
            ("Toolpath-generation result", "Successful"),
            ("Detected layers", row["layer_count"]),
            ("Positive object-extrusion movements", row["object_extrusion_move_count"]),
            ("Support generation", latex_escape(row["support_generation_result"])),
        ]
        if row["estimated_print_time_seconds"] is not None:
            table_rows.append(("Estimated print time", _format_seconds(row["estimated_print_time_seconds"])))
        if row["filament_mass_g"] is not None:
            table_rows.append(("Estimated filament mass", f"{row['filament_mass_g']:.2f} g"))
        caption = "Representative reference-profile computational toolpath assessment for repaired case 720."

    table_body = "\n".join(f"{latex_escape(label)} & {value} \\\\" for label, value in table_rows)
    table = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{tab:production-slicer-validation}}\n"
        "\\begin{tabular}{ll}\n\\toprule\nMetric & Result " + r"\\" + "\n\\midrule\n"
        f"{table_body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    discussion = (
        "The reference-profile production-slicer evaluation provides supporting computational evidence that the repaired geometry can be converted into executable "
        "toolpaths under a fixed configuration; the clDice--mesh-fragmentation relationship remains the principal scientific finding.\n"
    )
    limitations = (
        "This computational toolpath-generation assessment does not establish physical fabrication quality, quantitative surface fidelity, lumen patency, "
        "bioink-specific resolution, perfusion, mechanical behavior, cell viability, endothelialization, or biological function.\n"
    )
    conclusion = (
        "The representative case-720 assessment supports computational continuity from a repaired Phase B STL to an "
        "executable toolpath under a standardized reference profile; no physical printer or physical fabrication was used.\n"
        if summary["result_scope"] != "full_cohort"
        else "Full-cohort production-slicer validation supports digital continuity from repaired Phase B STLs to executable toolpaths without establishing biological bioprintability.\n"
    )
    artifacts = {
        "slicer_validation_table.tex": table,
        "manuscript_methods_snippet.tex": methods,
        "manuscript_results_snippet.tex": results + "\n",
        "manuscript_discussion_snippet.tex": discussion,
        "manuscript_limitations_snippet.tex": limitations,
        "manuscript_conclusion_snippet.tex": conclusion,
    }
    for filename, content in artifacts.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    return True


def publication_artifact_status(summary: dict[str, Any], generated: bool, output_dir: Path) -> None:
    eligible = (
        summary["number_successful"] > 0
        and summary["number_failed"] == 0
        and bool(summary.get("prusaslicer_version"))
        and bool(summary.get("printer_profile_sha256"))
    )
    status = {
        "eligible_for_verified_slicer_snippets": eligible,
        "generated": generated,
        "reason": (
            "Verified successful production-slicer results are required before numerical manuscript snippets are generated."
            if not eligible
            else "Snippets were generated only from verified result rows."
        ),
    }
    write_json(output_dir / "publication_artifact_status.json", status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real PrusaSlicer production-slicer validation on repaired Phase B STL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stl", action="append", type=Path, default=[], help="Explicit repaired STL path; may be repeated.")
    parser.add_argument("--input-glob", action="append", default=[], help="Glob for repaired STL discovery; may be repeated.")
    parser.add_argument("--prusaslicer", required=True, type=Path, help="Path to the PrusaSlicer executable.")
    parser.add_argument("--profile", required=True, type=Path, help="Fixed exported PrusaSlicer .ini profile with binary G-code disabled.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/production_slicer_validation"), help="Validation output directory.")
    parser.add_argument("--case-filter", help="Process only the exact extracted case ID.")
    parser.add_argument("--expected-cohort-size", type=int, default=None, help="Expected repaired-STL cohort size used only to label full-cohort versus subset scope.")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Optional per-case slicer timeout.")
    parser.add_argument("--dry-run", action="store_true", help="Discover inputs and write planned commands without invoking PrusaSlicer.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_globs = args.input_glob or [DEFAULT_INPUT_GLOB]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "gcode").mkdir(exist_ok=True)
    (output_dir / "representative_case").mkdir(exist_ok=True)

    repo = Path(__file__).resolve().parent
    git_info = git_state(repo)
    stls = discover_stls(args.stl, input_globs, args.case_filter)
    executable = args.prusaslicer.expanduser().resolve()
    profile = args.profile.expanduser().resolve()
    slicer_version, version_error = get_slicer_version(executable)
    profile_hash = sha256_file(profile) if profile.is_file() else None
    profile_metadata = parse_prusa_profile(profile)

    rows: list[dict[str, Any]] = []
    used_case_ids: dict[str, int] = {}
    for stl in stls:
        extracted = extract_case_id(stl)
        used_case_ids[extracted] = used_case_ids.get(extracted, 0) + 1
        case_id = extracted if used_case_ids[extracted] == 1 else f"{extracted}_{used_case_ids[extracted]}"
        rows.append(
            run_case(
                stl=stl,
                case_id=case_id,
                executable=executable,
                slicer_version=slicer_version,
                profile=profile,
                profile_hash=profile_hash,
                output_dir=output_dir,
                dry_run=args.dry_run,
                timeout_seconds=args.timeout_seconds,
                profile_metadata=profile_metadata,
            )
        )

    summary = aggregate_summary(
        rows,
        discovered_count=len(stls),
        slicer_version=slicer_version,
        profile_hash=profile_hash,
        expected_cohort_size=args.expected_cohort_size,
        git_info=git_info,
    )
    summary["printer_profile_metadata"] = profile_metadata
    run_command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "validator_script": str(Path(__file__).resolve()),
        "validator_script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "created_utc": utc_now(),
        "dry_run": args.dry_run,
        "input_globs": input_globs,
        "explicit_stls": [str(path.expanduser().resolve()) for path in args.stl],
        "case_filter": args.case_filter,
        "discovered_inputs": [
            {"case_id": extract_case_id(path), "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in stls
        ],
        "prusaslicer_executable": str(executable),
        "prusaslicer_version": slicer_version,
        "prusaslicer_version_error": version_error,
        "printer_profile": str(profile),
        "printer_profile_sha256": profile_hash,
        "success_definition": [
            "PrusaSlicer exit code is zero",
            "a nonempty textual G-code file is generated",
            "at least one printed layer is detected",
            "at least one motion with positive extrusion is detected",
        ],
        **git_info,
    }
    write_csv(output_dir / "per_case_slicer_results.csv", rows)
    write_json(output_dir / "summary_slicer_results.json", summary)
    write_json(output_dir / "validation_manifest.json", manifest)
    (output_dir / "slicer_version.txt").write_text((slicer_version or f"UNAVAILABLE: {version_error}") + "\n", encoding="utf-8")
    (output_dir / "run_command.txt").write_text(run_command + "\n", encoding="utf-8")
    generated = generate_publication_artifacts(summary, rows, profile_metadata, output_dir)
    publication_artifact_status(summary, generated, output_dir)

    if not stls:
        print("No repaired STL files were discovered.", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Dry run complete: discovered {len(stls)} STL file(s); PrusaSlicer was not executed.")
        return 0
    return 0 if summary["number_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
