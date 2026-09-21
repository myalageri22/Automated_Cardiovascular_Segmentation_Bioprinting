#!/usr/bin/env python3
"""Secondary, software-level toolpath check on the topology-correction meshes.

Every repaired STL of the control (S0) and of each primary correction variant is
run through one fixed slicer profile. The endpoint is whether the slicer can
produce a complete toolpath from the reconstructed surface, and what it reports
while doing so -- warnings, layer count, layers that carry no extrusion. This is
a software-level property of the geometry. It is NOT evidence of physical
printability and must never be described as such.

The profile is fixed for all cases and all variants; it is written into the
run manifest. No per-case tuning, no retries with different settings.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE: Dict[str, str] = {
    "layer-height": "0.2",
    "first-layer-height": "0.2",
    "nozzle-diameter": "0.4",
    "filament-diameter": "1.75",
    "temperature": "210",
    "first-layer-temperature": "215",
    "bed-temperature": "60",
    "perimeters": "2",
    "fill-density": "15%",
    "bed-shape": "0x0,250x0,250x210,0x210",
    "max-print-height": "210",
    "center": "125,105",
    "gcode-flavor": "marlin",
    "filament-density": "1.24",
}

TIME_RE = re.compile(r";\s*estimated printing time \(normal mode\)\s*=\s*(.+)")
FILA_G_RE = re.compile(r";\s*(?:total )?filament used \[g\]\s*=\s*([0-9.]+)")
FILA_MM_RE = re.compile(r";\s*filament used \[mm\]\s*=\s*([0-9.]+)")
MAXZ_RE = re.compile(r";\s*max_print_height\s*=\s*([0-9.]+)")
EXTRUDE_RE = re.compile(r"^G1 .* E-?[0-9.]+", re.M)


def _parse_time_to_min(txt: str) -> Optional[float]:
    if not txt:
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+)([dhms])", txt):
        total += float(value) * {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}[unit]
    return total or None


def parse_gcode(path: Path) -> Dict[str, Any]:
    """Layer statistics and slicer estimates from a finished G-code file."""
    out: Dict[str, Any] = {"gcode_bytes": path.stat().st_size}
    layers_total = 0
    layers_without_extrusion = 0
    in_layer = False
    layer_has_extrusion = False
    tail: List[str] = []
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            if line.startswith(";LAYER_CHANGE"):
                if in_layer:
                    layers_total += 1
                    layers_without_extrusion += 0 if layer_has_extrusion else 1
                in_layer, layer_has_extrusion = True, False
                continue
            if in_layer and line.startswith("G1 ") and " E" in line:
                # a retraction is an E move with no X/Y/Z: not material laid down
                if ("X" in line or "Y" in line) and not re.search(r"E-", line):
                    layer_has_extrusion = True
            if line.startswith(";"):
                tail.append(line)
                if len(tail) > 3000:
                    tail = tail[-3000:]
    if in_layer:
        layers_total += 1
        layers_without_extrusion += 0 if layer_has_extrusion else 1
    out["layer_count"] = layers_total
    out["empty_layer_count"] = layers_without_extrusion
    blob = "".join(tail)
    m = TIME_RE.search(blob)
    out["estimated_print_time_text"] = m.group(1).strip() if m else None
    out["estimated_print_time_min"] = _parse_time_to_min(m.group(1)) if m else None
    m = FILA_G_RE.search(blob)
    out["filament_used_g"] = float(m.group(1)) if m else None
    m = FILA_MM_RE.search(blob)
    out["filament_used_mm"] = float(m.group(1)) if m else None
    return out


WARN_MARKERS = (
    "Detected print stability issues",
    "Floating object part",
    "Low bed adhesion",
    "Loose extrusions",
    "Consider enabling supports",
    "Consider enabling brim",
    "is not manifold",
    "auto-repair",
    "Repaired",
    "WARNING",
)


def slice_one(slicer: str, stl: Path, variant: str, case: str,
              keep_gcode_dir: Optional[Path]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case_id": case, "variant_id": variant, "stl": str(stl),
                           "stl_bytes": stl.stat().st_size}
    tmp = Path(tempfile.mkdtemp(prefix="slice_"))
    gcode = tmp / f"{case}__{variant}.gcode"
    cmd = [slicer, "--export-gcode", "--output", str(gcode)]
    for k, v in PROFILE.items():
        cmd += [f"--{k}"] if v == "" else [f"--{k}", v]
    cmd.append(str(stl))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        rc, so, se = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rc, so, se = -1, "", "TIMEOUT after 900 s"
    row["slice_seconds"] = round(time.time() - t0, 2)
    row["return_code"] = rc
    text = f"{so}\n{se}"
    row["toolpath_success"] = bool(rc == 0 and gcode.exists() and gcode.stat().st_size > 0)
    row["gcode_generated"] = bool(gcode.exists() and gcode.stat().st_size > 0)
    hits = sorted({m for m in WARN_MARKERS if m.lower() in text.lower()})
    row["warning_count"] = len(hits)
    row["warnings"] = "; ".join(hits) if hits else ""
    row["stderr_tail"] = se.strip().replace("\n", " | ")[-500:] if se.strip() else ""
    if row["toolpath_success"]:
        try:
            row.update(parse_gcode(gcode))
        except Exception as exc:                      # parsing must not fail a case
            row["parse_error"] = repr(exc)
        if keep_gcode_dir is not None:
            keep_gcode_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gcode, keep_gcode_dir / gcode.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slicer", required=True)
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--keep-gcode-cases", nargs="*", default=[],
                    help="case ids whose G-code to retain as samples")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--profile-json", default=None,
                    help="JSON file of extra/overriding slicer options. The merged "
                         "profile is written to the run's toolpath_profile.json.")
    ap.add_argument("--profile-name", default="P1_no_supports")
    args = ap.parse_args(argv)

    if args.profile_json:
        PROFILE.update(json.loads(Path(args.profile_json).read_text()))
    mesh_root = Path(args.mesh_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants or sorted(d.name for d in mesh_root.iterdir() if d.is_dir())

    jobs = []
    for v in variants:
        for stl in sorted((mesh_root / v).glob("case_outputs/*/segmentation_repaired.stl")):
            case = stl.parent.name.split("__")[0]
            keep = out_dir / "sample_gcode" if case in set(args.keep_gcode_cases) else None
            jobs.append((stl, v, case, keep))
    if args.limit:
        jobs = jobs[: args.limit]

    ver = subprocess.run([args.slicer, "--help"], capture_output=True, text=True
                         ).stdout.splitlines()[0].strip()
    print(f"slicer: {ver}\njobs: {len(jobs)}  workers: {args.workers}", flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(slice_one, args.slicer, s, v, c, k): (c, v)
                for s, v, c, k in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rows.append(fut.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  elapsed {time.time()-t0:.0f}s", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["variant_id", "case_id"])
    df["profile_name"] = args.profile_name
    df.to_csv(out_dir / "per_case_toolpath.csv", index=False)
    (out_dir / "toolpath_profile.json").write_text(json.dumps(
        {"slicer_version_line": ver, "profile_name": args.profile_name, "profile": PROFILE,
         "command_template": "prusa-slicer --export-gcode --output <out> "
                             + " ".join((f"--{k}" if v == "" else f"--{k} {v}") for k, v in PROFILE.items()) + " <stl>",
         "n_jobs": len(jobs), "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(df.groupby("variant_id")["toolpath_success"].agg(["sum", "count"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
