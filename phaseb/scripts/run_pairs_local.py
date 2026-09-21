#!/usr/bin/env python
"""Batch runner that auto-discovers <id>.img.nii.gz / <id>.label.nii.gz pairs locally."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from phaseb.src.phaseb import pipeline, report


def discover_pairs(root: Path) -> List[tuple[str, Path, Path]]:
    pairs = []
    for img_path in sorted(root.glob("*.img.nii.gz")):
        cid = img_path.name.replace(".img.nii.gz", "")
        label_path = root / f"{cid}.label.nii.gz"
        if not label_path.exists():
            print(f"[WARN] Missing label for {cid}, skipping")
            continue
        pairs.append((cid, img_path, label_path))
    return pairs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase B over img/label pairs in a directory")
    parser.add_argument("--root", default="data/processed/all/1-200", help="Root folder containing <id>.img.nii.gz and <id>.label.nii.gz")
    parser.add_argument("--outdir", default="phaseb_outputs", help="Output directory root")
    parser.add_argument("--seg-type", default="mask", choices=["mask", "prob", "auto"], help="Segmentation type")
    parser.add_argument("--generate-previews", action="store_true", help="Generate overlays and mesh previews")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of cases")
    args = parser.parse_args(argv)

    root = Path(args.root)
    outdir = Path(args.outdir)
    cases = discover_pairs(root)
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        parser.exit(message="No valid img/label pairs found.\n")

    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = pipeline.load_config(str(config_path))

    rows = []
    for cid, ct_path, seg_path in cases:
        print(f"[INFO] Processing {cid}")
        qc_report = pipeline.run_case(
            ct_path=str(ct_path),
            seg_path=str(seg_path),
            seg_type=args.seg_type,
            case_id=cid,
            outdir=outdir,
            config=config,
            threshold_override=None,
            threshold_sweep=False,
            resample_mm=None,
            min_component_mm3=None,
            min_radius_mm=None,
            generate_previews=args.generate_previews,
            seed_point=None,
        )
        pf, _ = report._pass_fail(qc_report)  # type: ignore[attr-defined]
        rows.append(
            {
                "case_id": cid,
                "volume_mm3": qc_report.get("mask_metrics", {}).get("volume_mm3"),
                "n_components": qc_report.get("mask_metrics", {}).get("component_count"),
                "is_watertight": qc_report.get("mesh_repaired", {}).get("is_watertight"),
                "min_radius_mm": qc_report.get("printability", {}).get("min_radius_mm"),
                "p5_radius_mm": qc_report.get("printability", {}).get("p5_radius_mm"),
                "pass_fail": pf,
            }
        )

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "batch_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    parser.exit(message=f"Processed {len(rows)} cases. Summary -> {summary_path}\n")


if __name__ == "__main__":
    main()
