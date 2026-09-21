"""CLI entrypoint for Phase B."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import pipeline
from .io import SegmentationType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase B: postprocessing, meshing, and QC")
    parser.add_argument("--ct", required=False, help="Path to CT volume (DICOM folder or NIfTI)")
    parser.add_argument("--seg", required=False, help="Path to segmentation (probability map or mask)")
    parser.add_argument("--seg-type", default=SegmentationType.AUTO, choices=[SegmentationType.PROBABILITY, SegmentationType.MASK, SegmentationType.AUTO], help="Segmentation type")
    parser.add_argument("--case-id", default=None, help="Case identifier")
    parser.add_argument("--outdir", default=str(pipeline.DEFAULT_OUTDIR), help="Output directory root")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold for probability maps")
    parser.add_argument("--threshold-sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--resample-mm", type=float, default=None, help="Optional isotropic resampling spacing (mm)")
    parser.add_argument("--min-component-mm3", type=float, default=None, help="Minimum component volume to keep (mm^3)")
    parser.add_argument("--min-radius-mm", type=float, default=None, help="Minimum printable radius (mm)")
    parser.add_argument("--generate-previews", action="store_true", help="Generate overlay and mesh previews")
    parser.add_argument("--batch-manifest", default=None, help="CSV or JSON manifest for batch processing")
    parser.add_argument("--seed-point", default=None, help="Seed point in mm (x,y,z) to keep connected component")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = pipeline.load_config(args.config)
    outdir = Path(args.outdir)

    if args.batch_manifest:
        reports = pipeline.run_batch(
            manifest_path=args.batch_manifest,
            outdir=outdir,
            config=config,
            threshold_override=args.threshold,
            threshold_sweep=args.threshold_sweep,
            resample_mm=args.resample_mm,
            min_component_mm3=args.min_component_mm3,
            min_radius_mm=args.min_radius_mm,
            generate_previews=args.generate_previews,
        )
        parser.exit(message=f"Processed {len(reports)} cases. Outputs in {outdir}\n")

    if not args.ct or not args.seg:
        parser.print_help()
        parser.exit(1, "\nPlease provide --ct and --seg (or use --batch-manifest for batch mode).\n")

    qc_report = pipeline.run_case(
        ct_path=args.ct,
        seg_path=args.seg,
        seg_type=args.seg_type,
        case_id=args.case_id,
        outdir=outdir,
        config=config,
        threshold_override=args.threshold,
        threshold_sweep=args.threshold_sweep,
        resample_mm=args.resample_mm,
        min_component_mm3=args.min_component_mm3,
        min_radius_mm=args.min_radius_mm,
        generate_previews=args.generate_previews,
        seed_point=args.seed_point,
    )
    parser.exit(message=f"Finished case {qc_report.get('case_id')} -> {outdir}\n")


if __name__ == "__main__":
    main()
