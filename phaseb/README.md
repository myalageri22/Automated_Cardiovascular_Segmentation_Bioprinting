# Phase B: Postprocessing, Meshing, and QC

Phase B converts vascular segmentations (probability maps or masks) into fabrication-oriented meshes and generates computational QC reports. It consumes a CT volume and a segmentation volume, cleans and thresholds the mask, extracts meshes in real-world millimeters, repairs them, and produces human-readable and machine-readable QC artifacts. These outputs do not by themselves establish physical printability.

## Inputs
- CT volume: DICOM directory or NIfTI (.nii/.nii.gz).
- Segmentation: preferred probability map float [0,1] as NIfTI; binary mask is accepted.
- Optional seed point (x,y,z in mm) to keep the component connected to the seed.

## Outputs (per case)
```
/workspace/cloud_bundle/phaseb_outputs/<case_id>/
  vessels_raw.stl          # marching cubes mesh (mm)
  vessels_repaired.stl     # repaired mesh
  qc_report.json           # structured QC + printability metrics
  qc_summary.txt           # human-readable summary
  qc_summary.csv           # one-line summary (append-friendly)
  used_config.yaml         # frozen parameters for reproducibility
  previews/                # overlays and optional mesh render (when enabled)
```

## Quickstart
Single case (probability map, local paths):
```bash
python -m phaseb.src.phaseb.cli \
  --ct /path/to/ct.nii.gz \
  --seg /path/to/seg_prob.nii.gz \
  --case-id sample \
  --outdir ./phaseb_outputs \
  --threshold-sweep
```

Binary mask (no sweep):
```bash
python -m phaseb.src.phaseb.cli --ct ct.nii.gz --seg seg_mask.nii.gz --seg-type mask --case-id sample
```

Batch (CSV manifest with columns: case_id, ct_path, seg_path, seg_type optional):
```bash
python -m phaseb.src.phaseb.cli --batch-manifest manifest.csv --outdir /workspace/cloud_bundle/phaseb_outputs
```

Local auto-discovery of `<id>.img.nii.gz` + `<id>.label.nii.gz` pairs:
```bash
python phaseb/scripts/run_pairs_local.py \
  --root data/processed/all/1-200 \
  --outdir ./phaseb_outputs \
  --generate-previews
```

Wrapper:
```bash
python run_phaseb.py --ct ... --seg ...
```

## Configuration
Parameters live in `phaseb/configs/default.yaml` and are saved per run to `used_config.yaml` for reproducibility. CLI flags override config values (threshold, resample mm, min component volume, min radius, previews, etc.).

Key knobs:
- `segmentation.default_threshold`: used when no sweep is requested.
- `segmentation.threshold_candidates`: list for sweep (0.2–0.8 by default).
- `postprocess`: hole filling, conservative closing radius (mm), min component volume in mm³, keep mode (`largest` or `seed`).
- `resample.target_spacing_mm`: optional isotropic resampling.
- `qc.min_radius_mm`: printability threshold for distance-transform radii.

## QC interpretation
- **Mask metrics**: voxel count, physical volume, component count, largest component ratio.
- **Topology**: component count after cleanup, skeleton endpoint proxy.
- **Mesh metrics**: surface area, physical bounding box, watertight flag, Euler number, component count.
- **Printability**: radius stats (min/p5/p50 mm) from distance transform; percent below `min_radius_mm` highlights fragile vessels.
- **Robustness**: re-threshold at ±0.05 to report volume/component/watertight deltas.

`qc_summary.txt` highlights pass/fail with reasons (empty mask, many components, small-radius burden). `qc_report.json` contains the full schema for judging/automation.

## Common issues & troubleshooting
- **Empty or tiny mask**: check segmentation path/type; adjust threshold or lower `min_component_mm3`.
- **Mesh not watertight**: ensure probability map quality; try enabling `pymeshfix` or lowering closing radius to avoid gaps.
- **Vessel breakage**: reduce closing radius, ensure resampling spacing is not too coarse, consider seed-based keeping to avoid losing branches.
- **Orientation/spacing mismatch**: mesh vertices use the complete SimpleITK physical transform (direction, origin, and spacing); ensure segmentation aligns with CT before running Phase B. Mesh repair must not recenter the result.

## Tests
`pytest phaseb/tests` runs small synthetic checks for IO, postprocessing invariants, physical scaling, and mesh/QC smoke coverage.
