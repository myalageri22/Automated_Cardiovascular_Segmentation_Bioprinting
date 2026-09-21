# Cloud Bundle: 3D Vascular Segmentation + Mesh Pipeline

End-to-end research pipeline for vascular segmentation in CT volumes and conversion into mesh/STL artifacts for downstream review and bioprintability checks.

This repo combines:
- Phase A: MONAI/PyTorch 3D U-Net training + inference
- Phase B: segmentation cleanup, meshing, repair, and QC reporting

## Highlights:

- 3D vessel segmentation training pipeline (`train_vascular.py`)
- Inference + postprocessing runner (`run_full_pipeline.py`)
- Standalone Phase B CLI (`run_phaseb.py` / `phaseb/src/phaseb/cli.py`)
- Mesh generation + repair (`vessels_raw.stl`, `vessels_repaired.stl`)
- QC artifacts (`qc_report.json`, `qc_summary.txt`, `qc_summary.csv`)
- Small test suite for Phase B (`phaseb/tests`)

## Repository Layout:

```text
.
├── run_full_pipeline.py          # Phase A inference -> Phase B pipeline
├── run_phaseb.py                 # Convenience wrapper for Phase B CLI
├── train_vascular.py         # Main training script (Phase A)
├── requirements.txt              # Core dependencies
├── requirements_phaseb.txt       # Additional Phase B dependencies
├── checkpoints/                  # Model checkpoints + split snapshots
├── logs/                         # Training logs
├── phaseb/
│   ├── configs/default.yaml
│   ├── scripts/
│   ├── src/phaseb/               # Phase B package
│   └── tests/
└── phaseb_outputs/               # Example generated outputs
```

## Environment Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_phaseb.txt
```

## Quick Start:

### 1) Run Phase B only (from CT + segmentation):

```bash
python run_phaseb.py \
  --ct /path/to/ct.nii.gz \
  --seg /path/to/seg_prob_or_mask.nii.gz \
  --seg-type auto \
  --case-id demo \
  --outdir ./phaseb_outputs \
  --threshold-sweep
```

### 2) Run full pipeline (Phase A inference -> Phase B):

```bash
python run_full_pipeline.py \
  --ct /path/to/ct.nii.gz \
  --checkpoint checkpoints/checkpoint_best.pt \
  --outdir pipeline_outputs \
  --case-id demo \
  --phaseb-threshold-sweep \
  --phaseb-previews
```

## Training (Phase A):

Example:

```bash
python "train_vascular.py" \
  --dataset_preset imagecas \
  --imagecas_root /path/to/data \
  --epochs 100 \
  --batch_size 1 \
  --roi_size 96,192,192 \
  --experiment_name bioprint_v1
```

Useful sanity modes:
- `--dry_run`: validates forward/backward pass and validation inference
- `--overfit_one`: overfit single case for debugging
- `--verify_data`: transform/data pipeline checks

## Inputs and Outputs:

### Supported inputs:
- CT: NIfTI (`.nii/.nii.gz`) or DICOM directory
- Segmentation for Phase B:
  - probability map (`float`, preferred), or
  - binary mask

### Typical output structure:

```text
pipeline_outputs/<case_id>/
├── phasea/
│   ├── ct_preprocessed.nii.gz
│   ├── seg_prob.nii.gz
│   └── seg_mask.nii.gz          
├── vessels_raw.stl
├── vessels_repaired.stl
├── qc_report.json
├── qc_summary.txt
├── qc_summary.csv
└── used_config.yaml
```

## Testing:

```bash
pytest phaseb/tests
```

## Corrected Phase B cohort checks

The retained lightweight cohort artifacts are:

- `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv`: repair, mesh integrity, and connected-component results.
- `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv`: 50-plane slicability, centroid displacement, and affine-aware world-coordinate bounding-box alignment.
- `outputs/phase_b_mesh_qc/TRACEABLE_V14_MISSING_CHECKS_REPORT.md`: provenance, definitions, and audited summary statistics.

`phaseb_mesh_qc.py` applies the complete NIfTI affine during mesh export, including orientation/reflection, translation, and voxel spacing. `run_missing_checks.py` reproduces the supplementary checks when the case-level repaired STLs and threshold-0.5 masks are available:

```bash
python run_missing_checks.py \
  --stl-glob 'outputs/phase_b_mesh_qc/case_outputs/*/segmentation_repaired.stl' \
  --mask-glob 'outputs/final_test_250_phaseb_traceable_v14/case_outputs/*/seg_mask_0.5.nii.gz' \
  --case-regex '(?<=case_outputs/)[0-9]{1,4}' \
  --mesh-space ras \
  --out-csv outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv
```

The primary slicability definition requires every plane that intersects a mesh to contain only closed contours. The strict definition additionally requires all 50 requested planes to intersect and close. These definitions must not be interchanged.

## Production-slicer cohort validation

`production_slicer_validation.py` automates PrusaSlicer invocation and verifies textual G-code, layer markers, positive-extrusion movements, support roles, warnings, complete-STL processing, and reference build-volume fit. The authoritative publication result is the complete 250-case held-out test cohort in `outputs/production_slicer_validation/cohort_250_real/`.

The fixed computational reference profile is PrusaSlicer 2.9.6, Original Prusa MK4S, Generic PLA, a 0.4 mm nozzle, `0.20mm STRUCTURAL @MK4S 0.4`, 0.20 mm layers, two perimeters, 15% infill, native scale, retained source orientation, and automatic organic supports everywhere. Success requires exit status zero, nonempty textual G-code, at least one printed layer, and at least one spatial movement with positive extrusion.

Recheck the retained CSV, summary, manifest, profile hash, cohort identity, warnings, and Phase B join with:

```bash
python verify_production_slicer_cohort.py
```

The earlier case-720 result remains an illustrative validation-partition example at threshold 0.4. It is not part of the 250-case held-out cohort, whose segmentation threshold is 0.5. See `outputs/production_slicer_validation/README.md` for the retained-artifact index and scope boundary.

Large/private artifacts are intentionally excluded: ImageCAS images and labels, predictions, checkpoints beyond those already tracked, case-level NIfTI/STL payloads, generated G-code, caches, and temporary LaTeX products. Their compact derived results, provenance, and available hashes are retained.

This is software-level production-slicer execution under one fixed computational profile. It is not physical printing, bioprinter or bioink validation, print-fidelity validation, biological validation, or evidence of clinical readiness.

## Conventional 3D U-Net Baseline

The finalized same-split comparison uses exactly 250 paired ImageCAS held-out cases at a fixed threshold of 0.5 without test-set threshold tuning. The Attention U-Net achieved Dice 0.7878 versus 0.7723 and clDice 0.8695 versus 0.8590 for the conventional plain 3D U-Net. It also had higher precision and lower HD95, while recall was statistically similar.

This is a same-split conventional 3D U-Net baseline comparison, not a pure architecture ablation. See [the detailed baseline documentation](docs/experiments/plain_unet_baseline/README.md) for cohort provenance, paired statistics, reproducibility scripts, and manuscript-ready text.

## Utility Scripts:

- `beforevsafter.py`: posterstyle pipeline visualizations and mesh export
- `graph1.py`: metric projection/plotting from `epoch_metrics.csv`
- `sparsity.py`: dataset sparsity analysis for label volumes
- `precompute_label_bboxes.py`: offline bbox cache generation (`label_bboxes.json`)

## Other Notes:

- This repository contains generated artifacts (checkpoints, logs, example outputs) to make local experimentation easier.
- For reproducibility, Phase B writes the effective runtime config to `used_config.yaml` per case.
