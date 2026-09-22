# Topology-aware image-to-mesh pipeline for coronary CTA

Code and derived results for the manuscript *A Topology-Aware Image-to-Mesh Pipeline for
Coronary Artery Segmentation and 3D Reconstruction from CT Angiography* (under
double-anonymous review).

The pipeline has two phases:

- **Phase A**: 3D Attention U-Net (MONAI `AttentionUnet`, channels 32-512, strides 2x4,
  dropout 0.1) segments coronary arteries from CCTA resampled to 0.6 mm isotropic RAS.
- **Phase B**: threshold-0.5 masks become world-coordinate surface meshes (Lewiner
  marching cubes, full NIfTI affine), repaired with trimesh, then checked by automated
  geometry QC; an optional topology-correction stage runs on the frozen masks first.

ImageCAS images, labels, predictions, per-case NIfTI/STL files and G-code are not
redistributed. Everything below is derived data that reproduces the reported numbers.

## Where each reported result lives

| Result | Artifact | Produced by |
|---|---|---|
| Held-out segmentation metrics (250 cases, IDs 751-1000) | `outputs/final_test_250/per_case_metrics.csv`, `summary_metrics.*` | `evaluate_full_test_a40.py` (run on Apple MPS; see `eval_command.txt`, `used_config.yaml`) |
| HD95 on the 0.6 mm grid | `outputs/final_test_250/hd95_corrected_per_case.csv` | recomputed from saved masks by the topology-correction pipeline (see *Known issue* below) |
| Mesh QC (integrity, components, roughness, volume change) | `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv`, `summary_mesh_qc.*` | `phaseb_mesh_qc.py` |
| Contour closure, centroid, bounding-box alignment | `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv` | `run_missing_checks.py` |
| Primary endpoint (clDice vs Dice association with fragmentation) | `outputs/final_test_250/primary_endpoint_bootstrap.json` | `scripts/evaluation/segmentation_geometry_association.py` |
| Metric-vs-component correlations; integrity-group comparison | `outputs/final_test_250/segmentation_component_correlations.csv`, `integrity_group_comparison.csv` | same script |
| clDice | `cldice@0.5` in `per_case_metrics.csv` | `compute_cldice.py`, `scripts/evaluation/compute_cldice_3d.py` |
| Same-split plain 3D U-Net baseline | `Results/plain_unet_baseline/` | `baseline_orchestration.py`, `scripts/evaluation/compare_attention_vs_plain.py` |
| Topology correction (4 primary strategies, 18 arms incl. control) | `outputs/topology_correction/` (`cohort_summary.csv`, `statistical_tests.csv`, `sensitivity_analysis.csv`) | `experiments/topology_correction/` |
| PrusaSlicer toolpath check (250 STLs) | `outputs/production_slicer_validation/cohort_250_real/` | `production_slicer_validation.py`, `verify_production_slicer_cohort.py` |
| Data split | `extra_information/data_information/dataset_splits.json` (train 1-700, val 701-750, test 751-1000) | fixed before evaluation |

Reproduce the association analysis (seconds, needs only pandas/scipy):

```bash
python scripts/evaluation/segmentation_geometry_association.py
```

## Evaluated model and training provenance

- Checkpoint: `checkpoints/best_dice05.pt` (Git LFS, ~284 MB). Epoch 79, validation
  Dice@0.5 0.7889, selected on validation Dice@0.5 only. Loads strictly into the
  architecture above.
- The model was developed in stages on the same 700-case training partition. Earlier
  stages (A40 GPU) trained the Attention U-Net with other loss settings. The evaluated
  checkpoint comes from a final fine-tuning stage that resumed from the stage-v11
  checkpoint and trained with the `tversky_dice` objective on Apple silicon (MPS):
  lr 2e-5, AdamW (weight decay 1e-5), ReduceLROnPlateau (factor 0.5, patience 6),
  early-stopping patience 15, batch 1 with gradient accumulation 8, one 96x192x192 patch
  per case (3:1 positive:negative), gradient checkpointing, seed 42. Epoch numbers
  continue across stages. The full config is stored in the checkpoint (`config` key).
- Training entry points: `train_a40_resume.py` (resume trainer; imports the data,
  model and loss code in `train_core.py`) and `train_v14_local_mps.py` (MPS wrapper used
  for the final stage). `evaluate_full_test_a40.py` also imports `train_core.py`.
  `train_vascular.py` is the original trainer used in early stages.

## Metric definitions worth knowing

- **Mesh integrity** = watertight repaired mesh AND zero non-manifold edges.
- **Surface roughness** (`surface_roughness_*` in mesh QC) is a proxy: the mean angle,
  in radians, between normals of consecutively indexed faces. It is not a curvature or
  a length and is reported only as a before/after smoothing comparison.
- **Topology-correction surface deviation** is the mean symmetric distance between the
  corrected and original reconstructed surfaces; the per-variant maximum (Hausdorff) is
  also in `cohort_summary.csv`.

## Known issue: HD95 in the first evaluation run

`per_case_metrics.csv` column `hd95@0.5` was computed with the loader's original header
spacing (~0.25-0.45 mm) instead of the 0.6 mm grid the arrays were resampled to, which
underestimates HD95 (cohort mean 5.01 mm vs 7.56 mm on the correct grid; a 0.50 mm
minimum is impossible on a 0.6 mm grid). `spacing_from_batch` in
`evaluate_full_imagecas_test_mps.py` now reads spacing from the resampled tensor's
affine (regression test: `tests/test_hd95_spacing.py`). Use
`hd95_corrected_per_case.csv` for HD95; Dice, clDice, precision and recall are voxel
counts and are unaffected. The plain U-Net baseline was scored by the same script
(`baseline_orchestration.py` calls `evaluate_full_test_a40.py`), so its `hd95@0.5`
values in `Results/plain_unet_baseline/` carry the same bias until re-scored.

## Scope

Software-level geometry and toolpath checks only. No physical printing, bioprinter or
bioink validation, biological validation, or clinical use.

## Setup and tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements_phaseb.txt
pytest tests phaseb/tests experiments/topology_correction/tests
```

Phase B on one case:

```bash
python run_phaseb.py --ct ct.nii.gz --seg seg_prob.nii.gz --seg-type auto --case-id demo --outdir ./phaseb_outputs
```

Full pipeline on one case:

```bash
python run_full_pipeline.py --ct ct.nii.gz --checkpoint checkpoints/best_dice05.pt --outdir pipeline_outputs --case-id demo
```
