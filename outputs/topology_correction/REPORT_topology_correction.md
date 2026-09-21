# Topology-aware postprocessing for robust patient-specific 3D coronary reconstruction from CCTA segmentation

Experiment report. Sections follow the required structure A-G. Every value below is read back from the artifacts in `outputs/topology_correction`; none is re-derived in this document.

- Run started: `2026-08-29T08:13:45.426152-05:00`
- Run finished: `2026-08-30T03:16:53.890433-05:00`
- Config SHA-256: `4a1b818d3339204b86e519fb15c116451edc2647399ac21101b146a2b1d1fef4`
- Git commit: `3ba711d0cef4ae5558329ddb09d835bf4557c332`
- Seed: `20260826`

## A. Repository audit

Authoritative artifacts of the reported 250-case experiment. These are read-only inputs; they were SHA-256 fingerprinted before and after the run and compared.

| Authoritative artifact | SHA-256 (before run) |
| --- | --- |
| `extra_information/data_information/dataset_splits.json` | `0c46dcb6e0b2e1e3…` |
| `outputs/final_test_250/per_case_fabrication_readiness.csv` | `89400c4d710933b4…` |
| `outputs/final_test_250/per_case_metrics.csv` | `bc0bac797154991f…` |
| `outputs/final_test_250/summary_metrics.json` | `4579b16c0ba1e051…` |
| `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv` | `3c40980049462164…` |

Inputs resolved for the experiment:

- Test split: `extra_information/data_information/dataset_splits.json` key `test` (250 requested, 250 resolved)
- Frozen predictions: `{pred_root}/{case}/seg_mask_0.5.nii.gz` (threshold 0.5, binarised from `mask`)
- Probability maps: `{pred_root}/{case}/seg_prob.nii.gz` (retained; not used to re-threshold)
- CTA: `Data/all/{case}/{case}.img.nii.gz`  ·  Ground truth: `Data/all/{case}/{case}.label.nii.gz`
- Phase A per-case metrics: `outputs/final_test_250/per_case_metrics.csv`
- Phase B mesh QC: `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv`
- Phase B reconstruction source SHA-256: `0512a4637ff45d07…` (imported unchanged; marching cubes, affine handling, smoothing, repair rules and QC definitions are the authoritative implementations)
- Per-case input manifest: `input_manifest.json` (250 cases with resolved paths and hashes)

Pre-existing connected-component filtering: `phaseb/configs/default.yaml` carries `min_component_mm3: 50`, supported by the Phase B code but **disabled for the reported cohort**. It is included here as the 50 mm³ point of the Strategy 1 grid so the legacy setting is measured rather than assumed.

New code lives in `experiments/topology_correction/` and new results in `outputs/topology_correction`. No authoritative file is written by this experiment (verified: `authoritative_artifacts_unchanged = True`).

## B. Method implemented

Deterministic postprocessing applied to the frozen binary predictions, before Phase B reconstruction. No retraining, no re-thresholding, no case-specific tuning, no manual repair.

**Stage A — component characterisation.** For every predicted mask, 3D connected components (structure rank 1) with, per component: voxel count, physical volume (mm³), centroid in voxel and world/RAS coordinates, bounding box, maximum extent, and surface-to-surface distance to the nearest larger component. Components are ranked by physical volume. The full pre-modification audit is `component_audit.csv`. No assumption is made that one component is anatomically correct: the left and right systems may legitimately be separate.

**Correction strategies (prespecified).**

- **S0 Original** — control, identity. Control. No topology correction. Reproduces the authoritative reconstruction.
- **S1 Absolute-volume filtering** — remove components below a physical volume. Primary 5.0 mm³; grid [0.216, 1.0, 2.0, 5.0, 10.0, 27.0, 50.0] mm³.
  - 0.216 mm³: single voxel
  - 1.0 mm³: sub-segment speckle
  - 2.0 mm³: half the minimum resolvable segment
  - 5.0 mm³: PRIMARY: minimum resolvable 1 mm x 5 mm vessel segment
  - 10.0 mm³: 2x primary
  - 27.0 mm³: 5x5x5 voxel cube
  - 50.0 mm³: legacy phaseb/configs/default.yaml min_component_mm3 (never applied to the reported cohort)
- **S2 Relative-volume filtering** — remove components below a fraction of the largest component. Primary 0.01; grid [0.001, 0.005, 0.01, 0.05].
  - 0.001: 0.1% of largest component
  - 0.005: 0.5% of largest component
  - 0.01: PRIMARY: 1% of largest component
  - 0.05: 5% - deliberately aggressive upper bound, expected to remove real anatomy
- **S3 Conservative short-gap reconnection** — components ≥ 5.0 mm³ separated by ≤ max-gap are joined by a bridge of radius 0.3 mm along the shortest surface-to-surface segment, at most 50 bridges per case. Primary gap 1.2 mm; grid [0.6, 1.2, 1.8, 2.4] mm. All distances are physical (mm), computed on the 0.6 mm isotropic representation.
  - 0.6 mm: one voxel
  - 1.2 mm: PRIMARY: two voxels; below typical coronary lumen diameter
  - 1.8 mm: three voxels
  - 2.4 mm: four voxels - deliberately permissive upper bound
- **S3c Morphological closing** — binary closing with a small physical radius. Primary 0.6 mm; grid [0.6, 1.2] mm.
  - 0.6 mm: PRIMARY: one voxel radius
  - 1.2 mm: two voxel radius - expected to inflate lumen and cost fidelity

Variants evaluated: **18** (all grid points of all strategies, every one reported).

**Metrics.** Segmentation fidelity: Dice, clDice, precision, recall, HD95, predicted foreground volume, volume difference from the original prediction and from ground truth. Topology: connected components, skeleton voxel/endpoint/branch-point statistics, skeleton component count, centreline distance. Mesh (identical Phase B pipeline): extraction and repair success, watertightness, non-manifold edge count, mesh-integrity pass/fail, mesh component count, mesh-to-mask centroid displacement, bounding-box alignment, cross-sectional contour closure. Geometry preservation, corrected vs original reconstruction (and vs ground-truth meshes built with the identical procedure, for evaluation only): Chamfer distance, symmetric mean and 95th-percentile surface distance, Hausdorff distance, relative mesh-volume change, centroid displacement, bounding-box extent change. No branch labels were invented; the dataset contains none.

## C. Reproducibility

```bash
cd experiments/topology_correction
python -m pip install -r requirements_topology_correction.txt
# full experiment from the frozen predictions
python run_topology_correction_experiment.py \
    --config config/experiment_config.yaml
# synthetic end-to-end validation, no cohort data
python run_topology_correction_experiment.py --mode phantom
# regenerate Figure D and this report from existing result CSVs
python build_final_report.py --config config/experiment_config.yaml
```

Recorded environment:

- `python`: 3.10.19 | packaged by conda-forge | (main, Oct 22 2025, 22:46:49) [Clang 19.1.7 ]
- `platform`: macOS-26.5.2-arm64-arm-64bit

Seed `20260826` is set for the bootstrap resampling; the correction itself is deterministic. Configuration is YAML (`experiment_config.yaml`, SHA-256 recorded), not hard-coded. Inputs, outputs, timestamps, versions and the git state are in `run_manifest.json`.

## D. Results

Cohort: **250** held-out cases attempted, **250** succeeded, **0** failed. Recomputed control mean Dice **0.7878** (inside the prespecified provenance interval: True); provenance check passed for 250 of 250 control cases.

### Primary variants — cohort means

| variant_id | n_cases | components_original_mean | components_corrected_mean | delta_components_median | dice_original_mean | dice_corrected_mean | delta_dice_mean | cldice_original_mean | cldice_corrected_mean | delta_cldice_mean | hd95_original_mean | hd95_corrected_mean | delta_hd95_mean | mesh_integrity_rate_original | mesh_integrity_rate_corrected | surface_deviation_mean_mm | surface_deviation_p95_mm | centroid_displacement_mm | mesh_volume_change_relative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1_absolute_volume__mm3_5 | 250 | 11.0920 | 7.3560 | -3 | 0.7878 | 0.7879 | 0.000186 | 0.8695 | 0.8718 | 0.0023 | 7.5643 | 7.6164 | 0.0521 | 0.9120 | 0.9320 | 0.0135 | 1.33e-14 | 0.1186 | -0.0093 |
| s2_relative_volume__frac_0.01 | 250 | 11.0920 | 4.3560 | -6 | 0.7878 | 0.7896 | 0.0018 | 0.8695 | 0.8763 | 0.0067 | 7.5643 | 7.4766 | -0.0877 | 0.9120 | 0.9560 | 0.1447 | 1.57e-14 | 0.6394 | -0.0153 |
| s3_gap_bridge__gap_1.2mm | 250 | 11.0920 | 7.2400 | -3 | 0.7878 | 0.7879 | 0.000186 | 0.8695 | 0.8717 | 0.0022 | 7.5643 | 7.6165 | 0.0522 | 0.9120 | 0.9320 | 0.0135 | 1.33e-14 | 0.1186 | -0.0093 |
| s3c_closing__r_0.6mm | 250 | 11.0920 | 7.1760 | -3 | 0.7878 | 0.7877 | -5.88e-05 | 0.8695 | 0.8722 | 0.0027 | 7.5643 | 7.6002 | 0.0359 | 0.9120 | 0.9640 | 0.0152 | 1.59e-14 | 0.1246 | -0.0075 |


### Paired statistical tests — primary variants, primary endpoints

| variant_id | metric | n_pairs | test | median_difference | hodges_lehmann_difference | effect_size_name | effect_size | ci_low | ci_high | p_value | p_adjusted_bh | significant_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1_absolute_volume__mm3_5 | Connected components | 250 | wilcoxon_signed_rank | -3 | -3.5000 | rank_biserial | -1 | -4 | -3 | 5.01e-39 | 2.51e-38 | yes |
| s1_absolute_volume__mm3_5 | clDice | 250 | wilcoxon_signed_rank | 0.0016 | 0.0019 | rank_biserial | 0.8807 | 0.0012 | 0.0020 | 9.79e-30 | 2.45e-29 | yes |
| s1_absolute_volume__mm3_5 | Dice | 250 | wilcoxon_signed_rank | 0.000119 | 0.000159 | rank_biserial | 0.7526 | 9.53e-05 | 0.000162 | 1.03e-22 | 1.71e-22 | yes |
| s1_absolute_volume__mm3_5 | HD95 (mm) | 250 | wilcoxon_signed_rank | 0 | 0 | rank_biserial | 0.0095 | 0 | 0 | 0.9146 | 0.9146 | no |
| s1_absolute_volume__mm3_5 | Mesh integrity | 250 | mcnemar_exact | n/a | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0781 | no |
| s1_absolute_volume__mm3_5 | Precision | 250 | wilcoxon_signed_rank | 0.000306 | 0.00038 | rank_biserial | 0.9506 | 0.000243 | 0.000391 | 2.99e-35 | 7.47e-35 | yes |
| s1_absolute_volume__mm3_5 | Recall | 250 | wilcoxon_signed_rank | 0 | -3.42e-05 | rank_biserial | -1 | 0 | 0 | 3.9e-18 | 6.49e-18 | yes |
| s1_absolute_volume__mm3_5 | Average symmetric surface distance (mm) | 250 | wilcoxon_signed_rank | -0.0016 | -0.0028 | rank_biserial | -0.2490 | -0.0035 | -0.000616 | 0.0012 | 0.0012 | yes |
| s1_absolute_volume__mm3_5 | Centreline distance (mm) | 250 | wilcoxon_signed_rank | -0.0194 | -0.0307 | rank_biserial | -0.6238 | -0.0278 | -0.0137 | 1.2e-15 | 1.5e-15 | yes |
| s1_absolute_volume__mm3_5 | Predicted foreground volume (mm^3) | 250 | wilcoxon_signed_rank | -3.8880 | -4.4280 | rank_biserial | -1 | -4.7520 | -3.2400 | 7.83e-39 | 3.92e-38 | yes |
| s2_relative_volume__frac_0.01 | Connected components | 250 | wilcoxon_signed_rank | -6 | -6 | rank_biserial | -1 | -6 | -5 | 7.43e-42 | 3.71e-41 | yes |
| s2_relative_volume__frac_0.01 | clDice | 250 | wilcoxon_signed_rank | 0.0056 | 0.0062 | rank_biserial | 0.7791 | 0.0042 | 0.0067 | 8.05e-26 | 2.01e-25 | yes |
| s2_relative_volume__frac_0.01 | Dice | 250 | wilcoxon_signed_rank | 0.0014 | 0.0017 | rank_biserial | 0.7289 | 0.0012 | 0.0018 | 5.56e-23 | 9.27e-23 | yes |
| s2_relative_volume__frac_0.01 | HD95 (mm) | 250 | wilcoxon_signed_rank | -0.0892 | -0.1150 | rank_biserial | -0.1814 | -0.1942 | 0 | 0.0191 | 0.0191 | yes |
| s2_relative_volume__frac_0.01 | Mesh integrity | 250 | mcnemar_exact | n/a | n/a | paired_rate_difference | 0.0440 | 0.0200 | 0.0720 | 0.000977 | 0.0012 | yes |
| s2_relative_volume__frac_0.01 | Precision | 250 | wilcoxon_signed_rank | 0.0043 | 0.0043 | rank_biserial | 0.9778 | 0.0037 | 0.0046 | 5.1e-40 | 1.28e-39 | yes |
| s2_relative_volume__frac_0.01 | Recall | 250 | wilcoxon_signed_rank | -0.000106 | -0.000937 | rank_biserial | -1 | -0.000281 | -2.48e-05 | 6.93e-25 | 1.15e-24 | yes |
| s2_relative_volume__frac_0.01 | Average symmetric surface distance (mm) | 250 | wilcoxon_signed_rank | -0.0316 | -0.0493 | rank_biserial | -0.4624 | -0.0528 | -0.0164 | 3.82e-10 | 3.82e-10 | yes |
| s2_relative_volume__frac_0.01 | Centreline distance (mm) | 250 | wilcoxon_signed_rank | -0.0874 | -0.1239 | rank_biserial | -0.6597 | -0.1224 | -0.0594 | 5.75e-19 | 7.19e-19 | yes |
| s2_relative_volume__frac_0.01 | Predicted foreground volume (mm^3) | 250 | wilcoxon_signed_rank | -49.0320 | -52.2720 | rank_biserial | -1 | -54.8640 | -40.8240 | 8.88e-42 | 4.44e-41 | yes |
| s3_gap_bridge__gap_1.2mm | Connected components | 250 | wilcoxon_signed_rank | -3 | -3.5000 | rank_biserial | -1 | -4 | -3 | 1.63e-39 | 8.14e-39 | yes |
| s3_gap_bridge__gap_1.2mm | clDice | 250 | wilcoxon_signed_rank | 0.0013 | 0.0017 | rank_biserial | 0.8082 | 0.000991 | 0.0018 | 9.93e-26 | 2.48e-25 | yes |
| s3_gap_bridge__gap_1.2mm | Dice | 250 | wilcoxon_signed_rank | 0.000118 | 0.000159 | rank_biserial | 0.7474 | 9.53e-05 | 0.000167 | 1.31e-22 | 2.19e-22 | yes |
| s3_gap_bridge__gap_1.2mm | HD95 (mm) | 250 | wilcoxon_signed_rank | 0 | 0 | rank_biserial | 0.0077 | 0 | 0 | 0.9299 | 0.9299 | no |
| s3_gap_bridge__gap_1.2mm | Mesh integrity | 250 | mcnemar_exact | n/a | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0781 | no |
| s3_gap_bridge__gap_1.2mm | Precision | 250 | wilcoxon_signed_rank | 0.000298 | 0.000378 | rank_biserial | 0.9448 | 0.000238 | 0.00038 | 3.85e-35 | 9.63e-35 | yes |
| s3_gap_bridge__gap_1.2mm | Recall | 250 | wilcoxon_signed_rank | 0 | -3.32e-05 | rank_biserial | -0.9907 | 0 | 0 | 5.45e-18 | 9.08e-18 | yes |
| s3_gap_bridge__gap_1.2mm | Average symmetric surface distance (mm) | 250 | wilcoxon_signed_rank | -0.0016 | -0.0028 | rank_biserial | -0.2507 | -0.0034 | -0.000616 | 0.0010 | 0.0010 | yes |
| s3_gap_bridge__gap_1.2mm | Centreline distance (mm) | 250 | wilcoxon_signed_rank | -0.0188 | -0.0300 | rank_biserial | -0.5911 | -0.0275 | -0.0125 | 1.72e-14 | 2.15e-14 | yes |
| s3_gap_bridge__gap_1.2mm | Predicted foreground volume (mm^3) | 250 | wilcoxon_signed_rank | -3.8880 | -4.4280 | rank_biserial | -0.9984 | -4.6440 | -3.2400 | 4.89e-39 | 2.44e-38 | yes |
| s3c_closing__r_0.6mm | Connected components | 250 | wilcoxon_signed_rank | -3 | -3.5000 | rank_biserial | -1 | -4 | -3 | 1.69e-39 | 8.44e-39 | yes |
| s3c_closing__r_0.6mm | clDice | 250 | wilcoxon_signed_rank | 0.0020 | 0.0023 | rank_biserial | 0.7426 | 0.0015 | 0.0023 | 8.66e-24 | 2.16e-23 | yes |
| s3c_closing__r_0.6mm | Dice | 250 | wilcoxon_signed_rank | -9.37e-05 | -8.59e-05 | rank_biserial | -0.2256 | -0.000162 | -4.32e-05 | 0.0020 | 0.0033 | yes |
| s3c_closing__r_0.6mm | HD95 (mm) | 250 | wilcoxon_signed_rank | 0 | 0 | rank_biserial | -0.0099 | 0 | 0 | 0.9074 | 0.9074 | no |
| s3c_closing__r_0.6mm | Mesh integrity | 250 | mcnemar_exact | n/a | n/a | paired_rate_difference | 0.0520 | 0.0120 | 0.0920 | 0.0146 | 0.0183 | yes |
| s3c_closing__r_0.6mm | Precision | 250 | wilcoxon_signed_rank | -0.000529 | -0.000496 | rank_biserial | -0.6052 | -0.000628 | -0.00043 | 1.09e-16 | 1.81e-16 | yes |
| s3c_closing__r_0.6mm | Recall | 250 | wilcoxon_signed_rank | 0.000359 | 0.000389 | rank_biserial | 0.8424 | 0.000339 | 0.000403 | 1.01e-30 | 5.04e-30 | yes |
| s3c_closing__r_0.6mm | Average symmetric surface distance (mm) | 250 | wilcoxon_signed_rank | -0.0015 | -0.0027 | rank_biserial | -0.1957 | -0.0038 | 0.000224 | 0.0073 | 0.0073 | yes |
| s3c_closing__r_0.6mm | Centreline distance (mm) | 250 | wilcoxon_signed_rank | -0.0136 | -0.0281 | rank_biserial | -0.4569 | -0.0244 | -0.0068 | 4.46e-10 | 5.58e-10 | yes |
| s3c_closing__r_0.6mm | Predicted foreground volume (mm^3) | 250 | wilcoxon_signed_rank | 7.5600 | 7.8840 | rank_biserial | 0.7989 | 6.6960 | 8.3160 | 1.75e-27 | 4.37e-27 | yes |


Continuous endpoints: paired test selected by a normality check (Shapiro alpha 0.05); Wilcoxon signed-rank when non-normal, paired t-test only where justified. Binary mesh-integrity pass/fail: mcnemar_exact (McNemar). Confidence intervals: 95% by 10000 bootstrap resamples. Multiplicity: benjamini_hochberg FDR within endpoint families at alpha 0.05. Full detail, including every sensitivity variant, is in `statistical_tests.csv`; per-strategy manuscript tables are in `tables/primary_table__<variant>.{csv,md}`.

### Sensitivity / ablation

`sensitivity_analysis.csv` (identical content to `cohort_summary.csv`) reports every grid point of every strategy against component count, Dice, clDice, HD95, mesh integrity and geometric displacement. Figure S1 plots correction strength against those endpoints.

| variant_id | strength | delta_components_median | delta_dice_mean | delta_cldice_mean | delta_hd95_mean | mesh_integrity_rate_corrected | surface_deviation_mean_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s1_absolute_volume__mm3_0.216 | 0.2160 | 0 | 0 | 0 | 0 | 0 | n/a |
| s1_absolute_volume__mm3_1 | 1 | -2 | 2.43e-05 | 0.0011 | 0.0712 | 0 | n/a |
| s1_absolute_volume__mm3_2 | 2 | -2 | 7.05e-05 | 0.0017 | 0.0421 | 0 | n/a |
| s1_absolute_volume__mm3_5 | 5 | -3 | 0.000186 | 0.0023 | 0.0521 | 0.9320 | 0.0135 |
| s1_absolute_volume__mm3_10 | 10 | -4 | 0.000421 | 0.0031 | 0.0521 | 0 | n/a |
| s1_absolute_volume__mm3_27 | 27 | -5 | 0.0014 | 0.0058 | -0.0550 | 0 | n/a |
| s1_absolute_volume__mm3_50 | 50 | -6 | 0.0024 | 0.0080 | -0.1571 | 0 | n/a |
| s2_relative_volume__frac_0.001 | 0.0010 | -3 | 0.000149 | 0.0021 | 0.0584 | 0 | n/a |
| s2_relative_volume__frac_0.005 | 0.0050 | -5 | 0.000919 | 0.0046 | 0.0858 | 0 | n/a |
| s2_relative_volume__frac_0.01 | 0.0100 | -6 | 0.0018 | 0.0067 | -0.0877 | 0.9560 | 0.1447 |
| s2_relative_volume__frac_0.05 | 0.0500 | -7 | 0.0065 | 0.0153 | -0.7296 | 0 | n/a |
| s3_gap_bridge__gap_0.6mm | 0.6000 | -3 | 0.000186 | 0.0023 | 0.0521 | 0 | n/a |
| s3_gap_bridge__gap_1.2mm | 1.2000 | -3 | 0.000186 | 0.0022 | 0.0522 | 0.9320 | 0.0135 |
| s3_gap_bridge__gap_1.8mm | 1.8000 | -3 | 0.000184 | 0.0016 | 0.0522 | 0 | n/a |
| s3_gap_bridge__gap_2.4mm | 2.4000 | -3 | 0.000184 | 0.0011 | 0.0526 | 0 | n/a |
| s3c_closing__r_0.6mm | 0.6000 | -3 | -5.88e-05 | 0.0027 | 0.0359 | 0.9640 | 0.0152 |
| s3c_closing__r_1.2mm | 1.2000 | -4 | -0.0011 | 0.0020 | 0.0278 | 0 | n/a |


### Stratified analysis

Strata are prespecified: component-count tertiles (quantiles [0.3333, 0.6667]) and an anatomical low-fragmentation stratum (≤ 2 original components) versus the rest. Cutoffs were fixed in the config before the endpoints were inspected. Results: `stratified_analysis.csv`.

| stratum | metric | median_difference | effect_size | ci_low | ci_high | p_adjusted_bh |
| --- | --- | --- | --- | --- | --- | --- |
| high_fragmentation | Connected components | -3 | -1 | -4 | -3 | 1e-38 |
| low_fragmentation | Connected components | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | clDice | 0.0016 | 0.8807 | 0.0012 | 0.0020 | 1.96e-29 |
| low_fragmentation | clDice | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Dice | 0.00012 | 0.7526 | 9.57e-05 | 0.000176 | 2.06e-22 |
| low_fragmentation | Dice | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | HD95 (mm) | 0 | 0.0095 | 0 | 0 | 1 |
| low_fragmentation | HD95 (mm) | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Mesh integrity | n/a | 0.0202 | 0.0040 | 0.0403 | 0.1250 |
| low_fragmentation | Mesh integrity | n/a | 0 | 0 | 0 | 1 |
| high_fragmentation | Connected components | -6 | -1 | -6 | -5 | 1.49e-41 |
| low_fragmentation | Connected components | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | clDice | 0.0056 | 0.7791 | 0.0042 | 0.0068 | 1.61e-25 |
| low_fragmentation | clDice | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Dice | 0.0015 | 0.7289 | 0.0012 | 0.0018 | 1.11e-22 |
| low_fragmentation | Dice | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | HD95 (mm) | -0.0932 | -0.1814 | -0.2041 | 0 | 0.0382 |
| low_fragmentation | HD95 (mm) | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Mesh integrity | n/a | 0.0444 | 0.0202 | 0.0726 | 0.0020 |
| low_fragmentation | Mesh integrity | n/a | 0 | 0 | 0 | 1 |
| high_fragmentation | Connected components | -3 | -1 | -4 | -3 | 3.25e-39 |
| low_fragmentation | Connected components | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | clDice | 0.0014 | 0.8082 | 0.001 | 0.0018 | 1.99e-25 |
| low_fragmentation | clDice | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Dice | 0.000119 | 0.7474 | 9.57e-05 | 0.000176 | 2.63e-22 |
| low_fragmentation | Dice | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | HD95 (mm) | 0 | 0.0077 | 0 | 0 | 1 |
| low_fragmentation | HD95 (mm) | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Mesh integrity | n/a | 0.0202 | 0.0040 | 0.0403 | 0.1250 |
| low_fragmentation | Mesh integrity | n/a | 0 | 0 | 0 | 1 |
| high_fragmentation | Connected components | -3 | -1 | -4 | -3 | 3.38e-39 |
| low_fragmentation | Connected components | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | clDice | 0.0020 | 0.7439 | 0.0015 | 0.0024 | 2.2e-23 |
| low_fragmentation | clDice | 0.00028 | 0.3333 | -6.85e-05 | 0.000629 | 1 |
| high_fragmentation | Dice | -9.37e-05 | -0.2215 | -0.000167 | -4.19e-05 | 0.0050 |
| low_fragmentation | Dice | -0.000181 | -1 | -0.000274 | -8.82e-05 | 0.5000 |
| high_fragmentation | HD95 (mm) | 0 | -0.0099 | 0 | 0 | 1 |
| low_fragmentation | HD95 (mm) | 0 | 0 | 0 | 0 | 1 |
| high_fragmentation | Mesh integrity | n/a | 0.0524 | 0.0160 | 0.0927 | 0.0293 |
| low_fragmentation | Mesh integrity | n/a | 0 | 0 | 0 | 1 |


Exploratory covariate associations between correction benefit (Δ components) and baseline Dice, clDice, HD95 and component count are in `covariate_analysis.csv` and Figure E. These are labelled exploratory; they were not used to select any parameter.

| strategy | benefit_metric | covariate | n | spearman_rho | p_value | analysis | family |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s1_absolute_volume__mm3_5 | benefit_delta_components | dice_original | 250 | 0.1810 | 0.0041 | exploratory | s1_absolute_volume__mm3_5|covariates|benefit_delta_components |
| s1_absolute_volume__mm3_5 | benefit_delta_components | cldice_original | 250 | 0.3021 | 1.14e-06 | exploratory | s1_absolute_volume__mm3_5|covariates|benefit_delta_components |
| s1_absolute_volume__mm3_5 | benefit_delta_components | hd95_original | 250 | -0.1996 | 0.0015 | exploratory | s1_absolute_volume__mm3_5|covariates|benefit_delta_components |
| s1_absolute_volume__mm3_5 | benefit_delta_components | components_original | 250 | -0.7930 | 2.72e-55 | exploratory | s1_absolute_volume__mm3_5|covariates|benefit_delta_components |
| s2_relative_volume__frac_0.01 | benefit_delta_components | dice_original | 250 | 0.2256 | 0.000323 | exploratory | s2_relative_volume__frac_0.01|covariates|benefit_delta_components |
| s2_relative_volume__frac_0.01 | benefit_delta_components | cldice_original | 250 | 0.3507 | 1.21e-08 | exploratory | s2_relative_volume__frac_0.01|covariates|benefit_delta_components |
| s2_relative_volume__frac_0.01 | benefit_delta_components | hd95_original | 250 | -0.2208 | 0.000437 | exploratory | s2_relative_volume__frac_0.01|covariates|benefit_delta_components |
| s2_relative_volume__frac_0.01 | benefit_delta_components | components_original | 250 | -0.9070 | 3.89e-95 | exploratory | s2_relative_volume__frac_0.01|covariates|benefit_delta_components |
| s3_gap_bridge__gap_1.2mm | benefit_delta_components | dice_original | 250 | 0.1912 | 0.0024 | exploratory | s3_gap_bridge__gap_1.2mm|covariates|benefit_delta_components |
| s3_gap_bridge__gap_1.2mm | benefit_delta_components | cldice_original | 250 | 0.3150 | 3.67e-07 | exploratory | s3_gap_bridge__gap_1.2mm|covariates|benefit_delta_components |
| s3_gap_bridge__gap_1.2mm | benefit_delta_components | hd95_original | 250 | -0.2053 | 0.0011 | exploratory | s3_gap_bridge__gap_1.2mm|covariates|benefit_delta_components |
| s3_gap_bridge__gap_1.2mm | benefit_delta_components | components_original | 250 | -0.8012 | 3.11e-57 | exploratory | s3_gap_bridge__gap_1.2mm|covariates|benefit_delta_components |
| s3c_closing__r_0.6mm | benefit_delta_components | dice_original | 250 | 0.1982 | 0.0016 | exploratory | s3c_closing__r_0.6mm|covariates|benefit_delta_components |
| s3c_closing__r_0.6mm | benefit_delta_components | cldice_original | 250 | 0.3204 | 2.25e-07 | exploratory | s3c_closing__r_0.6mm|covariates|benefit_delta_components |
| s3c_closing__r_0.6mm | benefit_delta_components | hd95_original | 250 | -0.2121 | 0.000737 | exploratory | s3c_closing__r_0.6mm|covariates|benefit_delta_components |
| s3c_closing__r_0.6mm | benefit_delta_components | components_original | 250 | -0.8050 | 3.66e-58 | exploratory | s3c_closing__r_0.6mm|covariates|benefit_delta_components |


## E. Failure analysis

Representative cases were selected by fixed rules, not by inspection:

- **high_dice_high_fragmentation** — highest original component count among cases with original Dice above the cohort median
- **largest_component_reduction** — largest reduction in component count (original - corrected)
- **largest_fidelity_loss** — largest decrease in Dice (corrected - original), i.e. the worst anatomical cost
- **moderate_dice_good_topology** — original components <= 2 and original Dice closest to the cohort median
- **no_improvement** — zero change in component count, tie-broken by highest original component count

55 case panels were rendered (23 distinct cases), each showing CTA, ground truth, original prediction and corrected prediction slices plus original and corrected meshes. Files: `failure_analysis/<category>/<case>_slices.png` and `_meshes.png`; index `failure_analysis/rendered_cases.csv`.

| category | n_cases |
| --- | --- |
| high_dice_high_fragmentation | 12 |
| largest_component_reduction | 11 |
| largest_fidelity_loss | 12 |
| moderate_dice_good_topology | 8 |
| no_improvement | 12 |




Figure D assembles one case per category — including the no-improvement and largest-fidelity-loss categories — so successes and failures appear side by side.

## F. Scientific interpretation

Outcome labels are assigned by fixed thresholds recorded in `interpretation_rules.json` before the numbers were read:

- `meaningful_component_reduction_median` = -1.0
- `negligible_dice_loss` = -0.01
- `negligible_hd95_increase` = 0.5
- `negligible_surface_deviation_mm` = 0.6

| variant_id | outcome | interpretation | delta_components_median | delta_dice_mean | delta_hd95_mean | surface_deviation_mean_mm |
| --- | --- | --- | --- | --- | --- | --- |
| s1_absolute_volume__mm3_5 | A | Topology improves substantially with negligible geometric cost. | -3 | 0.000186 | 0.0521 | 0.0135 |
| s2_relative_volume__frac_0.01 | A | Topology improves substantially with negligible geometric cost. | -6 | 0.0018 | -0.0877 | 0.1447 |
| s3_gap_bridge__gap_1.2mm | A | Topology improves substantially with negligible geometric cost. | -3 | 0.000186 | 0.0522 | 0.0135 |
| s3c_closing__r_0.6mm | A | Topology improves substantially with negligible geometric cost. | -3 | -5.88e-05 | 0.0359 | 0.0152 |


**How to read the geometry columns.** The surface-deviation distribution is extremely heavy-tailed by construction: after component filtering the retained surface is bit-identical to the original, so more than 95% of sampled points sit at numerical zero and the 95th-percentile symmetric surface distance is ~1e-14 mm. The mean (0.013-0.14 mm) and the Hausdorff (mean 18.6-33.2 mm) are driven entirely by the points on the *removed* islands, which are far from the retained tree. Hausdorff here therefore measures how far away the deleted fragment was, not how much the kept anatomy moved. The manuscript should report the mean and the paired distributions and state this explicitly; quoting the Hausdorff alone would misread deletion as distortion, and quoting the p95 alone would hide the deletion entirely.

Reading of the primary variants: component filtering removes small disconnected islands and the volume-based strategies reduce the median component count while leaving Dice, HD95 and the reconstructed surface essentially where they were — the geometric cost sits below the 0.6 mm voxel pitch, i.e. below what the representation can resolve. The correction therefore removes fragmentation that the segmentation produced, not anatomy that the segmentation found. Two qualifications belong in the manuscript. First, the benefit is bounded: filtering cannot restore a vessel the network missed, so component count falls but clDice barely moves, which is the signature of Outcome A on a narrow endpoint rather than a general topology repair. Second, the aggressive grid points behave as predicted a priori (5% relative filtering, 2.4 mm bridging, 1.2 mm closing) and are reported in full precisely because they show where the trade-off turns; they were not removed for being unfavourable. No parameter was re-selected after seeing an endpoint, and every variant that was run is reported.

## G. Manuscript recommendations

Proposed, not applied. No manuscript text has been rewritten.

| Manuscript element | Recommendation |
| --- | --- |
| Title / abstract | Frame as topology-aware postprocessing for robust patient-specific 3D coronary reconstruction from CCTA segmentation. Bioprinting stays a downstream application, not the motivation. |
| Methods, new subsection | Add the Stage A characterisation and the four prespecified strategies with their physical-unit rationale (Section B here). State that the segmentation model, threshold, split and Phase B pipeline are unchanged and frozen. |
| Results, new subsection | Add the primary paired table (`tables/primary_table__<variant>.md`) and the cohort summary. Report effect sizes and CIs alongside FDR-adjusted P values. |
| Figure A | Pipeline diagram: CTA → frozen Attention U-Net → original segmentation → topology diagnosis → conservative correction → reconstruction → geometry QC. |
| Figure B | Paired connected-component count, original vs corrected. |
| Figure C | Topology–fidelity trade-off across all variants. |
| Figure D | Representative cases, successes and failures. |
| Figure E / S1 | Supplementary: benefit vs baseline metrics; sensitivity to correction strength. |
| Discussion | State the bounded benefit explicitly: postprocessing removes fragmentation but does not recover missing vasculature, which supports preserving topology during training/inference rather than relying on downstream repair. |
| Limitations | Single dataset; no branch-level annotation; the toolpath check is a software-level test and not evidence of physical printability. |
| Claim boundary | Unchanged. No clinical deployment, diagnostic benefit, physical printing, perfusion, biological function, patient outcome, 'bioprinting-ready' geometry or state-of-the-art segmentation claim. Novelty remains the topology↔reconstruction relationship plus a rigorously evaluated correction stage. |

### Secondary toolpath experiment (Section 12)

Every repaired STL of the control and of each primary variant was sliced under one fixed profile. **Software-level slicing outcome only: this is not evidence of physical printability, print quality or fabrication success.**

- Slicer build: `PrusaSlicer-2.8.1+linux-x64-GTK3-202409181416 based on Slic3r (with GUI support)`
- Authoritative cohort run used: PrusaSlicer 2.9.6 (per-case results not in the working copy). That build ships no Linux binary on its release page and its own CDN is unreachable here, and the per-case results of that run are not in the working copy, so this is a fresh paired comparison under settings fixed here -- not a reproduction of the 2.9.6 numbers.
- Profiles, each applied identically to every case and every variant:
  - **P1_no_supports**: `layer-height=0.2`, `first-layer-height=0.2`, `nozzle-diameter=0.4`, `filament-diameter=1.75`, `temperature=210`, `first-layer-temperature=215`, `bed-temperature=60`, `perimeters=2`, `fill-density=15%`, `bed-shape=0x0,250x0,250x210,0x210`, `max-print-height=210`, `center=125,105`, `gcode-flavor=marlin`, `filament-density=1.24`
  - **P2_supports_brim**: `layer-height=0.2`, `first-layer-height=0.2`, `nozzle-diameter=0.4`, `filament-diameter=1.75`, `temperature=210`, `first-layer-temperature=215`, `bed-temperature=60`, `perimeters=2`, `fill-density=15%`, `bed-shape=0x0,250x0,250x210,0x210`, `max-print-height=210`, `center=125,105`, `gcode-flavor=marlin`, `filament-density=1.24`, `--support-material`, `--support-material-auto`, `brim-width=3`
- P1_no_supports: 1250 slices across 5 variants, 101 failures; failure message: "There is an object with no extrusions in the first layer. | Object name: segmentation_repaired.stl" x101
- P2_supports_brim: 1250 slices across 5 variants, 0 failures

| profile_name | variant_id | n_cases | n_toolpath_success | toolpath_success_rate | layer_count_median | empty_layer_count_mean | cases_with_empty_layers | cases_with_any_warning | estimated_print_time_min_median | filament_used_g_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_no_supports | s0_original | 250 | 225 | 0.9000 | 444 | 0 | 0 | 250 | 108.7333 | 8.8100 |
| P1_no_supports | s1_absolute_volume__mm3_5 | 250 | 230 | 0.9200 | 444 | 0 | 0 | 250 | 107.5250 | 8.7300 |
| P1_no_supports | s2_relative_volume__frac_0.01 | 250 | 234 | 0.9360 | 441 | 0 | 0 | 250 | 105.6917 | 8.6300 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | 250 | 230 | 0.9200 | 444 | 0 | 0 | 250 | 107.5250 | 8.7300 |
| P1_no_supports | s3c_closing__r_0.6mm | 250 | 230 | 0.9200 | 444 | 0 | 0 | 250 | 107.6583 | 8.7400 |
| P2_supports_brim | s0_original | 250 | 250 | 1 | 504 | 0 | 0 | 2 | 344.8333 | 56.7550 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | 250 | 250 | 1 | 499.5000 | 0 | 0 | 0 | 341.2083 | 55.8550 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | 250 | 250 | 1 | 496.5000 | 0 | 0 | 0 | 323.4000 | 53.6350 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | 250 | 250 | 1 | 499.5000 | 0 | 0 | 0 | 341.2083 | 55.8550 |
| P2_supports_brim | s3c_closing__r_0.6mm | 250 | 250 | 1 | 502 | 0 | 0 | 0 | 337.9250 | 55.6750 |


Paired against the S0 control, same statistical machinery as the main experiment:

| profile_name | variant_id | metric | n_pairs | test | median_difference | effect_size_name | effect_size | ci_low | ci_high | p_value | p_adjusted_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_no_supports | s1_absolute_volume__mm3_5 | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0625 |
| P1_no_supports | s1_absolute_volume__mm3_5 | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0625 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Layer count | 225 | wilcoxon_signed_rank | 0 | rank_biserial | -0.9900 | 0 | 0 | 2.07e-05 | 2.89e-05 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Layers without extrusion | 225 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | 1 | 0 | 0 | 0.0082 | 0.0095 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Estimated print time (min) | 225 | wilcoxon_signed_rank | -0.1333 | rank_biserial | -0.7675 | -0.1667 | -0.1000 | 3.89e-21 | 2.72e-20 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Filament used (g) | 225 | wilcoxon_signed_rank | 0 | rank_biserial | -0.5465 | 0 | 0 | 5.04e-06 | 8.81e-06 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Filament used (mm) | 225 | wilcoxon_signed_rank | -0.7300 | rank_biserial | -0.7060 | -0.9900 | -0.4400 | 3.3e-18 | 7.71e-18 |
| P1_no_supports | s1_absolute_volume__mm3_5 | G-code size (bytes) | 225 | wilcoxon_signed_rank | -4703 | rank_biserial | -0.7417 | -6038 | -3453 | 4.12e-20 | 1.44e-19 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0360 | 0.0080 | 0.0640 | 0.0225 | 0.0225 |
| P1_no_supports | s2_relative_volume__frac_0.01 | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0360 | 0.0080 | 0.0640 | 0.0225 | 0.0225 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Layer count | 223 | wilcoxon_signed_rank | 0 | rank_biserial | -0.9979 | 0 | 0 | 2.5e-10 | 3.5e-10 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Layers without extrusion | 223 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | 0.6545 | 0 | 0 | 0.0517 | 0.0604 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Estimated print time (min) | 223 | wilcoxon_signed_rank | -1.1833 | rank_biserial | -0.9977 | -1.3500 | -1.0500 | 5.02e-37 | 2.49e-36 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Filament used (g) | 223 | wilcoxon_signed_rank | -0.0600 | rank_biserial | -0.9868 | -0.0600 | -0.0500 | 7.49e-35 | 1.31e-34 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Filament used (mm) | 223 | wilcoxon_signed_rank | -18.5700 | rank_biserial | -0.9878 | -20.2700 | -16.9900 | 1.71e-36 | 4e-36 |
| P1_no_supports | s2_relative_volume__frac_0.01 | G-code size (bytes) | 223 | wilcoxon_signed_rank | -53446 | rank_biserial | -0.9932 | -63230 | -48577 | 7.11e-37 | 2.49e-36 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0625 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0625 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Layer count | 225 | wilcoxon_signed_rank | 0 | rank_biserial | -0.9900 | 0 | 0 | 2.07e-05 | 2.89e-05 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Layers without extrusion | 225 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | 1 | 0 | 0 | 0.0082 | 0.0095 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Estimated print time (min) | 225 | wilcoxon_signed_rank | -0.1167 | rank_biserial | -0.7689 | -0.1667 | -0.0833 | 2.63e-21 | 1.84e-20 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Filament used (g) | 225 | wilcoxon_signed_rank | 0 | rank_biserial | -0.5426 | 0 | 0 | 6.72e-06 | 1.18e-05 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Filament used (mm) | 225 | wilcoxon_signed_rank | -0.7300 | rank_biserial | -0.6962 | -0.9600 | -0.4400 | 5.45e-18 | 1.27e-17 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | G-code size (bytes) | 225 | wilcoxon_signed_rank | -4658 | rank_biserial | -0.7243 | -6038 | -3130 | 1.33e-19 | 4.66e-19 |
| P1_no_supports | s3c_closing__r_0.6mm | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0625 |
| P1_no_supports | s3c_closing__r_0.6mm | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0.0200 | 0.0040 | 0.0400 | 0.0625 | 0.0625 |
| P1_no_supports | s3c_closing__r_0.6mm | Layer count | 225 | wilcoxon_signed_rank | 0 | rank_biserial | -0.9073 | 0 | 0 | 5.25e-06 | 7.35e-06 |
| P1_no_supports | s3c_closing__r_0.6mm | Layers without extrusion | 225 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P1_no_supports | s3c_closing__r_0.6mm | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | 0.7500 | 0 | 0 | 0.0339 | 0.0395 |
| P1_no_supports | s3c_closing__r_0.6mm | Estimated print time (min) | 225 | wilcoxon_signed_rank | -0.0667 | rank_biserial | -0.3590 | -0.1333 | -0.0167 | 3.34e-06 | 5.85e-06 |
| P1_no_supports | s3c_closing__r_0.6mm | Filament used (g) | 225 | wilcoxon_signed_rank | 0.0100 | rank_biserial | 0.6149 | 0.0100 | 0.0100 | 3.67e-13 | 8.56e-13 |
| P1_no_supports | s3c_closing__r_0.6mm | Filament used (mm) | 225 | wilcoxon_signed_rank | 3.7300 | rank_biserial | 0.5876 | 2.6200 | 4.2200 | 2.14e-14 | 7.5e-14 |
| P1_no_supports | s3c_closing__r_0.6mm | G-code size (bytes) | 225 | wilcoxon_signed_rank | -20235 | rank_biserial | -0.9216 | -22790 | -16784 | 4.27e-33 | 2.99e-32 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Layer count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -0.0748 | 0 | 0 | 0.4605 | 0.5373 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Layers without extrusion | 250 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -1 | 0 | 0 | 0.1573 | 0.2202 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Estimated print time (min) | 250 | wilcoxon_signed_rank | -3.3750 | rank_biserial | -0.9648 | -4.1333 | -2.7417 | 4.16e-36 | 2.91e-35 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Filament used (g) | 250 | wilcoxon_signed_rank | -0.4200 | rank_biserial | -0.9252 | -0.4900 | -0.3500 | 1.21e-32 | 2.12e-32 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | Filament used (mm) | 250 | wilcoxon_signed_rank | -141.8900 | rank_biserial | -0.9237 | -163.6600 | -117.4800 | 2.22e-33 | 5.19e-33 |
| P2_supports_brim | s1_absolute_volume__mm3_5 | G-code size (bytes) | 250 | wilcoxon_signed_rank | -71976 | rank_biserial | -0.9274 | -9.34e+04 | -57540 | 8.86e-34 | 3.1e-33 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Layer count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -0.0234 | 0 | 1 | 0.7742 | 0.9032 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Layers without extrusion | 250 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -1 | 0 | 0 | 0.1573 | 0.2202 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Estimated print time (min) | 250 | wilcoxon_signed_rank | -15.7333 | rank_biserial | -0.9977 | -18.3250 | -14.0500 | 1.37e-41 | 4.79e-41 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Filament used (g) | 250 | wilcoxon_signed_rank | -2.1400 | rank_biserial | -0.9918 | -2.4300 | -1.9100 | 4e-41 | 7.05e-41 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | Filament used (mm) | 250 | wilcoxon_signed_rank | -717.4450 | rank_biserial | -0.9918 | -817.0800 | -642.0300 | 4.03e-41 | 7.05e-41 |
| P2_supports_brim | s2_relative_volume__frac_0.01 | G-code size (bytes) | 250 | wilcoxon_signed_rank | -3.32e+05 | rank_biserial | -0.9977 | -390905 | -3.01e+05 | 9.25e-42 | 4.79e-41 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Layer count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -0.0779 | 0 | 0 | 0.4421 | 0.5158 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Layers without extrusion | 250 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -1 | 0 | 0 | 0.1573 | 0.2202 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Estimated print time (min) | 250 | wilcoxon_signed_rank | -3.3750 | rank_biserial | -0.9646 | -4.1333 | -2.7333 | 1.5e-36 | 1.05e-35 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Filament used (g) | 250 | wilcoxon_signed_rank | -0.4200 | rank_biserial | -0.9247 | -0.4900 | -0.3500 | 9.53e-33 | 1.67e-32 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | Filament used (mm) | 250 | wilcoxon_signed_rank | -141.8900 | rank_biserial | -0.9214 | -163.6600 | -117.4800 | 1.22e-33 | 2.84e-33 |
| P2_supports_brim | s3_gap_bridge__gap_1.2mm | G-code size (bytes) | 250 | wilcoxon_signed_rank | -71976 | rank_biserial | -0.9251 | -9.34e+04 | -57540 | 4.88e-34 | 1.71e-33 |
| P2_supports_brim | s3c_closing__r_0.6mm | Toolpath generation success | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s3c_closing__r_0.6mm | G-code generated | 250 | mcnemar_exact | n/a | paired_rate_difference | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s3c_closing__r_0.6mm | Layer count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | 0.0650 | 0 | 1 | 0.4088 | 0.4770 |
| P2_supports_brim | s3c_closing__r_0.6mm | Layers without extrusion | 250 | no_variation | 0 | rank_biserial | 0 | 0 | 0 | 1 | 1 |
| P2_supports_brim | s3c_closing__r_0.6mm | Slicer warning count | 250 | wilcoxon_signed_rank | 0 | rank_biserial | -1 | 0 | 0 | 0.1573 | 0.2202 |
| P2_supports_brim | s3c_closing__r_0.6mm | Estimated print time (min) | 250 | wilcoxon_signed_rank | -4.9417 | rank_biserial | -0.9484 | -5.5919 | -3.8167 | 2.41e-38 | 8.44e-38 |
| P2_supports_brim | s3c_closing__r_0.6mm | Filament used (g) | 250 | wilcoxon_signed_rank | -0.6400 | rank_biserial | -0.9317 | -0.8300 | -0.4800 | 1.69e-36 | 2.96e-36 |
| P2_supports_brim | s3c_closing__r_0.6mm | Filament used (mm) | 250 | wilcoxon_signed_rank | -214.7600 | rank_biserial | -0.9273 | -280.9200 | -160.7100 | 5.16e-37 | 1.2e-36 |
| P2_supports_brim | s3c_closing__r_0.6mm | G-code size (bytes) | 250 | wilcoxon_signed_rank | -1.27e+05 | rank_biserial | -0.9705 | -1.43e+05 | -91098 | 2.25e-40 | 1.57e-39 |


Slicer messages raised, by variant:

| profile_name | variant_id | warning | n_cases | fraction |
| --- | --- | --- | --- | --- |
| P1_no_supports | s0_original | Consider enabling supports | 250 | 1 |
| P1_no_supports | s0_original | Detected print stability issues | 250 | 1 |
| P1_no_supports | s0_original | Floating object part | 250 | 1 |
| P1_no_supports | s0_original | Repaired | 250 | 1 |
| P1_no_supports | s0_original | WARNING | 250 | 1 |
| P1_no_supports | s0_original | Loose extrusions | 249 | 0.9960 |
| P1_no_supports | s0_original | Consider enabling brim | 242 | 0.9680 |
| P1_no_supports | s0_original | Low bed adhesion | 242 | 0.9680 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Consider enabling supports | 250 | 1 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Detected print stability issues | 250 | 1 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Floating object part | 250 | 1 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Repaired | 250 | 1 |
| P1_no_supports | s1_absolute_volume__mm3_5 | WARNING | 250 | 1 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Consider enabling brim | 249 | 0.9960 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Loose extrusions | 249 | 0.9960 |
| P1_no_supports | s1_absolute_volume__mm3_5 | Low bed adhesion | 249 | 0.9960 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Consider enabling supports | 250 | 1 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Detected print stability issues | 250 | 1 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Floating object part | 250 | 1 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Repaired | 250 | 1 |
| P1_no_supports | s2_relative_volume__frac_0.01 | WARNING | 250 | 1 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Consider enabling brim | 248 | 0.9920 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Low bed adhesion | 248 | 0.9920 |
| P1_no_supports | s2_relative_volume__frac_0.01 | Loose extrusions | 247 | 0.9880 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Consider enabling supports | 250 | 1 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Detected print stability issues | 250 | 1 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Floating object part | 250 | 1 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Repaired | 250 | 1 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | WARNING | 250 | 1 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Consider enabling brim | 249 | 0.9960 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Loose extrusions | 249 | 0.9960 |
| P1_no_supports | s3_gap_bridge__gap_1.2mm | Low bed adhesion | 249 | 0.9960 |
| P1_no_supports | s3c_closing__r_0.6mm | Consider enabling supports | 250 | 1 |
| P1_no_supports | s3c_closing__r_0.6mm | Detected print stability issues | 250 | 1 |
| P1_no_supports | s3c_closing__r_0.6mm | Floating object part | 250 | 1 |
| P1_no_supports | s3c_closing__r_0.6mm | Repaired | 250 | 1 |
| P1_no_supports | s3c_closing__r_0.6mm | WARNING | 250 | 1 |
| P1_no_supports | s3c_closing__r_0.6mm | Loose extrusions | 249 | 0.9960 |
| P1_no_supports | s3c_closing__r_0.6mm | Consider enabling brim | 248 | 0.9920 |
| P1_no_supports | s3c_closing__r_0.6mm | Low bed adhesion | 248 | 0.9920 |
| P2_supports_brim | s0_original | Repaired | 2 | 0.0080 |
| P2_supports_brim | s0_original | WARNING | 2 | 0.0080 |


**Reading these two profiles.** Under P1 (no supports, no brim) a minority of cases in every arm is rejected by the slicer with one message -- an object with no extrusions in the first layer -- which is what a branching vessel tree resting on a knife-edge distal tip produces; correction reduces those rejections (control 225/250; corrected 230-234/250, S2 significant on exact McNemar). Under P2, which adds the supports and brim the slicer itself suggests, every arm reaches 250/250, so the P1 rejections are a property of the profile rather than a defect of the reconstructed surface. Report P2 as the headline (it matches the fixed-profile shape of the authoritative run) and P1 as the stress condition where a connectivity difference between arms becomes visible at all. Neither is evidence about physical printing.

For CMIG the endpoint that matters is reconstruction quality; this check belongs in the supplement, framed as a downstream software compatibility observation.

## Quality control

| Check | Result |
| --- | --- |
| 250 held-out cases attempted | 250 |
| Cases failed | 0 (none) |
| No case excluded | True |
| Authoritative artifacts unchanged | True |
| Segmentation threshold retuned | False (threshold 0.5) |
| Provenance enforced against archived Dice | True (250 pass / 0 fail) |
| Variants documented and reported | 18 |
| Figures traceable to source CSV | 15 figures, each with `*_source_data.csv` |
| Git commit recorded | 3ba711d0cef4ae5558329ddb09d835bf4557c332 |
