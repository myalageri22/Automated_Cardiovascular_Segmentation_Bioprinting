# Manuscript claim-to-artifact mapping

The canonical manuscript is `manuscript/finalv2_production_slicer.tex`. “Verified” means the reported value was recomputed from the retained per-case artifact or checked directly against the retained summary. Private ImageCAS images, labels, prediction volumes, case-level meshes, and full G-code are intentionally excluded; their lightweight derived outputs and available hashes are retained.

## Numerical claims

| Section or item | Claim | Supporting path | Column, key, or method | Status |
|---|---|---|---|---|
| Abstract; Phase A Results; Tables 4 and 10 | 250 successful evaluations; mean Dice@0.5 0.7878; median 0.7934; 95% CI 0.7820-0.7935 | `outputs/final_test_250/per_case_metrics.csv`; `summary_metrics.json` | `status`, `dice@0.5`; `metrics.dice@0.5` | Verified |
| Phase A Results; Table 4 | Mean precision 0.7636 and recall 0.8205 | `outputs/final_test_250/per_case_metrics.csv`; `summary_metrics.json` | `precision@0.5`, `recall@0.5` | Verified |
| Phase A Results; Table 4 | HD95 mean 5.01 mm, median 3.69 mm | `outputs/final_test_250/per_case_metrics.csv`; `summary_metrics.json` | `hd95@0.5` | Verified |
| Paired analysis; Table 4 | Mean clDice 0.8695 | `outputs/final_test_250/per_case_metrics.csv`; `summary_metrics.json` | `cldice@0.5` | Verified |
| Paired analysis | Bootstrap advantage Δ\|ρ\| = 0.118, 95% CI 0.054-0.183 | `outputs/phase_b_mesh_qc/segmentation_mesh_correlation_summary.json` | `bootstrap_comparison` | Verified |
| Paired analysis | Dice-primary-slicability ρ = 0.075, p = 0.239 | `outputs/final_test_250/seg_to_mesh_correlations.csv` | Dice row versus `slicability_pass` | Verified |
| Paired analysis | Dice-components approximately -0.37; clDice-components -0.49 | `outputs/final_test_250/seg_to_mesh_correlations.csv`; `outputs/phase_b_mesh_qc/segmentation_mesh_correlation_summary.json` | metric rows versus `connected_component_count` | Verified |
| Phase B Results; Tables 3 and 10 | 228/250 (91.2%) met watertight plus zero-detected-non-manifold-edge criterion | `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv`; `summary_mesh_qc.json` | `watertight`, `non_manifold_edge_count` | Verified |
| Phase B Results; Tables 3 and 10 | Connected components mean 10.152, median 9, range 2-39 | `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv` | `connected_component_count` | Verified |
| Phase B Results; Tables 3 and 10 | Primary slicability 125/250 (50.0%) | `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv` | `slicability_pass` | Verified |
| Phase B Results; Tables 3 and 10 | Strict all-50-plane slicability 61/250 (24.4%) | same | `planes_with_intersection == 50` and `planes_all_contours_closed == 50` | Verified |
| Phase B Results; Tables 3 and 10 | Centroid displacement mean 0.447 mm, median 0.140 mm, maximum 6.948 mm | same | `centroid_offset_mm` | Verified |
| Phase B Results; Tables 3 and 10 | 250/250 within two-voxel affine-aware alignment criterion | same | `alignment_pass`, `bbox_max_error_mm`, `mask_spacing_mm` | Verified |
| Production-slicer Methods and Results; Tables 8 and 10 | 250/250 successful textual toolpaths; zero failures | `outputs/production_slicer_validation/cohort_250_real/per_case_slicer_results.csv`; `summary_slicer_results.json` | `process_exit_code`, `gcode_is_textual`, `layer_count`, `extrusion_move_count`, `toolpath_generation_success`; cohort counts | Verified |
| Production-slicer Results; Table 8 | 247 warning-free; warnings in cases 803, 821, and 822 | same | `slicer_warnings_json`; `warned_cases` | Verified |
| Production-slicer Results; Table 8 | Layers median 450, IQR 423-489, range 309-677 | same | `layer_count`; inclusive quartiles | Verified |
| Production-slicer Results; Table 8 | Positive-extrusion movements median 174,104, range 45,667-317,816 | same | `extrusion_move_count` | Verified |
| Production-slicer Results; Table 8 | Estimated time median 7,488.5 s, IQR 6,307.25-8,725.5 s, range 2,568-13,419 s | same | `estimated_print_time_seconds`; inclusive quartiles | Verified |
| Production-slicer Results; Table 8 | Complete-STL processing, native-scale build fit, and organic supports: 250/250 each | same | `complete_stl_processed`, `build_volume_fit`, `support_generation_result` | Verified |
| Production-slicer Results | All 22 earlier mesh-integrity failures met slicer criteria | slicer CSV joined to `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv` | case ID; integrity criterion versus `toolpath_generation_success` | Verified by `verify_production_slicer_cohort.py` |
| Illustrative case 720; Table 9 | Case-level threshold 0.4 geometry and QC | `outputs/production_slicer_validation/case720_verified_summary.json` and retained Phase B case report provenance | case-level fields only | Verified; not part of held-out cohort |

## Tables

| Manuscript table | Supporting artifact | Status |
|---|---|---|
| Table 1, dataset and preprocessing | `extra_information/data_information/dataset_splits.json`, `outputs/final_test_250/used_config.yaml`, training/evaluation source | Present |
| Table 2, training configuration | `outputs/final_test_250/used_config.yaml`, checkpoint metadata, training source | Present |
| Table 3, Phase B cohort QC | `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv`, `missing_checks_per_case_traceable_v14.csv`, summaries | Verified |
| Table 4, evaluation metrics and QC outputs | Phase A and Phase B per-case files above | Verified |
| Table 5, full held-out metric statistics | `outputs/final_test_250/summary_metrics.json`, `per_case_metrics.csv` | Verified |
| Table 6, literature context | cited primary literature in `manuscript/references.bib` | Contextual, not a same-split result |
| Table 7, production-slicer success definition | `production_slicer_validation.py`, reference profile | Present |
| Table 8, full-cohort production-slicer validation | `cohort_250_real/per_case_slicer_results.csv`, summary, manifest, verification report | Verified |
| Table 9, illustrative case-720 QC | `case720_verified_summary.json`; case-level QC provenance | Verified; illustrative only |
| Table 10, consolidated numerical results | all per-case and summary artifacts listed above | Verified |
| Appendix augmentation table | training source and `used_config.yaml` | Present |
| Appendix hyperparameter table | training source, `used_config.yaml`, `phaseb/configs/default.yaml` | Present |

## Result figures

| Manuscript figure | Committed figure | Supporting artifact or source | Status |
|---|---|---|---|
| Pipeline overview | `manuscript/figures/figure1_pipeline_overview.png` | repository workflow and scripts | Present |
| Preprocessing panel | `manuscript/figures/figure2_preprocessing.png` | preprocessing configuration; private input image excluded | Present; derived figure retained |
| Taubin smoothing example | `manuscript/figures/figure3_taubin_smoothing.png` | illustrative case-level Phase B QC | Present; illustrative |
| clDice versus components | `manuscript/figures/figure11_cldice_vs_components.png` | `per_case_metrics.csv` joined to `per_case_mesh_qc.csv` | Present; values verified |
| Metric-mesh heatmap | `manuscript/figures/fig_metric_mesh_heatmap.png` | `seg_to_mesh_correlations.csv`, correlation summary JSON | Present; values verified |
| Training curves | `manuscript/figures/figure4_training_curves.png` | retained training history and `scripts/build_combined_training_curve.py` | Present |
| Dice distribution | `manuscript/figures/figure7_full_test_dice_distribution.png` | `per_case_metrics.csv` | Present; values verified |
| Qualitative segmentation | `manuscript/figures/fig_qualitative_segmentation.png` | private case volumes excluded; derived panel retained | Present |
| Ranked Dice | `manuscript/figures/figure8_per_case_dice_ranked.png` | `per_case_metrics.csv` | Present; values verified |
| Foreground volume versus Dice | `manuscript/figures/figure9_foreground_volume_vs_dice.png` | `per_case_metrics.csv` | Present; values verified |
| HD95 distribution | `manuscript/figures/fig_hd95_distribution.png` | `per_case_metrics.csv`, `hd95@0.5` | Authentic supplied figure restored; values verified |
| Precision versus recall | `manuscript/figures/fig_precision_recall.png` | `per_case_metrics.csv`, `precision@0.5`, `recall@0.5`, `dice@0.5` | Authentic supplied figure restored; values verified |
| Case-720 fabrication-readiness panel | `manuscript/figures/figure10_fabrication_readiness_qc.png` | illustrative case-level Phase B report | Present; illustrative |
| Cohort STL views | `manuscript/figures/figure6_vessel_stl_cohort.png` | selected held-out Phase B meshes; large case-level meshes excluded | Present; derived figure retained |

## Production-slicer provenance and exclusions

The canonical profile is `outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini` (SHA-256 `eb43615821fc138e5c6ec7f566b734a1492cf6aed62cabbc4b1139b78c7cf4dd`). `outputs/production_slicer_validation/cohort_250_real/validation_manifest.json` records the PrusaSlicer version, success definition, input STL hashes, and run provenance. `verify_production_slicer_cohort.py` independently checks all retained claims.

Full G-code (approximately 2.0 GiB), case-level STL/NIfTI payloads, medical images, labels, and predictions are intentionally excluded. The compact per-case parsed results, payload hashes, profile, manifest, warning details, and verification report remain sufficient to audit the reported manuscript values. No physical fabrication was performed.
