# Manuscript claim-to-artifact mapping

| Manuscript claim | Supporting artifact | Relevant field |
|---|---|---|
| 250/250 successful held-out evaluations | `outputs/final_test_250/per_case_metrics.csv` | one unique successful row per case |
| 228/250 (91.2%) combined mesh-integrity criterion | `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv` | `watertight`, `non_manifold_edge_count` |
| Median 9 connected components (mean 10.2, range 2-39) | `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv` | `connected_component_count` |
| Primary slicability 125/250 (50.0%) | `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv` | `slicability_pass` |
| Strict all-50-plane slicability 61/250 (24.4%) | `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv` | `planes_with_intersection == 50` and `planes_all_contours_closed == 50` |
| Centroid displacement mean 0.447 mm, median 0.140 mm, maximum 6.948 mm | `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv` | `centroid_offset_mm` |
| 250/250 within the two-voxel affine-aware alignment criterion | `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv` | `alignment_pass`, `bbox_max_error_mm`, `mask_spacing_mm` |
| clDice predicts connected-component count more strongly than Dice | `outputs/final_test_250/seg_to_mesh_correlations.csv` | clDice/Dice versus `connected_component_count`; manuscript reports $\rho=-0.49$ versus approximately $-0.37$ |
| Representative case-720 PrusaSlicer toolpath result | `outputs/production_slicer_validation/case720_verified_summary.json` | `slicer`, `layer_count`, extrusion counts, supports, warnings, build fit, and estimated time |
| Exact production-slicer profile | `outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini` | printer, filament, nozzle, preset, support fields |
| Production-slicer success checks | `production_slicer_validation.py` | `assess_toolpath`, G-code parser, geometry/build-volume checks |

The case-720 toolpath result is a representative software-level demonstration only. It is not a cohort metric and does not establish physical printing, bioprinter compatibility, bioink compatibility, geometric print fidelity, perfusion, mechanics, or biological performance.
