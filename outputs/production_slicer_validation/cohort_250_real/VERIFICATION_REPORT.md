# Production-slicer cohort verification

## Verdict

Verified with no discrepancies. The retained artifacts describe exactly 250 unique held-out test cases, all of which met the predefined software-level PrusaSlicer toolpath-generation criteria. Case 720 is absent from this cohort.

## Command

```bash
python verify_production_slicer_cohort.py --verify-payload-hashes --json-output outputs/production_slicer_validation/cohort_250_real/verification_report.json --markdown-output outputs/production_slicer_validation/cohort_250_real/VERIFICATION_REPORT.md
```

Quartiles use `Python statistics.quantiles(n=4, method='inclusive')`.

## Inputs

- `outputs/production_slicer_validation/cohort_250_real/per_case_slicer_results.csv` — SHA-256 `5b920ba63f2d29572207374b412dc958587088e2f047fb8de59167f5544b12a3`, 402101 bytes
- `outputs/production_slicer_validation/cohort_250_real/summary_slicer_results.json` — SHA-256 `8f9fa187608344ffb66a24581bb00845eb43484e22d5b6b57f4f3460fce9cb36`, 5082 bytes
- `outputs/production_slicer_validation/cohort_250_real/validation_manifest.json` — SHA-256 `9dde4f913025b2ac647ca2f638a45c92b1866d45c76c1e754cf232c4fdcce2e4`, 62399 bytes
- `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv` — SHA-256 `5e8855639beee385cd0309b678852a417ec1a80f7f988b698b9e771f24ebdd7d`, 127084 bytes
- `outputs/final_test_250/per_case_metrics.csv` — SHA-256 `16485207d864405882d7c2426dd0174cb4fc46bbe6c454e55cbc5e8f73afcc4f`, 91125 bytes
- `outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini` — SHA-256 `eb43615821fc138e5c6ec7f566b734a1492cf6aed62cabbc4b1139b78c7cf4dd`, 15491 bytes

## Recomputed results

- Attempts / successes / failures: 250 / 250 / 0
- Warning-free / warned cases: 247 / 3
- Complete STLs processed: 250/250
- Native-scale reference build-volume fits: 250/250
- Organic support paths generated: 250/250
- Layers: median 450, IQR 423-489, range 309-677
- Positive-extrusion movements: median 174104, IQR 146955.75-206327.75, range 45667-317816
- Estimated time: median 7488.5 s, IQR 6307.25-8725.5 s, range 2568-13419 s
- Earlier combined mesh-integrity failures that met slicer criteria: 22/22

## Warnings

- Case 803: print warning: Empty layer between 110.8 and 111.4.
- Case 821: print warning: Empty layer between 129 and 130.
- Case 822: print warning: Empty layer between 108.4 and 109.

## Integrity and consistency

- The per-case CSV, summary JSON, validation manifest, held-out Phase A cohort, and corrected Phase B cohort use the same 250 case IDs.
- Every required success field is populated with a finite, non-placeholder value.
- The profile hash and PrusaSlicer version agree across the retained artifacts.
- Recorded STL and G-code hashes were checked for 500 payloads during final verification.
- No discrepancy was found between recomputed statistics and the committed summary.

No physical fabrication was performed. This report verifies software-level production-slicer execution under one fixed computational reference profile only.
