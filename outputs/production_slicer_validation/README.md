# Production-slicer validation

## Purpose and scope

This workflow tests whether repaired Phase B STL files can be converted by the real PrusaSlicer command-line interface into executable textual G-code. It is a production-slicer and toolpath-generation validation, not a biological or bioink validation. The repository's existing mesh checks remain geometric proxies and are reported separately.

## Prerequisites

- The retained run used PrusaSlicer 2.9.6. Supply the local PrusaSlicer executable path with `--prusaslicer`.
- Open PrusaSlicer, select the actual printer, nozzle, and PLA profile, then choose **File → Export → Export Config**.
- Disable binary G-code before export. The initial verified merged profile is `printer_profile_text_gcode.ini`. The primary reference experiment uses the minimally derived `reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini`, which changes only automatic support placement and support style; see `reference_profile_diff.json`.

The verified publication run must use the PrusaSlicer version recorded in `slicer_version.txt`. No slicer version is claimed until the executable has been queried successfully.

## Representative case-720 command

Run from the repository root after the profile and PrusaSlicer are available:

```bash
python production_slicer_validation.py --stl /path/to/case720/vessels_repaired.stl --prusaslicer /path/to/PrusaSlicer --profile outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini --output-dir outputs/production_slicer_validation/reference_profile_case720 --case-filter 720 --expected-cohort-size 250
```

For a prerequisite-only check, add `--dry-run`. A dry run is not a successful slicer validation.

## Cohort command

The completed publication run is retained under `cohort_250_real/`. To repeat it only when a new experiment is explicitly intended and the excluded Phase B case outputs have been restored beneath `outputs/phase_b_mesh_qc/case_outputs/`, run:

```bash
python production_slicer_validation.py --input-glob 'outputs/phase_b_mesh_qc/case_outputs/*/segmentation_repaired.stl' --prusaslicer /path/to/PrusaSlicer --profile outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini --output-dir outputs/production_slicer_validation/cohort_250_real --expected-cohort-size 250 --timeout-seconds 600
```

Do not rerun segmentation merely to recreate ignored STL files. Restore the local Phase B outputs instead. The completed cohort does not require another experimental run for publication verification; use `python verify_production_slicer_cohort.py` to recompute the retained statistics and consistency checks.

## Input and outputs

Inputs are repaired STL files and one fixed exported PrusaSlicer configuration. Discovery is sorted deterministically. The workflow records per-case CSV results, a JSON summary, a hash manifest, the slicer version, the exact invocation, standard-output and standard-error logs, and textual G-code. Large G-code is narrowly ignored by Git; compact evidence remains retainable.

Production-slicer success requires all four conditions:

1. PrusaSlicer exits with status zero.
2. A nonempty textual G-code file is generated.
3. At least one printed layer is detected.
4. At least one spatial motion with positive extrusion is detected.

Layer markers are preferred; print-Z changes during positive-extrusion motion are the fallback. PrusaSlicer `;TYPE:` comments classify object, support-material, and support-interface extrusion paths. If role comments are absent, these classifications remain unavailable rather than being guessed. Estimated time and total material are parsed only from recognized PrusaSlicer comments. Missing or ambiguous fields remain null.

The validator emits a cohort percentage only when the full expected cohort is discovered and attempted. Representative and subset runs remain explicitly labeled and do not emit a cohort success percentage. Warnings are reported separately from failures.

## Retained publication evidence

The sole source of the reported 250-case production-slicer values is `cohort_250_real/`:

- `per_case_slicer_results.csv`: 250 normalized, publication-safe per-case rows.
- `summary_slicer_results.json`: deterministic cohort counts and inclusive-quartile summaries.
- `validation_manifest.json`: input hashes, PrusaSlicer version, profile hash, and prospective success definition.
- `COHORT_250_PRODUCTION_SLICER_REPORT.md`: concise run report and scope boundary.
- `VERIFICATION_REPORT.md` and `verification_report.json`: independently recomputed statistics and cross-artifact checks.
- `run_command.txt` and `slicer_version.txt`: repository-relative command template and version record.

The canonical profile is not duplicated inside the cohort folder. It remains `reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini` with SHA-256 `eb43615821fc138e5c6ec7f566b734a1492cf6aed62cabbc4b1139b78c7cf4dd`.

The standalone case-720 Phase B report records `720.label.nii.gz` as its segmentation input. Case 720 belongs to the validation partition and is not part of the held-out 250-case test cohort. Its separate illustrative analysis uses threshold 0.4; the cohort uses threshold 0.5. The verified case-level reference-profile run used the repaired STL at native 100% scale and remains useful only as an illustrative example.

Physical fabrication was intentionally not performed. The compact publication
record excludes full G-code, original case-level STL collections, medical images,
predictions, and private model artifacts. The per-case hashes, parsed output
records, manifest, profile, version, warnings, and verification results are
retained so the manuscript claims remain auditable.
