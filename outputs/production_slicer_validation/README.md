# Production-slicer validation

## Purpose and scope

This workflow tests whether repaired Phase B STL files can be converted by the real PrusaSlicer command-line interface into executable textual G-code. It is a production-slicer and toolpath-generation validation, not a biological or bioink validation. The repository's existing mesh checks remain geometric proxies and are reported separately.

## Prerequisites

- The verified executable on this Mac is `/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer` (PrusaSlicer 2.9.6).
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

After the ignored local Phase B case outputs have been restored beneath `outputs/phase_b_mesh_qc/case_outputs/`, run:

```bash
python production_slicer_validation.py --input-glob 'outputs/phase_b_mesh_qc/case_outputs/*/segmentation_repaired.stl' --prusaslicer '/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer' --profile outputs/production_slicer_validation/reference_profile_mk4s_0.4_pla_0.20_organic_auto.ini --output-dir outputs/production_slicer_validation/reference_profile_cohort --expected-cohort-size 250
```

Do not rerun segmentation merely to recreate ignored STL files. Restore the local Phase B outputs instead. The current exhaustive audit found no cohort repaired STLs and no cached locked-final prediction volumes suitable for faithful mesh regeneration; therefore this command has not been run.

## Input and outputs

Inputs are repaired STL files and one fixed exported PrusaSlicer configuration. Discovery is sorted deterministically. The workflow records per-case CSV results, a JSON summary, a hash manifest, the slicer version, the exact invocation, standard-output and standard-error logs, and textual G-code. Large G-code is narrowly ignored by Git; compact evidence remains retainable.

Production-slicer success requires all four conditions:

1. PrusaSlicer exits with status zero.
2. A nonempty textual G-code file is generated.
3. At least one printed layer is detected.
4. At least one spatial motion with positive extrusion is detected.

Layer markers are preferred; print-Z changes during positive-extrusion motion are the fallback. PrusaSlicer `;TYPE:` comments classify object, support-material, and support-interface extrusion paths. If role comments are absent, these classifications remain unavailable rather than being guessed. Estimated time and total material are parsed only from recognized PrusaSlicer comments. Missing or ambiguous fields remain null.

The corrected primary result is explicitly reported as `1/1 available representative case`; no cohort success percentage is emitted. The earlier no-automatic-support run is retained at the output root as a preliminary run and is not the primary reference result.

## Retained publication evidence

The available standalone case-720 Phase B report records `720.label.nii.gz`
as its segmentation input. Case 720 belongs to the validation partition and is
not part of the held-out 250-case test metrics. The verified reference-profile
run used the repaired STL at native 100% scale; no rescaled phantom result is
reported.

Physical fabrication was intentionally not performed. The compact publication
record is `case720_verified_summary.json`; full G-code, the original
case-level STL, medical images, predictions, and private model artifacts are
intentionally excluded.
