# Topology-aware postprocessing for patient-specific 3D coronary reconstruction

Deterministic postprocessing of **frozen** coronary segmentation predictions,
evaluated for connectivity benefit against anatomical fidelity cost.

Research question:

> Can an automated topology-aware postprocessing stage reduce segmentation-derived
> coronary fragmentation while preserving the original patient-specific geometry?

This is a medical-image reconstruction and postprocessing experiment. It makes no
claim about physical printability, clinical deployment, diagnostic benefit, or
segmentation state of the art.

---

## Status

The code is complete and validated on synthetic phantoms. **It has not been run
on the held-out cohort**, because the frozen per-case predictions are not present
in this working copy — see `KNOWN_ISSUES.md` item 6 and
`analysis/cmig_robustness/RECOVERY_RUNBOOK.md`. No cohort number in this
directory is real; there are none.

Check what is reachable without computing anything:

```bash
python3 run_topology_correction_experiment.py --dry-run
```

It resolves every case in the test split, hashes what it finds, writes
`input_manifest.json` and `missing_inputs.csv`, and exits 2 if anything is
missing.

---

## What is and is not touched

Read-only, never written: `outputs/final_test_250/`, `outputs/phase_b_mesh_qc/`,
`outputs/cmig_robustness/`, `analysis/cmig_robustness/`, `paper/`, `checkpoints/`,
`extra_information/`. `topocorr.io_utils.assert_not_protected` raises before any
write that would land inside them, and the runner fingerprints every authoritative
file with SHA-256 before and after the run and fails loudly on any change.

Reused unmodified, by import rather than by copy:

| Component | Source |
|---|---|
| Phase B reconstruction, repair, smoothing, QC | `phaseb_mesh_qc.run_phaseb_for_case` |
| Cross-sectional contour closure | `phaseb_mesh_qc.slicability_plane_check` |
| clDice | `compute_cldice.cldice` |
| Physical-radius binary closing | `phaseb.src.phaseb.postprocess.binary_closing_mm` |

All four are SHA-256 hashed into `run_manifest.json`, so a later reader can prove
which version produced a result.

The control (Strategy 0) is the uncorrected prediction put through that same
reconstruction, so any difference between arms comes from the correction alone.

---

## Correction strategies

Prespecified in `config/experiment_config.yaml`. Every variant is reported; the
`primary` variant of each strategy carries the headline comparison and the rest
form the sensitivity analysis. Nothing here is selected using a held-out endpoint.

| Strategy | Mechanism | Grid | Primary |
|---|---|---|---|
| `s0_original` | none (control) | - | - |
| `s1_absolute_volume` | drop components below an absolute physical volume | 0.216, 1, 2, 5, 10, 27, 50 mm^3 | **5 mm^3** |
| `s2_relative_volume` | drop components below a fraction of the largest | 0.001, 0.005, 0.01, 0.05 | **0.01** |
| `s3_gap_bridge` | prefilter, then bridge short gaps with a minimal connector | 0.6, 1.2, 1.8, 2.4 mm | **1.2 mm** |
| `s3c_closing` | prefilter, then binary closing at a small physical radius | 0.6, 1.2 mm | **0.6 mm** |

The 5 mm^3 primary is geometric, not empirical: at 0.6 mm isotropic a voxel is
0.216 mm^3, and a 1 mm diameter, 5 mm long coronary segment is about 3.9 mm^3, so
components below roughly 4 mm^3 cannot represent a resolvable vessel segment. The
50 mm^3 entry is the repository's own legacy `min_component_mm3` default, which
was supported but never applied to the reported cohort; it is included for
continuity, not because it is defensible.

Two safeguards are deliberate. No filter ever empties a non-empty mask. And no
strategy forces a mask to one component: the left and right coronary systems may
legitimately be separate, and the phantom suite asserts they are never merged.

### Gap bridging

Deterministic, no randomness, no ground truth:

1. label components; ignore anything below `min_component_mm3_to_bridge` (5 mm^3), so speckle is never bridged;
2. for every eligible pair compute the minimum surface-to-surface distance **in mm** and the closest voxel pair;
3. sort candidates by `(gap, -smaller component volume, label_a, label_b)` — a total order independent of labelling;
4. walk that list with a union-find, bridging only when the pair is not already connected and the gap is within tolerance;
5. each bridge is the straight discrete segment between the closest voxel pair, dilated to `bridge_radius_mm = 0.3`, which at 0.6 mm spacing is a single-voxel-thick connector: the least invasive bridge the grid can represent.

---

## Running it

```bash
pip install -r requirements_topology_correction.txt

# validate the implementation without any cohort data
python3 -m unittest discover -s tests -v
python3 run_topology_correction_experiment.py --mode phantom

# the real experiment, once the frozen predictions are restored
python3 run_topology_correction_experiment.py \
    --config config/experiment_config.yaml \
    --mesh-variants primary
```

`--mesh-variants primary` reconstructs meshes for the primary variant of each
strategy; `all` reconstructs every grid point (far slower). Sensitivity variants
are fully evaluated in the voxel domain either way. Other flags: `--limit`,
`--cases`, `--no-geometry`, `--no-figures`, `--no-failure-analysis`,
`--allow-provenance-mismatch`.

Exit codes: `0` success, `2` inputs missing or unusable, `3` environment
self-test failed (a broken skeletoniser would invalidate every clDice, so the run
refuses to start).

### Guarantees the runner enforces

- every case in the split is attempted; failures are recorded in `failed_cases.csv` with a reason, never dropped;
- the segmentation threshold is read from the config and never tuned, and `qc_checklist.json` records it;
- each recomputed control Dice is checked against `outputs/final_test_250/per_case_metrics.csv`, and the cohort mean is checked against the archived 0.7820-0.7935 interval, so a recovered artifact from a different run cannot be mixed in unnoticed;
- every figure writes the exact data it plots to `<figure>_source_data.csv`.

---

## Outputs

Written to `outputs/topology_correction/` (phantom mode writes to
`outputs/topology_correction_phantom_validation/`):

```
experiment_config.yaml      resolved configuration actually used
input_manifest.json         per-case inputs with SHA-256 hashes
component_audit.csv         Stage A: every component of every case, pre-modification
per_case_original.csv       control arm
per_case_corrected.csv      every strategy variant
paired_comparison.csv       case-paired original vs corrected, with deltas
cohort_summary.csv          one row per variant
sensitivity_analysis.csv    the same rows keyed by correction strength
statistical_tests.csv       paired tests, effect sizes, 95% CIs, BH-adjusted P
stratified_analysis.csv     prespecified quantile and anatomical strata
covariate_analysis.csv      exploratory: does baseline quality predict benefit
outcome_classification.csv  neutral Outcome A / B / C label per variant
tables/                     manuscript-ready primary table, CSV and Markdown
figures/                    Figures A-E and S1, each with its source CSV
failure_analysis/           representative cases, successes and failures alike
logs/                       run log, environment self-test, authoritative fingerprints
qc_checklist.json           the section 22 checklist, machine-readable
run_manifest.json           versions, seed, config hash, code hashes, timing, outputs
```

---

### The committed phantom validation record

`outputs/topology_correction_phantom_validation/` holds a complete phantom run:
6 phantoms, 18 variants, 0 failures, authoritative artifacts verified unchanged.
It was produced with `--no-failure-analysis` only because the machine it was run
on caps a single command at 45 s; the rendering path itself is exercised by
`topocorr.failure_analysis.render_case` and completes in about 1 s per case.
Reproduce the full record with:

```bash
python3 run_topology_correction_experiment.py --mode phantom --mesh-variants primary
```

## Analysis design

Every comparison is paired by case against the control. Continuous endpoints get
Wilcoxon signed-rank unless Shapiro-Wilk on the paired differences supports a
t-test, with the choice recorded per row; degenerate near-constant differences
are forced to the distribution-free test. Paired binary outcomes such as mesh
integrity get exact McNemar. Every row carries an effect size (matched-pairs
rank-biserial, or Cohen's dz, or the paired rate difference) and a seeded
percentile bootstrap 95% CI. Benjamini-Hochberg FDR is applied within each family
of endpoints, and the family and its size are recorded. Magnitude is always
reported alongside significance.

Stratification is by tertiles of the original component count (rule prespecified,
cut values reported) and by the anatomical split at 2 components. The covariate
analysis asking whether baseline Dice, clDice, HD95 or component count predicts
who benefits is labelled exploratory in the output.

Representative cases are chosen by fixed rules, and the rule that produced each
one is written next to it in `failure_analysis/selected_cases.csv`. The
categories force failures into view: `no_improvement` and `largest_fidelity_loss`
are selected on every run.

## Interpretation

`topocorr.report.classify_outcome` assigns Outcome A (topology improves at
negligible cost), B (improves but fidelity degrades) or C (little benefit) from
thresholds fixed before any result was seen and written to
`interpretation_rules.json`: a median reduction of at least one component,
against a Dice loss no worse than 0.01, an HD95 increase no worse than 0.5 mm,
and a mean surface displacement no worse than 0.6 mm (one voxel). All three
outcomes are informative. B would establish a genuine topology-fidelity tradeoff;
C would argue that topology has to be preserved during training or inference
rather than repaired downstream.

## Secondary toolpath check (Section 12)

`run_toolpath_experiment.py` slices every repaired STL of the control and of
each primary variant under one fixed profile; `analyze_toolpath.py` runs the
paired control-vs-corrected comparison with the same statistics as the main
experiment. Software-level slicing only -- never evidence of physical
printability.

```bash
# 1250 slices per profile (S0 control + 4 primary variants x 250 cases)
python run_toolpath_experiment.py --slicer <prusa-slicer binary> \
    --mesh-root ../../outputs/topology_correction/mesh_qc \
    --out ../../outputs/topology_correction/toolpath \
    --profile-name P1_no_supports
python run_toolpath_experiment.py --slicer <prusa-slicer binary> ... \
    --profile-json p2.json --profile-name P2_supports_brim
python analyze_toolpath.py --config config/experiment_config.yaml
python build_final_report.py            # folds the results into the report
```

Slicer used here: PrusaSlicer 2.8.1 official Linux x64 AppImage. The
authoritative cohort run used 2.9.6, which publishes no Linux binary on its
release page, and its per-case results are not in the working copy -- so this
is a fresh paired comparison, not a reproduction of the 2.9.6 numbers.

Results: under P1 (no supports, no brim) the slicer rejects a minority of cases
in every arm with "an object with no extrusions in the first layer" -- control
225/250, corrected 230-234/250 (S2 exact McNemar p = 0.022, 11 gained / 2 lost).
Under P2 (supports + 3 mm brim) all arms reach 250/250, so those rejections
track the profile, not a defect of the reconstructed surface.
