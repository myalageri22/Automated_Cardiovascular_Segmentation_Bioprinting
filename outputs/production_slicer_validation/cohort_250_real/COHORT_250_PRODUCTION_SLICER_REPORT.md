# Full-cohort production-slicer validation report

## Intended use

This document is the source-of-truth brief for integrating the completed 250-case
production-slicer validation into the manuscript titled:

> A Computational Image-to-Mesh Pipeline for Cardiac Vascular Segmentation and
> Bioprinting-Oriented STL Generation from CT Angiography.

Only the verified values below should be used. This was a computational
toolpath-generation validation. No physical printer or physical fabrication was
used. It does not establish physical print quality, anatomical-scale
printability, bioink compatibility, lumen patency, perfusion, mechanical
behavior, quantitative surface fidelity, cell viability, endothelialization, or
biological function.

The clDice--mesh-fragmentation relationship remains the principal scientific
contribution. The production-slicer result is supporting fabrication-oriented
computational evidence.

## Validation scope

- Cohort: complete held-out test cohort, cases 751--1000.
- Repaired STLs discovered: 250.
- Repaired STLs attempted: 250.
- Source pattern during the retained run:
  `../outputs/phase_b_mesh_qc/case_outputs/*/segmentation_repaired.stl`
- Software: PrusaSlicer 2.9.6.
- Computational reference printer: Original Prusa MK4S.
- Printer build volume: 250 x 210 x 220 mm.
- Filament profile: Generic PLA.
- Nozzle: 0.4 mm.
- Print preset: `0.20mm STRUCTURAL @MK4S 0.4`.
- Layer height: 0.20 mm.
- Perimeters: 2.
- Infill: 15%.
- Object scale: native 100%.
- Source STL orientation: retained.
- Supports: automatic organic supports enabled everywhere.
- Binary G-code: disabled; textual G-code required.
- Profile SHA-256:
  `eb43615821fc138e5c6ec7f566b734a1492cf6aed62cabbc4b1139b78c7cf4dd`.
- Run interval: 2026-07-24T06:52:29Z to 2026-07-24T07:04:12Z.
- Aggregate recorded slicer subprocess runtime: 405.203369 seconds.
- Git commit recorded by the run:
  `a8135a5cdca77b9dbb2eda31f9304bf6e7f0d7ca`.
- The working tree was dirty; no commit or push was performed.

## Prospective success definition

Toolpath-generation success required all four conditions:

1. PrusaSlicer exit code zero.
2. A nonempty textual G-code file.
3. At least one detected printed layer.
4. At least one spatial motion command with positive extrusion.

Warnings were retained separately and did not automatically convert an otherwise
valid toolpath into a failure.

## Verified cohort results

| Measure | Verified result |
|---|---:|
| Repaired STLs evaluated | 250 |
| PrusaSlicer exit code zero | 250/250 |
| Nonempty textual G-code generated | 250/250 |
| Toolpaths meeting all prospective success criteria | 250/250 (100.0%) |
| Failed toolpaths | 0 |
| Build-volume fit at native scale | 250/250 |
| Complete STL processed | 250/250 |
| Organic support paths generated | 250/250 |
| Warning-free cases | 247/250 |
| Cases with PrusaSlicer warnings | 3/250 |
| Layers, median (IQR; range) | 450 (423--489; 309--677) |
| Positive extrusion movements, median (range) | 174,104 (45,667--317,816) |
| Positive object-extrusion movements, median (range) | 115,091.5 (28,752--208,448) |
| Support-material extrusion movements, median (range) | 55,761 (15,808--114,018) |
| Support-interface extrusion movements, median (range) | 3,353.5 (722--8,282) |
| Layers containing support extrusion, median (range) | 431 (290--662) |
| Estimated time, median (IQR; range) | 7,488.5 s (6,307.25--8,725.5; 2,568--13,419 s) |
| Estimated time, readable median | approximately 2 h 4 min 49 s |
| Filament length, median (IQR; range) | 7,023.655 mm (6,059.263--8,375.203; 1,945.72--14,869.49 mm) |
| Material volume, median (IQR; range) | 16.895 cm3 (14.573--20.145; 4.68--35.77 cm3) |
| Material mass, median (IQR; range) | 20.95 g (18.075--24.978; 5.80--44.35 g) |

All 250 G-code files were textual and nonempty. Layer detection used
PrusaSlicer `;LAYER_CHANGE` markers for every case. All 250 STL and G-code files
have recorded SHA-256 hashes in `per_case_slicer_results.csv`.

## Warnings

PrusaSlicer generated one empty-layer warning for each of three successful cases:

- Case 803: `Empty layer between 110.8 and 111.4.`
- Case 821: `Empty layer between 129 and 130.`
- Case 822: `Empty layer between 108.4 and 109.`

Each of these cases still returned exit code zero, produced nonempty textual
G-code, contained detected layers, and contained positive extrusion movements;
therefore, each met the prospective toolpath-generation success definition.
These warnings should be disclosed and should not be described as warning-free
toolpaths.

## Relationship to mesh-integrity QC

The earlier combined Phase B mesh-integrity criterion passed 228/250 meshes.
Its 22 failures were the 22 non-watertight repaired meshes in the recorded Phase B table.
Nevertheless, all 22 produced successful PrusaSlicer toolpaths under the
prospective production-slicer criteria. This shows that mesh-integrity QC and
production-slicer execution measure different properties; successful slicer
execution does not override a geometric QC failure. It does not demonstrate
physical fabrication success.

The 22 mesh-integrity-failing cases were:

`767, 768, 769, 772, 786, 798, 808, 821, 845, 855, 856, 861, 867, 868, 870,
873, 904, 925, 944, 959, 962, 995`.

## Manuscript-ready Methods text

Repaired Phase B STL outputs for the complete 250-case held-out test cohort were
processed using PrusaSlicer 2.9.6 with a fixed Original Prusa MK4S standardized
computational reference configuration, Generic PLA, a 0.4 mm nozzle, and the
`0.20mm STRUCTURAL @MK4S 0.4` preset. Native 100% scale and the source STL
orientation were retained, with automatic organic supports enabled everywhere.
Toolpath-generation success was predefined as a zero slicer exit status,
generation of a nonempty textual G-code file, detection of at least one printed
layer, and the presence of positive extrusion movements. Configuration and
artifact hashes, commands, logs, warnings, and per-case results were retained.
No physical printer or physical fabrication was used.

## Manuscript-ready Results text

Production-slicer validation evaluated all 250 repaired held-out test STLs.
All 250 cases generated nonempty textual G-code and met the predefined
toolpath-generation criteria (250/250, 100.0%), with zero failures. The median
layer count was 450 (IQR 423--489), the median estimated time was 7,488.5 s
(approximately 2 h 4 min 49 s; IQR 6,307.25--8,725.5 s), and the median
estimated material use was 7,023.655 mm of filament, 16.895 cm3, or 20.95 g.
Organic support paths were generated for every case. Three successful cases
(803, 821, and 822) produced an empty-layer warning; the remaining 247 cases
were warning-free.

## Manuscript-ready Discussion text

The full-cohort production-slicer assessment provides supporting computational
evidence that repaired Phase B STLs can be converted to textual executable
toolpaths under one fixed reference profile. The 22 meshes that failed the
earlier combined mesh-integrity criterion nevertheless generated toolpaths,
confirming that geometric QC and slicer execution measure different criteria. This
supporting result does not change the principal finding that topology-aware
clDice is more strongly associated with mesh fragmentation than volumetric Dice.

## Manuscript-ready limitation update

Although all repaired held-out test STLs generated toolpaths under one fixed
PrusaSlicer reference configuration, no physical printer or physical fabrication
was used. Consequently, physical print quality, quantitative surface fidelity,
lumen patency, bioink-specific resolution, perfusion, mechanical behavior, cell
viability, endothelialization, and biological function remain untested.

## Suggested compact LaTeX table

```latex
\begin{table}[!htbp]
\centering
\caption{Full-cohort production-slicer validation of repaired Phase B STL outputs.}
\label{tab:production-slicer-validation}
\begin{tabular}{ll}
\toprule
Metric & Verified result \\
\midrule
Repaired STLs evaluated & 250 \\
Successful toolpaths & 250/250 (100.0\%) \\
Failed toolpaths & 0 \\
Warning-free / warned cases & 247 / 3 \\
Layers, median (IQR) & 450 (423--489) \\
Estimated time, median (IQR) & 7488.5 s (6307.25--8725.5 s) \\
Filament length, median (IQR) & 7023.655 mm (6059.263--8375.203 mm) \\
Material mass, median (IQR) & 20.95 g (18.075--24.978 g) \\
\bottomrule
\end{tabular}
\end{table}
```

## Source files and hashes

- `per_case_slicer_results.csv`
  - SHA-256:
    `5b920ba63f2d29572207374b412dc958587088e2f047fb8de59167f5544b12a3`
- `summary_slicer_results.json`
  - SHA-256:
    `8f9fa187608344ffb66a24581bb00845eb43484e22d5b6b57f4f3460fce9cb36`
- `validation_manifest.json`
  - SHA-256:
    `9dde4f913025b2ac647ca2f638a45c92b1866d45c76c1e754cf232c4fdcce2e4`
- Total G-code size: 2,142,447,254 bytes (approximately 1.995 GiB).

## Important folder distinction

Use only `cohort_250_real/` for publication results.

The sibling `cohort_250/` directory records an earlier prerequisite failure
caused by using an obsolete PrusaSlicer executable path. It generated no G-code
and is not a production-slicer result. The preliminary folder must not be used
for manuscript values.
