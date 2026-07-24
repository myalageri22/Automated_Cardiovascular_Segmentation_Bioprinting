# Traceable v14 Phase B missing-checks report

## Scope and claim boundary

This report records the completed 250-case held-out analysis for the retained,
traceable local v14 checkpoint. It does **not** recreate the unavailable
ephemeral checkpoint originally referenced by the manuscript, and the results
must not be attributed to that checkpoint.

The run evaluates computational geometry only. It does not establish physical
printing, slicer toolpath success, lumen patency, perfusion, biological
viability, or clinical readiness.

## Provenance

- Cohort: all 250 unique case IDs in `splits.json` under `test`
- Checkpoint:
  `outputs/train_runs/bioprint_v14_tversky_fn_local_rebuild_mps_lowmem/checkpoints/best_dice05.pt`
- Checkpoint epoch: 79
- Checkpoint recorded best metric: 0.7888903796672821
- Checkpoint SHA-256:
  `792178759e9e579eb65f89a5a63f6538d7d0202da7e8eeeefda4825bfffc743b`
- Inference threshold: 0.5
- ROI: 96 x 192 x 192
- Sliding-window overlap: 0.625
- Model: Attention U-Net
- Device: Apple MPS, with output accumulation on CPU
- Mesh coordinate convention: NIfTI RAS world coordinates
- Sampling planes per mesh: 50, along the longest mesh bounding-box axis
- Checker SHA-256:
  `eef65d749f42819c366291fb964fc10e0d064096d3bb15558aea1d8bccb95f8e`
- Result CSV SHA-256:
  `84fd1656fd97a05c2381bd043ea5463a650e2c9e07db09b2a558c87870d12cf6`

The focused Phase B mode retained the exact thresholded mask and repaired,
affine-correct STL needed for these checks. Unrelated mesh smoothing,
wall-thickness distance transforms, and fabrication proxy calculations were
not recomputed for the later cases.

## Cohort integrity audit

- Evaluation rows: 250
- Unique evaluated cases: 250
- Successful evaluation rows: 250
- Failed evaluation rows: 0
- Masks: 250
- Repaired meshes: 250
- Missing mask/mesh pairs: 0
- Duplicate case IDs: 0
- Extra cases outside the test split: 0
- Test-split cases omitted: 0
- Missing-check rows with status `ok`: 250

## Results

### 1. Slicability

The supplied check defines its primary fraction over planes that actually
intersect the mesh.

- Closed contours on every intersecting plane: 125/250 (50.0%)
- Mean closed fraction among intersecting planes: 95.2%
- Median closed fraction among intersecting planes: 99.0%
- Meshes intersected by all 50 requested planes: 115/250 (46.0%)
- Strictly closed on all 50 requested planes: 61/250 (24.4%)
- Mean number of intersecting planes: 46.512/50
- Mean number of closed-intersection planes: 44.276/50
- Mean closed count as a share of all 50 requested planes: 88.552%
- Median closed count as a share of all 50 requested planes: 94.0%

Therefore, a claim that every mesh produced closed contours on every one of the
50 requested planes is not supported. The more limited statement that half the
meshes had closed contours on every plane that intersected them is supported.

### 2. Centre-of-mass agreement

- Mean mesh-to-mask centroid offset: 0.447 mm
- Median offset: 0.140 mm
- Maximum offset: 6.948 mm (case 873)
- Mean offset relative to mesh bounding-box diagonal: 0.224%
- Median relative offset: 0.078%
- Maximum relative offset: 2.994%
- Cases above 1 mm: 22/250 (8.8%)
- Cases above 2 mm: 21/250 (8.4%)
- 95th-percentile offset: 3.144 mm

The cohort shows close typical agreement but contains material outliers. Report
the distribution rather than claiming uniform sub-voxel centroid agreement.

### 3. Spatial alignment

- Bounding-box alignment within two voxels: 250/250 (100.0%)
- Mean maximum bounding-box error: 0.300 mm
- Maximum maximum-axis error: approximately 0.300 mm

This error is consistent with the half-voxel placement of the marching-cubes
isosurface for the 0.6 mm inference grid. The result supports preservation of
the NIfTI world-coordinate bounding box across mask-to-mesh export.

## Interpretation

The corrected pipeline preserves world-coordinate spatial alignment across the
entire retained traceable v14 cohort. Typical centroid agreement is also close,
although 22 cases exceed 1 mm and should be acknowledged. Universal
slicability is not demonstrated: the supplied metric passes 50.0% when
restricted to intersecting planes, and only 24.4% pass the stricter requirement
of a closed contour on all 50 requested planes.

The checker's `mask_spacing_mm` field records voxel spacing and uses it to set
the two-voxel alignment tolerance. It does not independently implement a
separate spacing round-trip reconstruction test, despite the wording in the
original pasted script.

## Primary artifacts

- `outputs/final_test_250_phaseb_traceable_v14/per_case_metrics.csv`
- `outputs/final_test_250_phaseb_traceable_v14/used_config.yaml`
- `outputs/final_test_250_phaseb_traceable_v14/eval_command.txt`
- `outputs/phase_b_mesh_qc/per_case_mesh_qc.csv`
- `outputs/phase_b_mesh_qc/missing_checks_per_case_traceable_v14.csv`
