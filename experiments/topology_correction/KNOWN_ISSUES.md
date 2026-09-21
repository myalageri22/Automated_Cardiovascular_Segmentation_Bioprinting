# Environment compatibility gates

Environment conditions verified automatically at the start of every
topology-correction run. Each has a gate that aborts the run if the condition
is not met, so no reported artifact can be produced under a failing backend.

---

## 1. scikit-image 0.25.2 empties symmetric even-sided solid blocks under `skeletonize`

**Status:** upstream edge case. Does not affect the reported cohort numbers.
**Gate:** `topocorr.seg_metrics.verify_skeletonisation_backend()`, run at the start
of every experiment. A failure aborts with exit code 3.

A topology-preserving thinning must never delete an entire connected component.
With scikit-image 0.25.2 the 3D (Lee) `skeletonize` violates that invariant for
perfectly symmetric, even-sided solid blocks:

| input (in a 40^3 volume) | input voxels | skeleton voxels |
|---|---|---|
| solid cube, side 2 | 8 | **0** |
| solid cube, side 4 | 64 | **0** |
| solid cube, side 6 | 216 | **0** |
| solid cube, side 10 | 1000 | **0** |
| solid cube, side 3 (odd) | 27 | 3 |
| solid cube, side 5 (odd) | 125 | 3 |

Tube-like objects, which is what coronary anatomy actually is, behave correctly:

| cylinder, length 20 | input voxels | skeleton voxels | components preserved |
|---|---|---|---|
| radius 1 | 100 | 20 | yes |
| radius 2 | 260 | 20 | yes |
| radius 3 | 580 | 18 | yes |
| radius 4 | 980 | 16 | yes |

**Consequence for the manuscript: none.** `compute_cldice.py` calls the same
function, but coronary masks are tubular, not symmetric solid blocks, and the
reported cohort clDice of 0.8695 is consistent with a functioning skeletoniser.
The artifact matters only for synthetic box phantoms, which is why the phantom
suite uses cylinders.

**What was changed:** nothing in the authoritative code. The experiment gates on
a cylinder self-test and records the result in `run_manifest.json`.

---

## 2. Meshes reloaded from STL are never watertight, so `mesh.volume` is NaN

**Status:** fixed inside this experiment. Affects any *future* analysis that
reloads the archived STLs.
**Where:** `topocorr.geometry._load`.

STL is a face soup: each triangle stores its own copy of every vertex. A mesh
loaded with `trimesh.load(..., process=False)` therefore has no face adjacency,
reports `is_watertight == False`, and `mesh.volume` returns NaN.

`phaseb_mesh_qc.run_phaseb_for_case` is **not** affected: it evaluates
`_mesh_summary` on the in-memory mesh built directly from marching cubes, whose
vertices are already shared. The archived per-case volumes are therefore valid.

Any downstream script that reloads `segmentation_repaired.stl` and reads
`.volume` or `.is_watertight` will silently get NaN / False unless it calls
`mesh.merge_vertices()` first. This experiment does. Welding identical
coordinates moves no vertex, so the surface is unchanged.

---

## 3. `trimesh` needs `rtree` for exact surface distance

**Status:** dependency documented in `requirements_topology_correction.txt`.
**Where:** `topocorr.geometry._point_to_surface`.

Without `rtree`, `trimesh.proximity.closest_point` raises `ModuleNotFoundError`
and the code falls back to a nearest-**vertex** KD-tree query. That fallback
carries a positive floor set by the sample spacing: two *identical* box surfaces
scored a mean deviation of **3.83 mm**. With exact point-to-triangle distance the
same pair scores **5.1e-16 mm**.

The floor is of the same order as the sub-millimetre distortions the experiment
must resolve, so an unnoticed fallback would make a genuinely distorting method
look indistinguishable from a faithful one. The method actually used is recorded
per row in the `distance_method` column; rows reading
`approximate_point_to_vertex` must not be compared against exact rows.

---

## 4. Floating-point boundary in the gap-bridging tolerance

**Status:** fixed. Found by the phantom suite.
**Where:** `topocorr.strategies.bridge_short_gaps`.

A surface-to-surface gap of exactly N voxels evaluates to
`N * 0.6 = 1.8000000000000003` mm, so the original `gap <= max_gap_mm` test was
silently *exclusive* at every round parameter value: a phantom with an exact
1.2 mm gap was not bridged at the 1.2 mm setting. Every strategy threshold in
this experiment sits on a voxel multiple, so this affected the primary setting
of Strategy 3. Fixed with an explicit `tolerance_mm = 1e-6`, recorded in the
per-case provenance.

---
