"""Validation suite for the topology-correction experiment.

Runs entirely on synthetic data. Establishes that the implementation does what
the protocol says BEFORE it is ever pointed at the held-out cohort.

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topocorr import components as comp
from topocorr import geometry as geom
from topocorr import seg_metrics as seg
from topocorr import stats_tests as st
from topocorr import strategies as strat
from topocorr.io_utils import Volume, load_config
from topocorr.phantoms import make_phantoms

SPACING = (0.6, 0.6, 0.6)


def make_volume(arr: np.ndarray, spacing=SPACING) -> Volume:
    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = spacing[i]
    affine[:3, 3] = [-10.0, -20.0, -30.0]
    return Volume(array=arr.astype(bool), affine=affine, spacing=spacing)


class TestComponents(unittest.TestCase):
    def setUp(self):
        self.arr = np.zeros((30, 30, 30), dtype=bool)
        self.arr[5:11, 5:11, 5:11] = True      # 216 voxels
        self.arr[20, 20, 20] = True            # 1 voxel
        self.vol = make_volume(self.arr)

    def test_counts_and_volumes(self):
        recs = comp.characterise(self.vol, self.arr, "t")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].volume_rank, 1)
        self.assertEqual(recs[0].voxel_count, 216)
        self.assertAlmostEqual(recs[0].volume_mm3, 216 * 0.216, places=6)
        self.assertEqual(recs[1].voxel_count, 1)
        self.assertAlmostEqual(recs[1].volume_mm3, 0.216, places=9)

    def test_ranked_by_physical_volume(self):
        recs = comp.characterise(self.vol, self.arr, "t")
        vols = [r.volume_mm3 for r in recs]
        self.assertEqual(vols, sorted(vols, reverse=True))

    def test_world_centroid_uses_affine(self):
        recs = comp.characterise(self.vol, self.arr, "t")
        small = [r for r in recs if r.voxel_count == 1][0]
        # voxel (20,20,20) -> 20*0.6 + origin
        self.assertAlmostEqual(small.centroid_world_x, 20 * 0.6 - 10.0, places=6)
        self.assertAlmostEqual(small.centroid_world_y, 20 * 0.6 - 20.0, places=6)
        self.assertAlmostEqual(small.centroid_world_z, 20 * 0.6 - 30.0, places=6)

    def test_largest_component_has_infinite_gap(self):
        recs = comp.characterise(self.vol, self.arr, "t")
        self.assertFalse(np.isfinite(recs[0].distance_to_nearest_larger_mm))
        self.assertTrue(np.isfinite(recs[1].distance_to_nearest_larger_mm))

    def test_gap_is_measured_in_millimetres(self):
        arr = np.zeros((20, 20, 20), dtype=bool)
        arr[5, 5, 5] = True
        arr[5, 5, 9] = True            # 4 voxel steps == 2.4 mm
        arr[2:5, 2:5, 2:5] = True      # a larger component
        vol = make_volume(arr)
        recs = comp.characterise(vol, arr, "t")
        gaps = [r.distance_to_nearest_larger_mm for r in recs if np.isfinite(r.distance_to_nearest_larger_mm)]
        self.assertTrue(all(g > 0 for g in gaps))
        self.assertAlmostEqual(min(gaps), np.sqrt(3) * 0.6, places=6)


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.phantoms = {p.case_id: p for p in make_phantoms()}

    def test_identity_is_exactly_identity(self):
        ph = self.phantoms["phantom_mixed"]
        out, _ = strat.apply_strategy(ph.prediction, ph.spacing, "identity", {})
        np.testing.assert_array_equal(out, ph.prediction)

    def test_absolute_filter_removes_only_below_threshold(self):
        ph = self.phantoms["phantom_speckle"]
        out, info = strat.apply_strategy(ph.prediction, ph.spacing,
                                         "absolute_volume_filter", {"min_volume_mm3": 5.0})
        self.assertLess(info["components_after"], info["components_before"])
        # every surviving component must be at or above the threshold
        labeled, n = comp.label_components(out)
        from scipy import ndimage
        vols = np.asarray(ndimage.sum(out, labeled, index=range(1, n + 1))) * 0.216
        self.assertTrue((vols >= 5.0 - 1e-9).all())

    def test_filter_never_empties_a_nonempty_mask(self):
        ph = self.phantoms["phantom_clean"]
        out, _ = strat.apply_strategy(ph.prediction, ph.spacing,
                                      "absolute_volume_filter", {"min_volume_mm3": 1e9})
        self.assertGreater(out.sum(), 0)

    def test_two_legitimate_systems_are_not_merged(self):
        ph = self.phantoms["phantom_two_systems"]
        for kind, params in (("absolute_volume_filter", {"min_volume_mm3": 5.0}),
                             ("relative_volume_filter", {"min_fraction_of_largest": 0.01}),
                             ("gap_bridge", {"max_gap_mm": 2.4, "prefilter_mm3": 5.0,
                                             "bridge_radius_mm": 0.3,
                                             "min_component_mm3_to_bridge": 5.0,
                                             "max_bridges_per_case": 50})):
            out, info = strat.apply_strategy(ph.prediction, ph.spacing, kind, params)
            self.assertGreaterEqual(info["components_after"], 2,
                                    f"{kind} collapsed two separate coronary systems into one")

    def test_short_gap_bridged_at_primary_tolerance(self):
        ph = self.phantoms["phantom_short_gap"]
        out, info = strat.apply_strategy(
            ph.prediction, ph.spacing, "gap_bridge",
            {"max_gap_mm": 1.2, "prefilter_mm3": 5.0, "bridge_radius_mm": 0.3,
             "min_component_mm3_to_bridge": 5.0, "max_bridges_per_case": 50})
        self.assertGreater(info["bridges_added"], 0)
        self.assertLess(info["components_after"], info["components_before"])

    def test_gap_below_tolerance_is_not_bridged(self):
        ph = self.phantoms["phantom_short_gap"]
        _, info = strat.apply_strategy(
            ph.prediction, ph.spacing, "gap_bridge",
            {"max_gap_mm": 0.6, "prefilter_mm3": 5.0, "bridge_radius_mm": 0.3,
             "min_component_mm3_to_bridge": 5.0, "max_bridges_per_case": 50})
        self.assertEqual(info["bridges_added"], 0)

    def test_wide_gap_never_bridged_at_any_tested_tolerance(self):
        ph = self.phantoms["phantom_wide_gap"]
        for gap in (0.6, 1.2, 1.8, 2.4):
            _, info = strat.apply_strategy(
                ph.prediction, ph.spacing, "gap_bridge",
                {"max_gap_mm": gap, "prefilter_mm3": 5.0, "bridge_radius_mm": 0.3,
                 "min_component_mm3_to_bridge": 5.0, "max_bridges_per_case": 50})
            self.assertEqual(info["bridges_added"], 0,
                             f"a 4.2 mm gap was bridged at a {gap} mm tolerance")

    def test_bridging_is_monotone_in_tolerance(self):
        ph = self.phantoms["phantom_mixed"]
        counts = []
        for gap in (0.6, 1.2, 1.8, 2.4):
            _, info = strat.apply_strategy(
                ph.prediction, ph.spacing, "gap_bridge",
                {"max_gap_mm": gap, "prefilter_mm3": 5.0, "bridge_radius_mm": 0.3,
                 "min_component_mm3_to_bridge": 5.0, "max_bridges_per_case": 50})
            counts.append(info["components_after"])
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_deterministic(self):
        ph = self.phantoms["phantom_mixed"]
        params = {"max_gap_mm": 2.4, "prefilter_mm3": 5.0, "bridge_radius_mm": 0.3,
                  "min_component_mm3_to_bridge": 5.0, "max_bridges_per_case": 50}
        a, _ = strat.apply_strategy(ph.prediction, ph.spacing, "gap_bridge", params)
        b, _ = strat.apply_strategy(ph.prediction, ph.spacing, "gap_bridge", params)
        np.testing.assert_array_equal(a, b)

    def test_strategies_do_not_mutate_the_input(self):
        ph = self.phantoms["phantom_mixed"]
        before = ph.prediction.copy()
        for kind, params in (("identity", {}),
                             ("absolute_volume_filter", {"min_volume_mm3": 5.0}),
                             ("relative_volume_filter", {"min_fraction_of_largest": 0.01}),
                             ("gap_bridge", {"max_gap_mm": 2.4, "prefilter_mm3": 5.0,
                                             "bridge_radius_mm": 0.3,
                                             "min_component_mm3_to_bridge": 5.0,
                                             "max_bridges_per_case": 50}),
                             ("morphological_closing", {"radius_mm": 0.6, "prefilter_mm3": 5.0})):
            strat.apply_strategy(ph.prediction, ph.spacing, kind, params)
            np.testing.assert_array_equal(ph.prediction, before,
                                          f"{kind} mutated the frozen input mask")

    def test_variant_expansion_matches_config(self):
        cfg = load_config(Path(__file__).parents[1] / "config" / "experiment_config.yaml")
        variants = strat.expand_variants(cfg["strategies"])
        self.assertEqual(len(variants), 1 + 7 + 4 + 4 + 2)
        for name in ("s0_original", "s1_absolute_volume", "s2_relative_volume",
                     "s3_gap_bridge", "s3c_closing"):
            primaries = [v for v in variants if v["strategy"] == name and v["primary"]]
            self.assertEqual(len(primaries), 1, f"{name} must have exactly one primary variant")


class TestSegMetrics(unittest.TestCase):
    def test_dice_known_value(self):
        a = np.zeros((10, 10, 10), dtype=bool); a[:5] = True
        b = np.zeros((10, 10, 10), dtype=bool); b[2:7] = True
        m = seg.overlap_metrics(a, b)
        self.assertAlmostEqual(m["dice"], 2 * 300 / (500 + 500), places=9)
        self.assertAlmostEqual(m["precision"], 300 / 500, places=9)
        self.assertAlmostEqual(m["recall"], 300 / 500, places=9)

    def test_dice_identical_is_one(self):
        a = np.zeros((8, 8, 8), dtype=bool); a[2:6, 2:6, 2:6] = True
        self.assertAlmostEqual(seg.overlap_metrics(a, a)["dice"], 1.0, places=12)

    def test_surface_distance_zero_for_identical(self):
        a = np.zeros((20, 20, 20), dtype=bool); a[5:15, 5:15, 5:15] = True
        m = seg.surface_distance_metrics(a, a, SPACING)
        self.assertAlmostEqual(m["hd95"], 0.0, places=9)
        self.assertAlmostEqual(m["assd"], 0.0, places=9)

    def test_hd95_scales_with_spacing(self):
        a = np.zeros((30, 30, 30), dtype=bool); a[10:20, 10:20, 10:20] = True
        b = a.copy(); b[10:20, 10:20, 10:21] = True     # grown by one voxel on one face
        m1 = seg.surface_distance_metrics(a, b, (1.0, 1.0, 1.0))
        m2 = seg.surface_distance_metrics(a, b, (2.0, 2.0, 2.0))
        self.assertAlmostEqual(m2["assd"], 2 * m1["assd"], places=6)

    def test_cldice_uses_repo_implementation(self):
        a = np.zeros((20, 20, 20), dtype=bool); a[10, 5:15, 10] = True
        cl, tp, ts = seg.cldice(a, a)
        self.assertAlmostEqual(cl, 1.0, places=9)
        self.assertAlmostEqual(tp, 1.0, places=9)
        self.assertAlmostEqual(ts, 1.0, places=9)

    def test_removing_speckle_cannot_lower_dice_when_speckle_is_false_positive(self):
        gt = np.zeros((30, 30, 30), dtype=bool); gt[10:20, 10:20, 10:20] = True
        pred = gt.copy(); pred[2, 2, 2] = True; pred[27, 27, 27] = True
        before = seg.overlap_metrics(pred, gt)["dice"]
        cleaned, _ = strat.filter_absolute_volume(pred, SPACING, 5.0)
        after = seg.overlap_metrics(cleaned, gt)["dice"]
        self.assertGreater(after, before)

    def test_skeletonisation_backend_self_test_passes(self):
        r = seg.verify_skeletonisation_backend()
        self.assertTrue(r["gate_passed"], f"skeletonisation backend is unusable: {r}")
        self.assertTrue(r["cylinder_components_preserved"])

    def test_skeleton_stats_on_a_vessel_like_cylinder(self):
        a = np.zeros((40, 40, 40), dtype=bool)
        yy, zz = np.mgrid[-6:7, -6:7]
        disk = (yy ** 2 + zz ** 2) <= 4
        for x in range(10, 30):
            a[x, 14:27, 14:27] = disk
        s = seg.skeleton_stats(a, SPACING)
        self.assertGreater(s["skeleton_voxels"], 0)
        self.assertEqual(s["skeleton_components"], 1)
        self.assertGreaterEqual(s["skeleton_endpoints"], 1)


class TestGeometry(unittest.TestCase):
    def _write_box(self, path, translate=(0.0, 0.0, 0.0), scale=1.0):
        import trimesh
        m = trimesh.creation.box(extents=(10.0 * scale, 10.0 * scale, 10.0 * scale))
        m.apply_translation(np.asarray(translate, dtype=float))
        m.export(str(path))
        return path

    def test_identical_meshes_have_zero_deviation(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write_box(Path(d) / "a.stl")
            out = geom.surface_deviation(p, p, n_points=4000)
            self.assertEqual(out["geometry_status"], "ok")
            self.assertAlmostEqual(out["symmetric_mean_surface_distance_mm"], 0.0, places=6)
            self.assertAlmostEqual(out["centroid_displacement_mm"], 0.0, places=6)
            self.assertAlmostEqual(out["mesh_volume_change_relative"], 0.0, places=6)

    def test_translation_is_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._write_box(Path(d) / "a.stl")
            b = self._write_box(Path(d) / "b.stl", translate=(3.0, 0.0, 0.0))
            out = geom.surface_deviation(a, b, n_points=4000)
            self.assertAlmostEqual(out["centroid_displacement_mm"], 3.0, places=4)
            self.assertGreater(out["hausdorff_mm"], 0.0)

    def test_volume_change_is_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._write_box(Path(d) / "a.stl")
            b = self._write_box(Path(d) / "b.stl", scale=2.0)
            out = geom.surface_deviation(a, b, n_points=4000)
            self.assertAlmostEqual(out["mesh_volume_change_relative"], 7.0, places=3)

    def test_missing_mesh_is_reported_not_fabricated(self):
        out = geom.surface_deviation("/nonexistent/a.stl", "/nonexistent/b.stl")
        self.assertTrue(str(out["geometry_status"]).startswith("missing_mesh"))


class TestStatistics(unittest.TestCase):
    def test_wilcoxon_direction_and_effect_sign(self):
        rng = np.random.default_rng(0)
        a = rng.normal(10, 1, 60)
        # heavy-tailed, clearly non-normal improvement so Wilcoxon is selected
        b = a - np.abs(rng.standard_cauchy(60)) - 0.5
        r = st.paired_continuous(a, b, "metric")
        self.assertEqual(r["test"], "wilcoxon_signed_rank")
        self.assertLess(r["p_value"], 1e-6)
        self.assertLess(r["median_difference"], 0)
        self.assertEqual(r["effect_size"], -1.0)      # every pair improved
        self.assertEqual(r["n_improved"], 60)

    def test_constant_difference_is_not_treated_as_normal(self):
        a = np.random.default_rng(0).normal(10, 1, 60)
        r = st.paired_continuous(a, a - 1.0, "metric")
        self.assertEqual(r["test"], "wilcoxon_signed_rank")

    def test_no_variation_returns_p_one(self):
        a = np.arange(20, dtype=float)
        r = st.paired_continuous(a, a.copy(), "metric")
        self.assertEqual(r["test"], "no_variation")
        self.assertEqual(r["p_value"], 1.0)
        self.assertEqual(r["effect_size"], 0.0)

    def test_paired_t_selected_for_normal_differences(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 400)
        b = a + rng.normal(0.5, 0.1, 400)
        r = st.paired_continuous(a, b, "metric")
        self.assertEqual(r["test"], "paired_t")
        self.assertGreater(r["effect_size"], 0)

    def test_small_sample_falls_back_to_wilcoxon(self):
        r = st.paired_continuous([1.0, 2, 3, 4, 5], [2.0, 3, 4, 5, 6], "metric")
        self.assertEqual(r["test"], "wilcoxon_signed_rank")

    def test_mcnemar_exact_matches_binomial(self):
        orig = [True] * 10 + [False] * 10
        corr = [True] * 10 + [True] * 8 + [False] * 2
        r = st.paired_binary(orig, corr, "integrity")
        self.assertEqual(r["discordant_gained"], 8)
        self.assertEqual(r["discordant_lost"], 0)
        from scipy import stats as _s
        self.assertAlmostEqual(r["p_value"], _s.binomtest(8, 8, 0.5).pvalue, places=12)

    def test_mcnemar_no_discordance(self):
        r = st.paired_binary([True, False] * 5, [True, False] * 5, "integrity")
        self.assertEqual(r["p_value"], 1.0)
        self.assertEqual(r["effect_size"], 0.0)

    def test_benjamini_hochberg_known_values(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        rej, adj = st.benjamini_hochberg(p, alpha=0.05)
        self.assertAlmostEqual(adj[0], 0.008, places=6)
        self.assertAlmostEqual(adj[1], 0.032, places=6)
        self.assertTrue(rej[0] and rej[1])
        self.assertFalse(rej[-1])
        self.assertTrue(np.all(np.diff(adj) >= -1e-12))     # monotone

    def test_bh_preserves_nan(self):
        rej, adj = st.benjamini_hochberg([0.01, np.nan, 0.5])
        self.assertTrue(np.isnan(adj[1]))
        self.assertFalse(rej[1])

    def test_bootstrap_ci_is_seeded_and_reproducible(self):
        d = np.random.default_rng(1).normal(0, 1, 100)
        self.assertEqual(st.bootstrap_ci(d, seed=5), st.bootstrap_ci(d, seed=5))

    def test_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(9)
        a = rng.normal(0, 1, 200)
        b = a + 0.4
        r = st.paired_continuous(a, b, "m")
        self.assertLessEqual(r["ci_low"], r["median_difference"])
        self.assertGreaterEqual(r["ci_high"], r["median_difference"])


class TestIntegrityGuards(unittest.TestCase):
    def test_refuses_to_write_into_authoritative_paths(self):
        from topocorr.io_utils import assert_not_protected, repo_root
        root = repo_root()
        for bad in ("outputs/final_test_250/x.csv", "outputs/phase_b_mesh_qc/y.csv",
                    "paper/main.tex", "analysis/cmig_robustness/z.csv"):
            with self.assertRaises(PermissionError, msg=f"{bad} was not protected"):
                assert_not_protected(root, root / bad)

    def test_allows_the_experiment_output_directory(self):
        from topocorr.io_utils import assert_not_protected, repo_root
        root = repo_root()
        assert_not_protected(root, root / "outputs" / "topology_correction" / "a.csv")

    def test_config_threshold_matches_authoritative_evaluation(self):
        cfg = load_config(Path(__file__).parents[1] / "config" / "experiment_config.yaml")
        self.assertEqual(float(cfg["segmentation"]["threshold"]), 0.5)
        self.assertEqual(int(cfg["provenance"]["expected_n_cases"]), 250)


if __name__ == "__main__":
    unittest.main(verbosity=2)
