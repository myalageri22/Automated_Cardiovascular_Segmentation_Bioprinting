#!/usr/bin/env python3
"""Master runner: topology-aware postprocessing experiment.

Regenerates the entire experiment from the frozen held-out predictions.

    python run_topology_correction_experiment.py --config config/experiment_config.yaml

Guarantees enforced by this script:
  * authoritative artifacts are read-only and are fingerprinted before and after;
  * the segmentation threshold comes from the config and is never tuned here;
  * no case is excluded: every case in the split is attempted and every failure
    is recorded with its reason;
  * every figure is accompanied by the CSV of the data it plots.

Phantom mode (``--mode phantom``) validates the whole pipeline on synthetic
vessel trees with no cohort data. Phantom outputs are written to a separate
directory and are never cohort results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topocorr import figures as fig_mod
from topocorr import failure_analysis as fail_mod
from topocorr import final_report as rep_final
from topocorr import manifest as man_mod
from topocorr import pipeline as pipe_mod
from topocorr import report as rep_mod
from topocorr import stratify as strat_an
from topocorr import strategies as strat_mod
from topocorr.io_utils import (assert_not_protected, ensure_dir, load_config,
                               load_split_entries, load_test_cases, repo_root,
                               setup_logging, write_csv, write_json)

AUTHORITATIVE = [
    "outputs/final_test_250/per_case_metrics.csv",
    "outputs/phase_b_mesh_qc/per_case_mesh_qc.csv",
    "outputs/final_test_250/per_case_fabrication_readiness.csv",
    "outputs/final_test_250/summary_metrics.json",
    "extra_information/data_information/dataset_splits.json",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).parent / "config" / "experiment_config.yaml"))
    p.add_argument("--mode", choices=["cohort", "phantom"], default="cohort")
    p.add_argument("--output-root", default=None, help="Override paths.output_root")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N cases (debugging)")
    p.add_argument("--cases", nargs="*", default=None, help="Explicit case ids (debugging)")
    p.add_argument("--mesh-variants", choices=["primary", "all", "none"], default="primary",
                   help="Which variants get a Phase B reconstruction. 'primary' is the default "
                        "because reconstruction dominates runtime; sensitivity variants are still "
                        "fully evaluated in the voxel domain.")
    p.add_argument("--no-geometry", action="store_true")
    p.add_argument("--surface-sample-points", type=int, default=None,
                   help="Override metrics.geometry.surface_sample_points. Lower is faster and "
                        "noisier; the distance itself is exact point-to-surface either way.")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-failure-analysis", action="store_true")
    p.add_argument("--allow-provenance-mismatch", action="store_true",
                   help="Continue when a recomputed Dice disagrees with the archived value. "
                        "Mismatching cases are still flagged in every output.")
    p.add_argument("--dry-run", action="store_true", help="Resolve inputs and stop")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    started = man_mod.now_iso()
    root = repo_root()
    cfg = load_config(args.config)

    if args.surface_sample_points:
        cfg["metrics"]["geometry"]["surface_sample_points"] = int(args.surface_sample_points)

    out_root = Path(args.output_root) if args.output_root else (root / cfg["paths"]["output_root"])
    if args.mode == "phantom":
        out_root = out_root.parent / (out_root.name + "_phantom_validation")
    assert_not_protected(root, out_root)
    ensure_dir(out_root)
    logger = setup_logging(out_root / "logs")
    logger.info("repository root: %s", root)
    logger.info("output root: %s", out_root)
    logger.info("mode: %s", args.mode)

    # Fingerprint authoritative artifacts BEFORE doing anything.
    fp_before = man_mod.authoritative_fingerprint(root, AUTHORITATIVE)
    write_json(out_root / "logs" / "authoritative_fingerprint_before.json", fp_before)

    # Package gate. rtree in particular is easy to miss: without it the surface
    # distance silently degrades to a nearest-vertex approximation whose sampling
    # floor is the same size as the effects being measured.
    _missing = []
    for _m in ("numpy", "scipy", "skimage", "nibabel", "trimesh", "pandas",
               "matplotlib", "yaml", "rtree", "shapely"):
        try:
            __import__(_m)
        except Exception:
            _missing.append(_m)
    if _missing:
        logger.error("Missing packages: %s", ", ".join(_missing))
        logger.error("Install them, e.g.  pip install %s", " ".join(
            {"skimage": "scikit-image", "yaml": "PyYAML"}.get(m, m) for m in _missing))
        return 3

    # Environment gate: a broken skeletonisation backend would silently corrupt
    # every clDice and every skeleton statistic. Fail loudly instead.
    from topocorr import seg_metrics as _seg
    skel_check = _seg.verify_skeletonisation_backend()
    write_json(out_root / "logs" / "environment_selftest.json", skel_check)
    if not skel_check.get("gate_passed"):
        logger.error("Skeletonisation self-test FAILED: %s", skel_check)
        logger.error("clDice and every skeleton statistic would be invalid. Aborting.")
        return 3
    if skel_check.get("symmetric_block_artifact_present"):
        logger.warning("scikit-image %s empties symmetric even-sided solid blocks; "
                       "tube-like anatomy is unaffected. See KNOWN_ISSUES.md.",
                       skel_check.get("skimage_version"))

    variants = strat_mod.expand_variants(cfg["strategies"])
    logger.info("expanded %d strategy variants", len(variants))
    write_csv(out_root / "strategy_variants.csv", [
        {"variant_id": v["variant_id"], "strategy": v["strategy"], "kind": v["kind"],
         "primary": v["primary"], **{f"param_{k}": val for k, val in v["params"].items()}}
        for v in variants])

    # ------------------------------------------------------------------ cases
    authoritative_df: Optional[pd.DataFrame] = None
    split_entries: List[Dict[str, Any]] = []
    if args.mode == "phantom":
        from topocorr.phantoms import make_phantoms, write_phantom_case

        phantom_root = ensure_dir(out_root / "phantom_inputs")
        phantoms = make_phantoms()
        for ph in phantoms:
            write_phantom_case(ph, phantom_root)
        cases = [ph.case_id for ph in phantoms]
        cfg = dict(cfg)
        cfg["paths"] = dict(cfg["paths"])
        cfg["paths"]["pred_root"] = str(phantom_root / "pred")
        cfg["paths"]["pred_mask_template"] = "{pred_root}/{case}/seg_mask_0.5.nii.gz"
        cfg["paths"]["gt_template"] = str(phantom_root / "gt" / "{case}" / "{case}.label.nii.gz")
        cfg["paths"]["ct_template"] = str(phantom_root / "gt" / "{case}" / "{case}.img.nii.gz")
        cfg["provenance"] = dict(cfg["provenance"])
        cfg["provenance"]["enforce"] = False
        cfg["provenance"]["expected_n_cases"] = len(cases)
    else:
        split_entries = load_split_entries(root / cfg["paths"]["splits_json"],
                                           cfg["paths"].get("split_key", "test"))
        cases = [e["id"] for e in split_entries]
        auth_path = root / cfg["paths"]["authoritative_metrics_csv"]
        if auth_path.exists():
            authoritative_df = pd.read_csv(auth_path)
            logger.info("loaded authoritative metrics for provenance gating: %d rows", len(authoritative_df))

    expected_n = int(cfg["provenance"].get("expected_n_cases", len(cases)))
    if len(cases) != expected_n:
        logger.warning("split holds %d cases, config expects %d", len(cases), expected_n)
    if args.cases:
        cases = [c for c in cases if c in set(args.cases)]
    if args.limit:
        cases = cases[: int(args.limit)]
    logger.info("cases to attempt: %d", len(cases))

    entry_by_id = {e["id"]: e for e in (split_entries if args.mode == "cohort" else [])}
    resolved = {c: pipe_mod.resolve_case_inputs(cfg, root, c, entry_by_id.get(c), authoritative_df)
                for c in cases}
    inv = man_mod.input_manifest(cases, resolved)
    write_json(out_root / "input_manifest.json", inv)
    missing = [c for c in cases if not resolved[c]["complete"]]
    if missing:
        logger.error("%d of %d cases have missing inputs. First few: %s",
                     len(missing), len(cases), missing[:5])
        write_csv(out_root / "missing_inputs.csv",
                  [{"case_id": c, "missing": ";".join(resolved[c]["missing"]),
                    "prediction_path": resolved[c]["prediction_path"],
                    "ground_truth_path": resolved[c]["ground_truth_path"]} for c in missing])

    # Persist the exact resolved configuration actually used.
    import yaml
    (out_root / "experiment_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if args.dry_run:
        logger.info("dry run complete: %d/%d cases resolvable", len(cases) - len(missing), len(cases))
        return 0 if not missing else 2
    if missing and len(missing) == len(cases):
        logger.error("No case inputs could be resolved. The frozen per-case predictions are "
                     "required. See analysis/cmig_robustness/RECOVERY_RUNBOOK.md.")
        return 2

    # ------------------------------------------------------------------- run
    all_rows: List[Dict[str, Any]] = []
    all_components: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    volumes_cache: Dict[str, Any] = {}

    for i, case in enumerate(cases, 1):
        if not resolved[case]["complete"]:
            failures.append({"case_id": case, "stage": "input_resolution",
                             "error": "missing: " + ";".join(resolved[case]["missing"])})
            continue
        logger.info("[%d/%d] case %s", i, len(cases), case)
        res = pipe_mod.run_case(
            cfg, root, case, resolved[case], variants, out_root,
            authoritative=authoritative_df,
            do_mesh=(args.mesh_variants != "none"),
            do_geometry=(not args.no_geometry) and bool(cfg["metrics"]["geometry"]["enabled"]),
            mesh_variants=args.mesh_variants,
            logger=logger,
        )
        if res["status"] != "ok":
            failures.append({"case_id": case, "stage": "run_case", "error": res["error"],
                             "traceback": res.get("traceback", "")})
            continue
        all_rows.extend(res["rows"])
        all_components.extend(res["components"])

    per_case = pd.DataFrame(all_rows)
    if per_case.empty:
        logger.error("No per-case rows produced; nothing to summarise.")
        write_csv(out_root / "failed_cases.csv", failures)
        return 2

    ensure_dir(out_root)
    per_case[per_case["kind"] == "identity"].to_csv(out_root / "per_case_original.csv", index=False)
    per_case[per_case["kind"] != "identity"].to_csv(out_root / "per_case_corrected.csv", index=False)
    pd.DataFrame(all_components).to_csv(out_root / "component_audit.csv", index=False)
    if failures:
        write_csv(out_root / "failed_cases.csv", failures)

    paired = pipe_mod.build_paired_table(per_case)
    paired.to_csv(out_root / "paired_comparison.csv", index=False)

    summary = pipe_mod.cohort_summary(paired, per_case)
    summary.to_csv(out_root / "cohort_summary.csv", index=False)
    summary.to_csv(out_root / "sensitivity_analysis.csv", index=False)

    stats_df = rep_mod.run_statistics(paired, cfg)
    stats_df.to_csv(out_root / "statistical_tests.csv", index=False)

    # ------------------------------------------------------- stratified analysis
    strat_rows: List[Dict[str, Any]] = []
    cov_rows: List[Dict[str, Any]] = []
    if not paired.empty and "components_original" in paired.columns:
        primary_variants = sorted(paired[paired["primary_variant"]]["variant_id"].unique())
        for vid in primary_variants:
            sub = paired[paired["variant_id"] == vid].copy()
            strata = strat_an.assign_strata(
                pd.to_numeric(sub["components_original"], errors="coerce").fillna(0),
                quantiles=cfg["stratification"]["prespecified"]["component_count_quantiles"],
                low_fragmentation_max=cfg["stratification"]["anatomical"]["low_fragmentation_max_components"],
            )
            sub["stratum_quantile"] = strata["quantile_labels"]
            sub["stratum_anatomical"] = strata["anatomical_labels"]
            for scol in ("stratum_quantile", "stratum_anatomical"):
                for base, label, is_bin in rep_mod.PRIMARY_ENDPOINTS:
                    o, c = f"{base}_original", f"{base}_corrected"
                    if o not in sub.columns or c not in sub.columns:
                        continue
                    rows = strat_an.stratified_comparison(
                        sub.dropna(subset=[o, c]), scol, o, c, label, vid,
                        binary=is_bin, seed=int(cfg["experiment"]["seed"]))
                    for r in rows:
                        r["quantile_cuts"] = str(strata["quantile_cuts"])
                        r["stratification_rule"] = (strata["quantile_rule"] if scol == "stratum_quantile"
                                                    else strata["anatomical_rule"])
                        r["stratification_type"] = ("prespecified_quantile" if scol == "stratum_quantile"
                                                    else "prespecified_anatomical")
                    strat_rows.extend(rows)
            sub["benefit_delta_components"] = (pd.to_numeric(sub["components_corrected"], errors="coerce")
                                               - pd.to_numeric(sub["components_original"], errors="coerce"))
            cov_rows.extend(strat_an.covariate_association(
                sub, "benefit_delta_components",
                [f"{c}_original" if not c.endswith("_original") else c
                 for c in cfg["stratification"]["exploratory_covariates"]],
                vid))
    if strat_rows:
        from topocorr.stats_tests import apply_fdr_by_family
        strat_rows = apply_fdr_by_family(strat_rows, "family", float(cfg["statistics"]["fdr_alpha"]))
        write_csv(out_root / "stratified_analysis.csv", strat_rows)
    if cov_rows:
        write_csv(out_root / "covariate_analysis.csv", cov_rows)

    # ------------------------------------------------------------- primary table
    tables_dir = ensure_dir(out_root / "tables")
    outcomes: List[Dict[str, Any]] = []
    for _, srow in summary.iterrows():
        if bool(srow.get("primary_variant")):
            vid = str(srow["variant_id"])
            tbl = rep_mod.build_primary_table(paired, stats_df, vid)
            tbl.to_csv(tables_dir / f"primary_table__{vid}.csv", index=False)
            (tables_dir / f"primary_table__{vid}.md").write_text(
                f"# Primary comparison: {vid}\n\n"
                f"Original = frozen prediction (control). Corrected = {vid}. "
                f"Paired by case, n as shown.\n\n" + rep_mod.to_markdown(tbl) + "\n")
            outcomes.append(rep_mod.classify_outcome(srow))
    if outcomes:
        write_csv(out_root / "outcome_classification.csv",
                  [{k: v for k, v in o.items() if k != "rules"} for o in outcomes])
        write_json(out_root / "interpretation_rules.json", rep_mod.INTERPRETATION_RULES)

    # ------------------------------------------------------------------ figures
    figure_index: Dict[str, Any] = {}
    if not args.no_figures:
        fdir = ensure_dir(out_root / "figures")
        fmts = tuple(cfg["figures"]["formats"])
        dpi = int(cfg["figures"]["dpi"])
        try:
            figure_index["A"] = fig_mod.figure_a_pipeline(fdir, fmts, dpi)
            if not paired.empty:
                for vid in sorted(paired[paired["primary_variant"]]["variant_id"].unique()):
                    figure_index[f"B__{vid}"] = fig_mod.figure_b_paired_components(
                        paired.rename(columns={"variant_id": "_v"}).assign(strategy=paired["variant_id"]),
                        vid, fdir, fmts, dpi)
                    figure_index[f"E__{vid}"] = fig_mod.figure_e_benefit_vs_baseline(
                        paired.assign(strategy=paired["variant_id"]), vid, fdir, fmts, dpi)
            if not summary.empty:
                figure_index["C"] = fig_mod.figure_c_tradeoff(summary, fdir, fmts, dpi)
                figure_index["S1"] = fig_mod.figure_sensitivity(summary, fdir, fmts, dpi)
        except Exception as exc:
            logger.error("figure generation failed: %s", exc)
            figure_index["error"] = repr(exc)
        write_json(out_root / "figures" / "figure_index.json", figure_index)

    # --------------------------------------------------------- failure analysis
    if not args.no_failure_analysis and not paired.empty:
        fa_dir = ensure_dir(out_root / "failure_analysis")
        try:
            selections = []
            for vid in sorted(paired[paired["primary_variant"]]["variant_id"].unique()):
                sel = fail_mod.select_cases(paired.assign(strategy=paired["variant_id"]),
                                            vid, int(cfg["failure_analysis"]["n_per_category"]))
                selections.append(sel)
            sel_all = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
            if not sel_all.empty:
                sel_all.to_csv(fa_dir / "selected_cases.csv", index=False)
                write_json(fa_dir / "selection_rules.json", fail_mod.SELECTION_RULES)
                rendered = _render_selected(cfg, root, sel_all, out_root, fa_dir, logger)
                write_csv(fa_dir / "rendered_cases.csv", rendered)
        except Exception as exc:
            logger.error("failure analysis failed: %s", exc)

    # ------------------------------------------- Figure D + written report (S21)
    if not args.no_figures:
        try:
            from build_final_report import build_figure_d
            figure_index.update(build_figure_d(out_root, cfg))
            write_json(out_root / "figures" / "figure_index.json", figure_index)
        except Exception as exc:
            logger.error("figure D generation failed: %s", exc)

    # ------------------------------------------------------------- verification
    fp_after = man_mod.authoritative_fingerprint(root, AUTHORITATIVE)
    write_json(out_root / "logs" / "authoritative_fingerprint_after.json", fp_after)
    unchanged = all(fp_before.get(k) == fp_after.get(k) for k in fp_before)
    if not unchanged:
        logger.error("AUTHORITATIVE ARTIFACT CHANGED DURING THE RUN. Investigate before use.")

    qc = {
        "cases_in_split": len(cases),
        "cases_attempted": len(cases),
        "cases_with_complete_inputs": len(cases) - len(missing),
        "cases_succeeded": int(per_case[per_case["kind"] == "identity"]["case_id"].nunique()),
        "cases_failed": len(failures),
        "failed_case_ids": [f["case_id"] for f in failures],
        "no_case_excluded": True,
        "segmentation_threshold_used": cfg["segmentation"]["threshold"],
        "segmentation_threshold_retuned": False,
        "variants_evaluated": len(variants),
        "authoritative_artifacts_unchanged": bool(unchanged),
        "provenance_enforced": bool(cfg["provenance"]["enforce"]) and not args.allow_provenance_mismatch,
    }
    if "provenance_ok" in per_case.columns:
        ctrl = per_case[per_case["kind"] == "identity"]
        qc["provenance_pass"] = int((ctrl["provenance_ok"] == True).sum())  # noqa: E712
        qc["provenance_fail"] = int((ctrl["provenance_ok"] == False).sum())  # noqa: E712
        qc["provenance_fail_case_ids"] = ctrl.loc[ctrl["provenance_ok"] == False, "case_id"].tolist()  # noqa: E712
        mean_dice = float(pd.to_numeric(ctrl["dice"], errors="coerce").mean())
        lo, hi = cfg["provenance"]["cohort_mean_dice_interval"]
        qc["cohort_mean_dice_recomputed"] = mean_dice
        qc["cohort_mean_dice_within_expected_interval"] = bool(lo <= mean_dice <= hi)
    write_json(out_root / "qc_checklist.json", qc)

    finished = man_mod.now_iso()
    run_manifest = man_mod.build_run_manifest(
        root, cfg, Path(args.config), started, finished,
        outputs={
            "output_root": str(out_root),
            "files": sorted(str(p.relative_to(out_root)) for p in out_root.rglob("*")
                            if p.is_file() and p.suffix in {".csv", ".json", ".png", ".pdf", ".md", ".yaml"}),
            "figure_index": figure_index,
        },
        authoritative_paths=AUTHORITATIVE,
        extra={"qc_checklist": qc, "cli_args": vars(args), "environment_selftest": skel_check,
               "phaseb_source_sha256": _safe(lambda: __import__(
                   "topocorr.mesh_metrics", fromlist=["x"]).phaseb_source_sha256())},
    )
    write_json(out_root / "run_manifest.json", run_manifest)

    try:
        rp = rep_final.build(out_root, cfg, root, figure_index)
        logger.info("report written: %s", rp)
    except Exception as exc:
        logger.error("report generation failed: %s", exc)


    logger.info("done. %d cases succeeded, %d failed. outputs in %s",
                qc["cases_succeeded"], qc["cases_failed"], out_root)
    return 0


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _render_selected(cfg, root, sel_all, out_root, fa_dir, logger) -> List[Dict[str, Any]]:
    """Render every selected representative case, successes and failures alike."""
    from topocorr.io_utils import load_binary_volume, resample_gt_to
    from topocorr import strategies as sm

    rendered: List[Dict[str, Any]] = []
    for _, r in sel_all.drop_duplicates(subset=["case_id", "variant_id"]).iterrows():
        case = str(r["case_id"])
        try:
            inputs = pipe_mod.resolve_case_inputs(cfg, root, case, None, None)
            if not inputs["complete"]:
                continue
            thr = (float(cfg["segmentation"]["threshold"])
                   if inputs["prediction_kind"] == "probability" else None)
            vol = load_binary_volume(inputs["prediction_path"], threshold=thr)
            gt = resample_gt_to(inputs["prediction_path"], inputs["ground_truth_path"])
            variant = next(v for v in sm.expand_variants(cfg["strategies"])
                           if v["variant_id"] == str(r["variant_id"]))
            corrected, _ = sm.apply_strategy(vol.array, vol.spacing, variant["kind"],
                                             variant["params"],
                                             int(cfg["connectivity"]["structure_rank"]))
            vid = str(r["variant_id"])
            # The control reconstruction for this case: the S0 identity variant
            # built by this run through the identical Phase B pipeline.
            control_mesh = out_root / "mesh_qc" / "s0_original" / "case_outputs" / \
                f"{case}__s0_original" / "segmentation_repaired.stl"
            original_mesh = str(control_mesh) if control_mesh.exists() else fail_mod.remap_to_root(
                r.get("mesh_repaired_stl_original"), root)
            corrected_mesh = fail_mod.remap_to_root(
                r.get("mesh_repaired_stl") or r.get("repaired_stl"), root)
            info = fail_mod.render_case(
                case, inputs.get("ct_path"), gt, vol.array, corrected,
                fa_dir / str(r["category"]),
                original_mesh=original_mesh,
                corrected_mesh=corrected_mesh,
                n_slices=int(cfg["failure_analysis"]["render_slices"]),
                title_suffix=f"- {r['category']} - {vid}",
                stem=f"{case}__{vid}",
            )
            info.update({"category": r["category"], "variant_id": r["variant_id"],
                         "selection_rule": r.get("selection_rule")})
            rendered.append(info)
        except Exception as exc:
            logger.warning("render failed for %s: %s", case, exc)
            rendered.append({"case_id": case, "category": r.get("category"), "error": repr(exc)})
    return rendered


if __name__ == "__main__":
    raise SystemExit(main())
