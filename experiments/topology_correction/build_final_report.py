#!/usr/bin/env python3
"""Figure D + the written report (Section 21), rebuilt from existing results.

Reads only the CSV/JSON artifacts of a completed run, so it is cheap to re-run
and cannot alter any measurement. The full experiment runner calls the same two
functions at the end of a run; this entry point exists so the deliverables can
be regenerated without recomputing the cohort.

    python build_final_report.py --config config/experiment_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topocorr import figures as fig_mod
from topocorr import final_report as rep_final
from topocorr import manifest as man_mod
from topocorr.io_utils import ensure_dir, load_config, repo_root, write_json


def build_figure_d(out_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble Figure D for every primary variant that has rendered panels."""
    fdir = ensure_dir(out_root / "figures")
    fa = out_root / "failure_analysis" / "rendered_cases.csv"
    paired_path = out_root / "paired_comparison.csv"
    index: Dict[str, Any] = {}
    if not fa.exists() or not paired_path.exists():
        return {"error": "rendered_cases.csv or paired_comparison.csv missing"}
    rendered = pd.read_csv(fa)
    paired = pd.read_csv(paired_path, low_memory=False)
    fmts = tuple(cfg["figures"]["formats"])
    dpi = int(cfg["figures"]["dpi"])
    n_per = int(cfg.get("figures", {}).get("figure_d_cases_per_category", 1))
    for vid in sorted(rendered["variant_id"].dropna().unique()):
        index[f"D__{vid}"] = fig_mod.figure_d_representative_cases(
            rendered, paired, str(vid), fdir, fmts, dpi, n_per_category=n_per)
    return index


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config" / "experiment_config.yaml"))
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--no-figure-d", action="store_true")
    args = ap.parse_args(argv)

    root = repo_root()
    cfg = load_config(args.config)
    out_root = Path(args.output_root) if args.output_root else (root / cfg["paths"]["output_root"])
    if not out_root.exists():
        print(f"no results at {out_root}", file=sys.stderr)
        return 2

    idx_path = out_root / "figures" / "figure_index.json"
    figure_index: Dict[str, Any] = {}
    if idx_path.exists():
        figure_index = json.loads(idx_path.read_text())

    if not args.no_figure_d:
        figure_index.update(build_figure_d(out_root, cfg))
        write_json(idx_path, figure_index)

    report_path = rep_final.build(out_root, cfg, root, figure_index)

    write_json(out_root / "report_manifest.json", {
        "generated": man_mod.now_iso(),
        "report": str(report_path),
        "figure_index": str(idx_path),
        "figure_d_keys": [k for k in figure_index if k.startswith("D__")],
        "source_artifacts": sorted(
            str(p.relative_to(out_root)) for p in out_root.glob("*.csv")),
        "note": "Derived from an existing run; no measurement was recomputed.",
    })
    print(f"report:   {report_path}")
    print(f"figure D: {[k for k in figure_index if k.startswith('D__')]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
