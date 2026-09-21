#!/usr/bin/env python3
"""Re-render the representative-case panels from an existing run.

Why this exists: the first cohort run wrote every panel as ``<case>_slices.png``
inside the category directory, so a case selected for more than one strategy had
its panel overwritten by whichever strategy rendered last, and the control mesh
was looked up under a column name that the paired table does not contain, so the
"Original mesh" half of every mesh panel came out unavailable. Both faults are in
the VISUALISATION only -- no measurement, statistic, table or figure that carries
a number was affected. This script re-renders the panels with variant-qualified
filenames and the correct control mesh, and rewrites
``failure_analysis/rendered_cases.csv``. Selection is untouched: the same
``selected_cases.csv`` rows are rendered.

    python rerender_failure_panels.py --config config/experiment_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_topology_correction_experiment import _render_selected
from topocorr.io_utils import ensure_dir, load_config, repo_root, setup_logging, write_csv


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config" / "experiment_config.yaml"))
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip rows whose variant-qualified panels are already on disk. "
                         "Lets a long re-render be resumed in chunks.")
    ap.add_argument("--limit", type=int, default=None, help="Render at most N rows this call.")
    args = ap.parse_args(argv)

    root = repo_root()
    cfg = load_config(args.config)
    out_root = Path(args.output_root) if args.output_root else (root / cfg["paths"]["output_root"])
    fa_dir = ensure_dir(out_root / "failure_analysis")
    sel_path = fa_dir / "selected_cases.csv"
    if not sel_path.exists():
        print(f"missing {sel_path}", file=sys.stderr)
        return 2

    logger = setup_logging(out_root / "logs")
    sel_all = pd.read_csv(sel_path, low_memory=False)
    todo = sel_all
    if args.skip_existing:
        def _done(r):
            stem = f"{r['case_id']}__{r['variant_id']}"
            d = fa_dir / str(r["category"])
            return (d / f"{stem}_slices.png").exists() and (d / f"{stem}_meshes.png").exists()
        todo = sel_all[~sel_all.apply(_done, axis=1)]
    if args.limit:
        todo = todo.head(int(args.limit))
    rendered = _render_selected(cfg, root, todo, out_root, fa_dir, logger) if len(todo) else []

    # Rebuild the index from what is on disk so a chunked re-render still yields
    # one complete, self-consistent rendered_cases.csv.
    rows = []
    for _, r in sel_all.drop_duplicates(subset=["case_id", "variant_id"]).iterrows():
        stem = f"{r['case_id']}__{r['variant_id']}"
        d = fa_dir / str(r["category"])
        sp, mp = d / f"{stem}_slices.png", d / f"{stem}_meshes.png"
        rows.append({"case_id": r["case_id"], "slices_png": str(sp) if sp.exists() else None,
                     "meshes_png": str(mp) if mp.exists() else None,
                     "category": r["category"], "variant_id": r["variant_id"],
                     "selection_rule": r.get("selection_rule")})
    write_csv(fa_dir / "rendered_cases.csv", rows)
    rendered = rows
    ok = sum(1 for r in rendered if r.get("meshes_png"))
    print(f"rendered {len(rendered)} panels ({ok} with a mesh pair) -> {fa_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
