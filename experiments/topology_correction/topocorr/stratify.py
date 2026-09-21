"""Stratified and covariate analyses.

Two stratifications are reported and are labelled distinctly:

  prespecified_quantile  cohort tertiles of the ORIGINAL connected-component
                         count. The RULE (tertiles) is prespecified; the cut
                         values are cohort quantiles and are reported.
  anatomical             components <= 2 (a plausible major-tree range, since the
                         left and right coronary systems may legitimately be
                         separate) versus > 2. Prespecified, not data-derived.

The covariate analysis answers the CMIG-style question of whether upstream
segmentation metrics identify which cases benefit most from correction. It is
correlational and is labelled exploratory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .stats_tests import paired_continuous, paired_binary


def assign_strata(
    components_original: Sequence[float],
    quantiles: Sequence[float] = (0.3333, 0.6667),
    low_fragmentation_max: int = 2,
) -> Dict[str, Any]:
    cc = np.asarray(components_original, dtype=float)
    cuts = [float(np.quantile(cc, q)) for q in quantiles]

    quantile_labels: List[str] = []
    for v in cc:
        if v <= cuts[0]:
            quantile_labels.append("q1_least_fragmented")
        elif v <= cuts[1]:
            quantile_labels.append("q2_middle")
        else:
            quantile_labels.append("q3_most_fragmented")

    anatomical_labels = [
        "low_fragmentation" if v <= float(low_fragmentation_max) else "high_fragmentation" for v in cc
    ]
    return {
        "quantile_labels": quantile_labels,
        "quantile_cuts": cuts,
        "quantile_rule": f"tertiles of original component count at q={list(quantiles)}",
        "anatomical_labels": anatomical_labels,
        "anatomical_rule": f"components <= {low_fragmentation_max} == low fragmentation",
    }


def stratified_comparison(
    df: pd.DataFrame,
    stratum_column: str,
    metric_original: str,
    metric_corrected: str,
    metric_name: str,
    strategy: str,
    binary: bool = False,
    seed: int = 20260826,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stratum, sub in df.groupby(stratum_column, sort=True):
        if binary:
            r = paired_binary(
                sub[metric_original].astype(bool), sub[metric_corrected].astype(bool),
                metric=metric_name, strategy=strategy, seed=seed,
            )
        else:
            r = paired_continuous(
                sub[metric_original], sub[metric_corrected],
                metric=metric_name, strategy=strategy, seed=seed,
            )
        r["stratum_column"] = stratum_column
        r["stratum"] = str(stratum)
        r["family"] = f"{strategy}|{stratum_column}|{metric_name}"
        rows.append(r)
    return rows


def covariate_association(
    df: pd.DataFrame,
    benefit_column: str,
    covariates: Sequence[str],
    strategy: str,
) -> List[Dict[str, Any]]:
    """Spearman association between baseline case difficulty and correction benefit.

    Exploratory. Reported with rho, P and n; no causal claim is made.
    """
    rows: List[Dict[str, Any]] = []
    y = pd.to_numeric(df[benefit_column], errors="coerce")
    for cov in covariates:
        if cov not in df.columns:
            rows.append({"strategy": strategy, "benefit_metric": benefit_column, "covariate": cov,
                         "n": 0, "spearman_rho": float("nan"), "p_value": float("nan"),
                         "analysis": "exploratory", "note": "covariate_absent"})
            continue
        x = pd.to_numeric(df[cov], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            rows.append({"strategy": strategy, "benefit_metric": benefit_column, "covariate": cov,
                         "n": int(ok.sum()), "spearman_rho": float("nan"), "p_value": float("nan"),
                         "analysis": "exploratory", "note": "insufficient_n"})
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append({
            "strategy": strategy, "benefit_metric": benefit_column, "covariate": cov,
            "n": int(ok.sum()), "spearman_rho": float(rho), "p_value": float(p),
            "analysis": "exploratory", "family": f"{strategy}|covariates|{benefit_column}",
        })
    return rows
