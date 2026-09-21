"""Paired statistical comparison of a correction strategy against the control.

Design rules enforced here:
  * every comparison is paired by case;
  * the continuous test is chosen by inspecting the distribution of the paired
    differences (Shapiro-Wilk), not by which test gives a smaller P value;
  * paired binary outcomes use exact McNemar;
  * effect size and a 95% confidence interval are always reported alongside P;
  * families of related endpoints get Benjamini-Hochberg FDR correction.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def _clean_pairs(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Unpaired inputs: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def hodges_lehmann(d: np.ndarray, max_n: int = 4000) -> float:
    """Hodges-Lehmann pseudo-median of paired differences (median of Walsh averages)."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return float("nan")
    if d.size > max_n:
        return float(np.median(d))
    i, j = np.triu_indices(d.size, k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.median,
    iterations: int = 10000,
    ci: float = 0.95,
    seed: int = 20260826,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic of paired differences."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, values.size, size=(int(iterations), values.size))
    boot = statistic(values[idx], axis=1)
    lo = (1.0 - ci) / 2.0 * 100.0
    return float(np.percentile(boot, lo)), float(np.percentile(boot, 100.0 - lo))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the Wilcoxon effect size."""
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    rpos = float(ranks[nz > 0].sum())
    rneg = float(ranks[nz < 0].sum())
    total = rpos + rneg
    return float((rpos - rneg) / total) if total else 0.0


def paired_continuous(
    original: Sequence[float],
    corrected: Sequence[float],
    metric: str,
    strategy: str = "",
    mode: str = "auto",
    shapiro_alpha: float = 0.05,
    iterations: int = 10000,
    ci: float = 0.95,
    seed: int = 20260826,
) -> Dict[str, Any]:
    """Paired comparison of one continuous endpoint. Effect direction is corrected - original."""
    a, b = _clean_pairs(original, corrected)
    n = int(a.size)
    row: Dict[str, Any] = {
        "strategy": strategy,
        "metric": metric,
        "n_pairs": n,
        "original_mean": float(a.mean()) if n else float("nan"),
        "original_median": float(np.median(a)) if n else float("nan"),
        "original_sd": float(a.std(ddof=1)) if n > 1 else float("nan"),
        "corrected_mean": float(b.mean()) if n else float("nan"),
        "corrected_median": float(np.median(b)) if n else float("nan"),
        "corrected_sd": float(b.std(ddof=1)) if n > 1 else float("nan"),
    }
    if n < 2:
        row.update({"test": "insufficient_n", "p_value": float("nan")})
        return row

    d = b - a
    row["mean_difference"] = float(d.mean())
    row["median_difference"] = float(np.median(d))
    row["hodges_lehmann_difference"] = hodges_lehmann(d)
    row["n_improved"] = int((d < 0).sum())
    row["n_worsened"] = int((d > 0).sum())
    row["n_unchanged"] = int((d == 0).sum())

    if np.allclose(d, 0):
        row.update({
            "test": "no_variation", "statistic": float("nan"), "p_value": 1.0,
            "effect_size_name": "rank_biserial", "effect_size": 0.0,
            "ci_low": 0.0, "ci_high": 0.0, "normality_p": float("nan"),
            "ci_statistic": "median_of_differences",
        })
        return row

    normal_p = float("nan")
    use_t = False
    if mode == "auto":
        # Below n=8 the Shapiro-Wilk statistic is not usable; default to the
        # distribution-free test rather than assuming normality.
        # A (near-)constant difference makes Shapiro-Wilk return p = 1.0, which
        # would route a degenerate comparison to a t-test. Treat it as non-normal.
        if d.size < 8 or float(np.std(d, ddof=1)) <= 1e-12 * max(1.0, float(np.abs(d).max())):
            use_t = False
        else:
            try:
                sample = d if d.size <= 5000 else d[:5000]
                normal_p = float(stats.shapiro(sample).pvalue)
                use_t = normal_p >= float(shapiro_alpha)
            except Exception:
                use_t = False
    elif mode == "t":
        use_t = True

    row["normality_p"] = normal_p

    if use_t:
        res = stats.ttest_rel(b, a)
        row["test"] = "paired_t"
        row["statistic"] = float(res.statistic)
        row["p_value"] = float(res.pvalue)
        sd = float(d.std(ddof=1))
        row["effect_size_name"] = "cohens_dz"
        row["effect_size"] = float(d.mean() / sd) if sd > 0 else float("nan")
        lo, hi = bootstrap_ci(d, statistic=np.mean, iterations=iterations, ci=ci, seed=seed)
        row["ci_statistic"] = "mean_of_differences"
    else:
        try:
            res = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided", mode="auto")
        except TypeError:  # newer scipy renamed the argument
            res = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided", method="auto")
        row["test"] = "wilcoxon_signed_rank"
        row["statistic"] = float(res.statistic)
        row["p_value"] = float(res.pvalue)
        row["effect_size_name"] = "rank_biserial"
        row["effect_size"] = rank_biserial(d)
        lo, hi = bootstrap_ci(d, statistic=np.median, iterations=iterations, ci=ci, seed=seed)
        row["ci_statistic"] = "median_of_differences"

    row["ci_low"], row["ci_high"] = lo, hi
    row["ci_level"] = float(ci)
    return row


def paired_binary(
    original: Sequence[bool],
    corrected: Sequence[bool],
    metric: str,
    strategy: str = "",
    iterations: int = 10000,
    ci: float = 0.95,
    seed: int = 20260826,
) -> Dict[str, Any]:
    """Exact McNemar test for a paired binary outcome such as mesh integrity."""
    a = np.asarray(original, dtype=bool)
    b = np.asarray(corrected, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Unpaired binary inputs")
    n = int(a.size)
    n01 = int(np.count_nonzero(~a & b))   # fail -> pass  (gained)
    n10 = int(np.count_nonzero(a & ~b))   # pass -> fail  (lost)

    row: Dict[str, Any] = {
        "strategy": strategy,
        "metric": metric,
        "n_pairs": n,
        "original_pass": int(a.sum()),
        "corrected_pass": int(b.sum()),
        "original_rate": float(a.mean()) if n else float("nan"),
        "corrected_rate": float(b.mean()) if n else float("nan"),
        "discordant_gained": n01,
        "discordant_lost": n10,
        "test": "mcnemar_exact",
        "effect_size_name": "paired_rate_difference",
        "effect_size": float(b.mean() - a.mean()) if n else float("nan"),
        "ci_statistic": "paired_rate_difference",
    }
    if n01 + n10 == 0:
        row["statistic"] = 0.0
        row["p_value"] = 1.0
        row["odds_ratio"] = float("nan")
        row["ci_low"] = row["ci_high"] = 0.0
        row["ci_level"] = float(ci)
        return row

    p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    row["statistic"] = float(min(n01, n10))
    row["p_value"] = p
    row["odds_ratio"] = float(n01 / n10) if n10 > 0 else float("inf")

    diff = b.astype(float) - a.astype(float)
    lo, hi = bootstrap_ci(diff, statistic=np.mean, iterations=iterations, ci=ci, seed=seed)
    row["ci_low"], row["ci_high"] = lo, hi
    row["ci_level"] = float(ci)
    return row


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """BH step-up FDR. Returns (rejected, adjusted p-values). NaNs pass through."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return rejected, adjusted
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    m = ranked.size
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    adjusted[finite] = out
    rejected[finite] = out <= float(alpha)
    return rejected, adjusted


def apply_fdr_by_family(rows: List[Dict[str, Any]], family_key: str = "family",
                        alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Attach BH-adjusted P values within each declared family of endpoints."""
    families: Dict[Any, List[int]] = {}
    for i, r in enumerate(rows):
        families.setdefault(r.get(family_key, "default"), []).append(i)
    for fam, idxs in families.items():
        rej, adj = benjamini_hochberg([rows[i].get("p_value", np.nan) for i in idxs], alpha=alpha)
        for k, i in enumerate(idxs):
            rows[i]["p_adjusted_bh"] = float(adj[k]) if np.isfinite(adj[k]) else float("nan")
            rows[i]["significant_fdr"] = bool(rej[k])
            rows[i]["fdr_family"] = fam
            rows[i]["fdr_family_size"] = len(idxs)
            rows[i]["fdr_alpha"] = float(alpha)
    return rows
