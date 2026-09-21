#!/usr/bin/env python3
"""Create an exact paired Attention-minus-plain segmentation comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


METRICS = {
    "dice@0.5": "Dice@0.5",
    "cldice@0.5": "clDice@0.5",
    "precision@0.5": "Precision@0.5",
    "recall@0.5": "Recall@0.5",
    "hd95@0.5": "HD95@0.5 (mm)",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, str]], expected_cases: int, name: str) -> dict[str, dict[str, str]]:
    ids = [row.get("case_id", "") for row in rows]
    if len(ids) != expected_cases or len(set(ids)) != expected_cases or any(not case_id for case_id in ids):
        raise ValueError(f"{name} must contain exactly {expected_cases} unique nonempty case IDs")
    for row in rows:
        for key in METRICS:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"{name} case {row['case_id']} has non-finite {key}")
            if key != "hd95@0.5" and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} case {row['case_id']} has out-of-range {key}={value}")
            if key == "hd95@0.5" and value < 0.0:
                raise ValueError(f"{name} case {row['case_id']} has negative HD95")
    return {row["case_id"]: row for row in rows}


def summarize(values: np.ndarray) -> dict[str, Any]:
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": std,
        "median": float(np.median(values)),
        "ci95_low": float(values.mean() - margin),
        "ci95_high": float(values.mean() + margin),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def fdr_bh(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    for index in range(len(p_values) - 1, -1, -1):
        original = int(order[index])
        running = min(running, p_values[original] * len(p_values) / (index + 1))
        adjusted[original] = running
    return adjusted.tolist()


def compare(
    plain_rows: list[dict[str, str]],
    attention_rows: list[dict[str, str]],
    expected_cases: int,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plain = validate_rows(plain_rows, expected_cases, "plain")
    attention = validate_rows(attention_rows, expected_cases, "attention")
    if set(plain) != set(attention):
        raise ValueError(f"Paired cohort mismatch: plain-only={sorted(set(plain)-set(attention))}, attention-only={sorted(set(attention)-set(plain))}")
    case_ids = [row["case_id"] for row in plain_rows]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, expected_cases, size=(resamples, expected_cases))

    paired_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        row: dict[str, Any] = {"case_id": case_id}
        for key in METRICS:
            plain_value = float(plain[case_id][key])
            attention_value = float(attention[case_id][key])
            row[f"plain_{key}"] = plain_value
            row[f"attention_{key}"] = attention_value
            row[f"difference_{key}"] = attention_value - plain_value
        paired_rows.append(row)

    summary_rows = []
    statistics_rows = []
    p_values = []
    for key, label in METRICS.items():
        plain_values = np.asarray([float(plain[case_id][key]) for case_id in case_ids])
        attention_values = np.asarray([float(attention[case_id][key]) for case_id in case_ids])
        differences = attention_values - plain_values
        boot_means = differences[indices].mean(axis=1)
        p_value = float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
        p_values.append(p_value)
        plain_summary = summarize(plain_values)
        attention_summary = summarize(attention_values)
        summary_rows.append(
            {
                "metric": label,
                "attention_mean": attention_summary["mean"],
                "attention_median": attention_summary["median"],
                "plain_mean": plain_summary["mean"],
                "plain_median": plain_summary["median"],
                "absolute_difference_attention_minus_plain": float(differences.mean()),
                "lower_is_better": key == "hd95@0.5",
            }
        )
        statistics_rows.append(
            {
                "metric": label,
                "difference_definition": "Attention - Plain",
                "n": expected_cases,
                "mean_paired_difference": float(differences.mean()),
                "median_paired_difference": float(np.median(differences)),
                "bootstrap_ci95_low": float(np.percentile(boot_means, 2.5)),
                "bootstrap_ci95_high": float(np.percentile(boot_means, 97.5)),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "wilcoxon_p": p_value,
            }
        )
    for row, q_value in zip(statistics_rows, fdr_bh(p_values)):
        row["fdr_bh_q"] = q_value
    return paired_rows, summary_rows, statistics_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exact paired plain and Attention U-Net result tables.")
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=250)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    paired, summary, statistics = compare(
        read_rows(args.plain),
        read_rows(args.attention),
        args.expected_cases,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    write_rows(args.output_dir / "attention_vs_plain_exact250_per_case.csv", paired)
    write_rows(args.output_dir / "attention_vs_plain_summary.csv", summary)
    write_rows(args.output_dir / "paired_model_statistics.csv", statistics)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "expected_cases": args.expected_cases,
                "threshold": 0.5,
                "threshold_tuning": False,
                "difference_definition": "Attention - Plain",
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_seed": args.bootstrap_seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
