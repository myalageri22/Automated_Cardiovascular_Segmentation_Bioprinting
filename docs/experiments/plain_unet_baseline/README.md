# Same-Split Conventional 3D U-Net Baseline

## Purpose

This experiment provides a conventional plain 3D U-Net baseline for the coronary CCTA segmentation stage and compares it with the frozen Attention U-Net used by the image-to-mesh pipeline. The analysis is a paired, same-split baseline comparison. It is not a pure architecture ablation because the two model lineages did not have identical training histories or initialization.

## Dataset and split

- Dataset: ImageCAS coronary CCTA.
- Fixed split: 700 training, 50 validation, and 250 held-out test cases.
- Authoritative source split SHA-256: `70e847d64dc020cc8233016052772bc9b4d119c56685f6333dd4a1495b72d2db`.
- Repository manifest: [`configs/baselines/imagecas_split_ids.json`](../../../configs/baselines/imagecas_split_ids.json), an ID-only serialization with unchanged membership.
- Final paired cohort: exactly 250 unique held-out cases, with no missing, extra, or failed cases.

## Models

### Plain 3D U-Net

The baseline is a MONAI conventional 3D U-Net without attention gates. It uses channels `(32, 64, 128, 256, 512)`, four stride-2 downsampling stages, three residual units per stage, batch normalization, and dropout 0.1.

- Frozen checkpoint: `outputs/baselines/plain_unet/bioprint_v6_plain_unet_baseline/checkpoints/best_dice05.pt`
- Selected epoch: 30
- Selection criterion: validation Dice at threshold 0.5
- SHA-256: `f48269a225c20b0f1eb173b92308bd215d34a7dcded05fca46e21dfb2a9afba0`
- Config: [`configs/baselines/plain_unet_baseline.yaml`](../../../configs/baselines/plain_unet_baseline.yaml)

### Attention U-Net reference

The reference is the frozen three-dimensional Attention U-Net selected by validation Dice.

- Frozen checkpoint: `outputs/train_runs/bioprint_v14_tversky_fn_local_rebuild_mps_lowmem/checkpoints/best_dice05.pt`
- Selected epoch: 79
- Selection criterion: validation Dice at threshold 0.5
- SHA-256: `792178759e9e579eb65f89a5a63f6538d7d0202da7e8eeeefda4825bfffc743b`
- Config: [`configs/baselines/attention_reference.yaml`](../../../configs/baselines/attention_reference.yaml)

Checkpoint binaries are not added by this experiment integration. Their hashes and expected local paths are retained for provenance.

## Fixed held-out protocol

- Threshold: 0.5 only.
- Test-set threshold tuning: none.
- Orientation: RAS.
- Voxel spacing target: `0.6 x 0.6 x 0.6`.
- CT intensity window: `[-200, 700]`, scaled to `[0, 1]`.
- Label binarization: `label > 0`.
- Sliding-window ROI: `96 x 192 x 192`.
- Sliding-window overlap: 0.625.
- Sliding-window batch size: 1.
- Plain mask regeneration device: CPU, matching the finalized plain evaluation.

A three-case plain U-Net pilot reproduced Dice, precision, recall, HD95, and voxel counts exactly before the complete inference-only mask regeneration proceeded. The complete 250-case regeneration also matched all frozen scalar results with maximum absolute difference `0.0`.

## Exact-250 cohort audit

The historical Attention U-Net table had 252 rows but only 250 unique test IDs. Cases `751` and `1000` each appeared twice. For both cases, scientific values and artifact paths were identical; only runtime differed. This is consistent with repeated evaluation output, not additional scientific cases. Historical files were not rewritten.

The authoritative comparison uses the separate exact-250 Attention source and the exact-250 plain source. See [`Results/plain_unet_baseline/cohort_audit.csv`](../../../Results/plain_unet_baseline/cohort_audit.csv) and [`attention_duplicate_audit.csv`](../../../Results/plain_unet_baseline/attention_duplicate_audit.csv).

## clDice compatibility

The legacy evaluator imported `skeletonize` and the removed `skeletonize_3d` symbol together. With scikit-image 0.26.0, that combined import failed and disabled both functions. The compatibility implementation uses the currently supported 3D `skeletonize(..., method="lee")` behavior while preserving the hard-skeleton formula:

```text
topology precision   = |S_pred intersect V_gt| / |S_pred|
topology sensitivity = |S_gt intersect V_pred| / |S_gt|
clDice               = 2 * Tprec * Tsens / (Tprec + Tsens)
```

Synthetic tests verify perfect overlap, defined empty-mask behavior, a decrease for a broken branch, binary output, and shape preservation. Recomputed Attention clDice was `0.8695228014`, differing from the historical approximate `0.86952` by `+0.0000028014`.

## Final metrics

| Metric | Attention U-Net | Plain 3D U-Net | Difference |
|---|---:|---:|---:|
| Dice@0.5 | 0.7878 | 0.7723 | +0.0155 |
| clDice@0.5 | 0.8695 | 0.8590 | +0.0105 |
| Precision@0.5 | 0.7636 | 0.7343 | +0.0294 |
| Recall@0.5 | 0.8205 | 0.8232 | -0.0027 |
| HD95@0.5 (mm) | 5.0120 | 6.1586 | -1.1466 |

Differences are Attention minus Plain. Lower HD95 is better.

## Paired statistics

All confidence intervals below use 10,000 paired case-level bootstrap resamples with seed 42. Wilcoxon signed-rank tests were corrected across the five prespecified metrics using Benjamini-Hochberg FDR.

| Metric | Mean paired difference | Bootstrap 95% CI | FDR q |
|---|---:|---:|---:|
| Dice | +0.015505 | +0.013156 to +0.018008 | 1.47e-27 |
| clDice | +0.010526 | +0.007119 to +0.014127 | 6.00e-08 |
| Precision | +0.029356 | +0.024990 to +0.033736 | 1.29e-26 |
| Recall | -0.002655 | -0.005765 to +0.000442 | 0.3319 |
| HD95 | -1.146557 mm | -1.843586 to -0.536552 | 2.80e-05 |

## Interpretation

The Attention U-Net had higher Dice, clDice, and precision and lower HD95. Recall was statistically similar. These results support its selection for the image-to-mesh pipeline, where false-positive structures, discontinuities, and boundary errors can propagate into mesh defects. They do not establish that attention gates alone caused the observed differences.

## Limitations

The segmentation comparison was limited to the Attention U-Net and a conventional 3D U-Net baseline; broader benchmarking against additional modern architectures was outside the scope of this study.

## Reproduction

Audit the committed exact-250 tables:

```bash
python scripts/evaluation/audit_exact_test_cohort.py \
  --split-json configs/baselines/imagecas_split_ids.json \
  --source plain=Results/plain_unet_baseline/per_case/plain_unet_per_case.csv \
  --source attention=Results/plain_unet_baseline/per_case/attention_unet_per_case.csv \
  --source paired=Results/plain_unet_baseline/per_case/attention_vs_plain_per_case.csv \
  --output /tmp/final_unet_cohort_audit.csv
```

Reproduce the paired comparison from the committed model tables:

```bash
python scripts/evaluation/compare_attention_vs_plain.py \
  --plain Results/plain_unet_baseline/per_case/plain_unet_per_case.csv \
  --attention Results/plain_unet_baseline/per_case/attention_unet_per_case.csv \
  --output-dir /tmp/final_unet_comparison \
  --expected-cases 250 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 42
```

Run the metric and statistics tests:

```bash
pytest tests/test_final_baseline_evaluation.py
```

`compute_cldice_3d.py` accepts aligned prediction and label patterns with a fixed threshold. Inputs must already be on the same evaluation grid; the script deliberately does not alter preprocessing or optimize a threshold.

## Output map

- `Results/plain_unet_baseline/plain_unet_summary.{csv,json}`: finalized plain metrics.
- `Results/plain_unet_baseline/attention_unet_summary.{csv,json}`: finalized Attention metrics.
- `Results/plain_unet_baseline/per_case/`: exact-250 model and paired tables.
- `Results/plain_unet_baseline/paired_model_statistics.csv`: bootstrap, Wilcoxon, and FDR results.
- `Results/plain_unet_baseline/attention_duplicate_audit.csv`: duplicate provenance for cases 751 and 1000.
- `Results/plain_unet_baseline/audit/`: environment and regeneration checks.
- `configs/baselines/`: split, checkpoint, protocol, and preprocessing provenance.
- `scripts/evaluation/`: cohort, clDice, and paired-comparison utilities.
- `docs/experiments/plain_unet_baseline/`: publication-ready report and manuscript text.
