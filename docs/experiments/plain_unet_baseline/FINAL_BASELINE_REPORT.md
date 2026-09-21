# Final Same-Split Conventional 3D U-Net Baseline Comparison

## Methods

A conventional plain 3D U-Net baseline was evaluated using the same fixed 700/50/250 ImageCAS split as the Attention U-Net. For each model, the frozen validation-selected checkpoint was applied to the same 250 held-out test cases using the prespecified preprocessing and sliding-window protocol, and binary segmentation metrics were calculated at a fixed threshold of 0.5 without test-set threshold tuning. The comparison is same-split and paired by case ID, but is not presented as a pure architecture ablation because training histories and initialization lineages differed.

## Results

Across 250 paired held-out cases with no missing or extra cases, the conventional 3D U-Net achieved Dice 0.7723, clDice 0.8590, precision 0.7343, recall 0.8232, and HD95 6.1586 mm. The Attention U-Net achieved Dice 0.7878, clDice 0.8695, precision 0.7636, recall 0.8205, and HD95 5.0120 mm. Attention-minus-plain mean paired differences were Dice +0.0155 (bootstrap 95% CI, +0.0132 to +0.0180; FDR q=1.47e-27), clDice +0.0105 (+0.0071 to +0.0141; q=6.00e-08), precision +0.0294 (+0.0250 to +0.0337; q=1.29e-26), recall -0.0027 (-0.0058 to +0.0004; q=0.3319), and HD95 -1.1466 mm (-1.8436 to -0.5366; q=2.80e-05).

## Discussion

The same-split comparison showed higher held-out Dice, clDice, and precision and lower HD95 for the Attention U-Net than for the conventional 3D U-Net baseline, while recall remained similar and its paired confidence interval included zero. The combined overlap, topology, and boundary-error results support use of the selected Attention U-Net in the image-to-mesh pipeline. They do not establish that attention gates alone caused the differences because the models did not share fully identical training histories or initialization lineages.

## Updated limitation

The segmentation comparison was limited to the Attention U-Net and a conventional 3D U-Net baseline; broader benchmarking against additional modern architectures was outside the scope of this study.

## Baseline table

| Metric | Attention U-Net | Plain 3D U-Net | Difference |
|---|---:|---:|---:|
| Dice@0.5 | 0.7878 (0.7820-0.7935) | 0.7723 (0.7661-0.7785) | +0.0155 |
| clDice@0.5 | 0.8695 (0.8629-0.8761) | 0.8590 (0.8517-0.8663) | +0.0105 |
| Precision@0.5 | 0.7636 (0.7554-0.7719) | 0.7343 (0.7252-0.7434) | +0.0294 |
| Recall@0.5 | 0.8205 (0.8120-0.8290) | 0.8232 (0.8144-0.8319) | -0.0027 |
| HD95@0.5 (mm) | 5.0120 (4.4676-5.5565) | 6.1586 (5.3677-6.9495) | -1.1466 |

Differences are Attention minus plain; lower HD95 is better.
