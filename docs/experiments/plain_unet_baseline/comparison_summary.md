# Attention vs Plain Exact-250 Summary

| Metric | Attention U-Net | Plain 3D U-Net | Difference |
|---|---:|---:|---:|
| Dice@0.5 | 0.7878 (0.7820-0.7935) | 0.7723 (0.7661-0.7785) | +0.0155 |
| clDice@0.5 | 0.8695 (0.8629-0.8761) | 0.8590 (0.8517-0.8663) | +0.0105 |
| Precision@0.5 | 0.7636 (0.7554-0.7719) | 0.7343 (0.7252-0.7434) | +0.0294 |
| Recall@0.5 | 0.8205 (0.8120-0.8290) | 0.8232 (0.8144-0.8319) | -0.0027 |
| HD95@0.5 (mm) | 5.0120 (4.4676-5.5565) | 6.1586 (5.3677-6.9495) | -1.1466 |

Differences are Attention minus plain; lower HD95 is better.
