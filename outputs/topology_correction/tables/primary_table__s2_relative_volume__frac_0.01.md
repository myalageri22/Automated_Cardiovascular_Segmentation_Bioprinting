# Primary comparison: s2_relative_volume__frac_0.01

Original = frozen prediction (control). Corrected = s2_relative_volume__frac_0.01. Paired by case, n as shown.

| Metric | Original | Corrected | Paired effect | 95% CI | Adjusted P | Test | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dice | 0.7878 | 0.7896 | 0.7289 (rank_biserial) | [0.0012, 0.0018] | 0.000000 | wilcoxon_signed_rank | 250 |
| clDice | 0.8695 | 0.8763 | 0.7791 (rank_biserial) | [0.0042, 0.0067] | 0.000000 | wilcoxon_signed_rank | 250 |
| HD95 (mm) | 7.564 | 7.477 | -0.1814 (rank_biserial) | [-0.194, 0.000] | 0.019118 | wilcoxon_signed_rank | 250 |
| Components | 11.09 | 4.36 | -1.0000 (rank_biserial) | [-6.00, -5.00] | 0.000000 | wilcoxon_signed_rank | 250 |
| Mesh integrity | 0.9120 | 0.9560 | 0.0440 (paired_rate_difference) | [0.0200, 0.0720] | 0.001221 | mcnemar_exact | 250 |
| Centroid displacement (mm) | 0 (reference) | 0.6394 | 0.6394 (mean displacement vs original geometry) | [0.0154, 1.7352] | not applicable | descriptive | 250 |
| Surface deviation (mm) | 0 (reference) | 0.1447 | 0.1447 (mean displacement vs original geometry) | [0.0004, 0.5828] | not applicable | descriptive | 250 |
