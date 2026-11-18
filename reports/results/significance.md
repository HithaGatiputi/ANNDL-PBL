# Significance tests (test set, seed-ensembled predictions)

McNemar: exact binomial test on disagreements at threshold 0.5. DM: Diebold–Mariano (HLN-corrected) on squared-error (Brier) loss; negative stat = model_a better.

| model_a    | model_b      | subset     |    n |   acc_a |   acc_b |   a_right_b_wrong |   a_wrong_b_right |   mcnemar_p |   brier_a |   brier_b |   dm_stat |   dm_p | significant_5pct   |
|:-----------|:-------------|:-----------|-----:|--------:|--------:|------------------:|------------------:|------------:|----------:|----------:|----------:|-------:|:-------------------|
| static_gnn | rcdgnn_daily | all        | 1159 |   0.528 |  0.528  |               111 |               111 |      1      |    0.249  |    0.2494 |   -0.8663 | 0.3865 | False              |
| static_gnn | rcdgnn_daily | post_event |  141 |   0.461 |  0.5106 |                11 |                18 |      0.2649 |    0.2517 |    0.252  |   -0.1326 | 0.8947 | False              |