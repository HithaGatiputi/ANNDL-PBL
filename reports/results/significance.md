# Significance tests (test set, seed-ensembled predictions)

McNemar: exact binomial test on disagreements at threshold 0.5. DM: Diebold–Mariano (HLN-corrected) on squared-error (Brier) loss; negative stat = model_a better.

| model_a       | model_b      | subset     |    n |   acc_a |   acc_b |   a_right_b_wrong |   a_wrong_b_right |   mcnemar_p |   brier_a |   brier_b |   dm_stat |   dm_p | significant_5pct   |
|:--------------|:-------------|:-----------|-----:|--------:|--------:|------------------:|------------------:|------------:|----------:|----------:|----------:|-------:|:-------------------|
| static_gnn    | rcdgnn_daily | all        | 1159 |  0.528  |  0.528  |               111 |               111 |      1      |    0.249  |    0.2494 |   -0.8663 | 0.3865 | False              |
| rcdgnn_daily  | rcdgnn_full  | all        | 1159 |  0.528  |  0.5289 |                19 |                20 |      1      |    0.2494 |    0.2495 |   -1.0271 | 0.3046 | False              |
| baseline_lstm | rcdgnn_full  | all        | 1159 |  0.5324 |  0.5289 |               144 |               140 |      0.8587 |    0.249  |    0.2495 |   -0.726  | 0.468  | False              |
| static_gnn    | rcdgnn_daily | post_event |  141 |  0.461  |  0.5106 |                11 |                18 |      0.2649 |    0.2517 |    0.252  |   -0.1326 | 0.8947 | False              |
| rcdgnn_daily  | rcdgnn_full  | post_event |  141 |  0.5106 |  0.5035 |                 1 |                 0 |      1      |    0.252  |    0.2519 |    1.0003 | 0.3189 | False              |
| baseline_lstm | rcdgnn_full  | post_event |  141 |  0.4539 |  0.5035 |                11 |                18 |      0.2649 |    0.2513 |    0.2519 |   -0.2511 | 0.8021 | False              |