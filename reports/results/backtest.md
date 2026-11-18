# Simplified backtest (test set) — NOT evidence of tradable profit

Long when P(up) > 0.55, flat otherwise; 2 bp per position change; no slippage, financing or spread modelling. The 2022–2026 gold rally makes buy-and-hold hard to beat.

| variant            |   days_long |   trades |   hit_rate |   sharpe |   total_log_ret |   max_drawdown |
|:-------------------|------------:|---------:|-----------:|---------:|----------------:|---------------:|
| baseline_majority  |           0 |        0 |   nan      |   0      |          0      |         0      |
| baseline_logreg    |         150 |      212 |     0.66   |   0.6982 |          0.3231 |        -0.1458 |
| baseline_lstm      |           2 |        2 |     0.5    |   0.0578 |          0.0097 |        -0.05   |
| baseline_lstm_news |          89 |      108 |     0.5393 |   0.3673 |          0.1137 |        -0.0899 |
| static_gnn         |          12 |       24 |     0.5    |  -0.2524 |         -0.0165 |        -0.031  |
| rcdgnn_daily       |          56 |       54 |     0.4643 |  -0.534  |         -0.1689 |        -0.2522 |
| buy_and_hold       |        1159 |        1 |   nan      |   0.9565 |          0.8056 |        -0.3034 |