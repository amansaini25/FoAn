# TransGoalNet Model Evaluation Report

| Metric Category | Primary Tool | Value | What it tells you |
| --- | --- | --- | --- |
| Statistical | Brier Score / MSE | **0.00062** | Is the probability math 'honest'? |
| Tactical | Attention Weights (Max focus) | **0.43154** | Is the model looking at the right players/lanes? |
| Stability | Mean Absolute Change | **0.00027** | Does the xT value jump around too much? |
| Success | Pearson Correlation | **0.66556** | Does 'high xT' actually lead to more points? |
