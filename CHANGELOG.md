# Changelog

## [0.2]

### What changed
- Replaced LinearRegression with RidgeCV. The regularisation strength (alpha) is chosen automatically with cross-validation and came out at 1.15.
- Changed the cutoff for the high-risk flag. A patient counts as high-risk if their progression is in the top 25% of the training set (>= 214). In v0.1 we applied 214 directly to the prediction, but the model's predictions are pulled toward the average, so it flagged very few patients. In v0.2 the cutoff is chosen on cross-validated predictions on the training set (best F1), which gives 186.98. The test set is not used for this.

### Metrics
Both versions use the same test split (20%, random_state=42).

| Metric | v0.1 | v0.2 |
|---|---|---|
| RMSE | 53.85 | 53.77 |
| Precision (high-risk) | 1.000 | 0.750 |
| Recall (high-risk) | 0.286 | 0.714 |

### Why v0.2 is better
RMSE is a bit lower, so the risk score is slightly more accurate. The bigger improvement is the high-risk flag: v0.1 caught only 2 of the 7 high-risk patients in the test set, while v0.2 catches 5 of 7. For a triage clinic, missing a patient who is getting worse is worse than making an extra phone call, so the lower precision is worth it.

We also tried RandomForest. It had a lower RMSE on the test split (52.86), but its cross-validation score was worse (57.22 vs 55.35 for Ridge), which suggests it overfits, so we kept Ridge.

## [0.1]

- Baseline model: StandardScaler + LinearRegression.
- FastAPI service with `/predict` and `/health`, Docker image and GitHub Actions pipeline.
- RMSE on the test split (20%, random_state=42): 53.85.
- High-risk flag (prediction >= 214, the top 25% of the training targets): precision 1.000, recall 0.286.
