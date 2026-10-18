# Changelog

## [0.1]

- Baseline model: StandardScaler + LinearRegression.
- FastAPI service with `/predict` and `/health`, Docker image and GitHub Actions pipeline.
- RMSE on the test split (20%, random_state=42): 53.85.
- High-risk flag (prediction >= 214, the top 25% of the training targets): precision 1.000, recall 0.286.
