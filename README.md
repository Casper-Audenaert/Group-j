# Diabetes progression prediction service

Assignment 3: an ML service for a virtual diabetes clinic. It predicts a disease progression score (higher = worse) from the scikit-learn diabetes dataset, so nurses can sort patients by risk and decide who to call first.

## Running the Docker image

```bash
docker pull ghcr.io/casper-audenaert/group-j:v0.2
docker run -it -p 80:80 ghcr.io/casper-audenaert/group-j:v0.2
```

For the baseline version, use `:v0.1` instead.

## Endpoints

### GET /health

```bash
curl http://localhost:80/health
```

```json
{"status": "ok", "model_version": "v0.2"}
```

### POST /predict

All 10 features are required and must be numbers.

```bash
curl -X POST http://localhost:80/predict -H "Content-Type: application/json" -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

```json
{"prediction": 225.89820790392278}
```

## Error responses

Invalid input returns HTTP status 422 with a JSON body. This happens when a feature is missing, a value is not a number, or the JSON is broken. `detail` has one entry per problem:

- `type`: kind of error, e.g. `missing` or `float_parsing`
- `loc`: which field is wrong, e.g. `["body", "s6"]`
- `msg`: readable error message
- `input`: the input that was received

Example with `s6` missing:

```json
{"detail": [{"type": "missing", "loc": ["body", "s6"], "msg": "Field required", "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}}]}
```

## Running it locally

You need Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

ruff check .                     # lint
pytest                           # tests
python -m app.train              # trains the model, saves model/model.joblib and model/metrics.json
uvicorn app.main:app --port 8000 # starts the API on http://localhost:8000
```

Or build the Docker image yourself (the model is trained during the build):

```bash
docker build -t diabetes-service .
docker run -p 80:80 diabetes-service
```

Training uses `random_state=42` everywhere and all package versions are pinned, so you get the same model and metrics every time. The metrics are printed and saved to `model/metrics.json`.

## CI/CD

- `.github/workflows/ci.yml` runs on every push and pull request. It lints, runs the tests, trains the model and uploads the model and metrics as an artifact.
- `.github/workflows/release.yml` runs when a `v*` tag is pushed. It builds the Docker image, tests the running container, pushes it to GHCR and creates a GitHub Release with the metrics and changelog.
