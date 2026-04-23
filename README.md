# MediWatch Capstone

MediWatch is an end-to-end MLOps capstone project for predicting patient readmission risk. This repository adapts the monitoring & continuous-training assignment structure into a full-fledged capstone with a training pipeline, MLflow experiment tracking, FastAPI inference service, Streamlit UI, Evidently drift detection, and Airflow-driven retraining.

## What is in this repo
- A reusable training pipeline with a single preprocessing-plus-model artifact
- MLflow logging and a small Ray Tune search space
- FastAPI REST API with Swagger docs at `/docs`
- Streamlit UI for user interaction
- Evidently monitoring with simulated drift generation
- Airflow DAG for retraining and model promotion
- Docker Compose stack for reproducibility
- EDA notebook and project documentation

## Dataset assumption
This repository expects the Kaggle diabetes readmission dataset file (available [here](https://www.kaggle.com/datasets/brandao/diabetes?resource=download)) to be available at:

`data/raw/diabetic_data.csv`

The code assumes the raw dataset contains a `readmitted` column with values like `NO`, `<30`, and `>30`, converting it to a binary target:
- `1` when `readmitted != "NO"`
- `0` when `readmitted == "NO"`

If you want the stricter `<30` interpretation instead, adjust the target mapping in `dataset.py` (mediwatch-capstone/src/mediwatch/dataset.py:37).

## Recommended setup: Docker Compose
Docker Compose is the primary and recommended way to run this project. It is the most reproducible setup for evaluation because it brings up the API, Streamlit UI, MLflow, Airflow, and supporting services together in a consistent environment.

TODO: We may need to update the above and below with Kubernetes information if we get to that

Before starting the stack:

1. Copy `.env.example` to `.env` and fill in an S3 bucket name
2. Place `diabetic_data.csv` (from [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes?resource=download)) in `data/raw/`.
3. If you changed project dependencies, regenerate the Airflow dependency file as described in [Regenerate Airflow requirements](#regenerate-airflow-requirements).

Start the stack with:

```bash
docker compose up --build
```

Service endpoints from the host machine:
- FastAPI API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- MLflow: `http://localhost:5001`
  - On first load the dashboard defaults to the "GenAI" toggle in the left sidebar. Toggle it to the "Model training" option.
  - Due to contention with port 5000 on macOS, from the host machine MLflow is reached at `http://localhost:5001`
    - from other containers in the Compose network, `http://mlflow:5000`
- Airflow: `http://localhost:8080`

### First-run bootstrap
On a fresh setup, the retraining DAG does not have any model to compare against yet. Before using the Airflow retraining flow for the first time, run one training job to create:

- the initial model artifact
- `models/latest_model.json`
- the S3-backed manifest/model state used by retraining

If you run that training command from your host shell, be careful with the MLflow tracking URI:

- inside Docker services, MLflow is `http://mlflow:5000`
- from the host machine, MLflow is `http://localhost:5001`

One reliable host-side bootstrap sequence is:

```bash
set -a
source .env
set +a
export MLFLOW_TRACKING_URI=http://localhost:5001
uv run python -m mediwatch.training.train
```

After that baseline run succeeds, the Airflow retraining DAG can be triggered normally.

## Optional local development workflow
The `uv` workflow below is useful for development, debugging, or generating artifacts quickly on the host, but it is not the primary recommended evaluation path.

1. Use Python `3.12`.
2. Create and sync the virtual environment with `uv`:

```bash
uv sync
```

4. Copy `.env.example` to `.env` and fill in with your S3 bucket values which MLflow and Airflow both rely on.
5. Place `diabetic_data.csv` (from [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes?resource=download)) in `data/raw/`.
6. If your host user is not UID/GID `1000`, update `LOCAL_UID` and `LOCAL_GID` in `.env` before starting Docker Compose so bind-mounted output files stay owned by your host user.

Activate the environment when you want to run commands directly:

```bash
source .venv/bin/activate
```

The repository includes a committed `uv.lock`, so `uv sync` installs the locked dependency set for reproducible setup. If you want to force the exact locked environment without letting `uv` refresh anything, use:

```bash
uv sync --frozen
```

If you intentionally change dependencies in `pyproject.toml`, regenerate the lockfile with:

```bash
uv lock
uv sync
```

This following part is not strictly for local dev but if deploying Docker and you intentionally change dependencies in `pyproject.toml`, in addition to regenerating the lockfile with the above commands, regenerate the `requirements-airflow.txt` file with the below script (see [Regenerate Airflow requirements](#regenerate-airflow-requirements)):

```bash
uv run python scripts/generate_requirements_airflow.py
```

### Train a model
```bash
uv run python -m mediwatch.training.train
```

If you are running this command from the host while the Docker Compose stack is up, make sure `MLFLOW_TRACKING_URI` points to `http://localhost:5001` rather than the container-only hostname `http://mlflow:5000`.

Optional tuning:

```bash
uv run python -m mediwatch.training.tune
```

Tune against a specific dataset path:

```bash
uv run python -m mediwatch.training.tune --data-path data/processed/simulated_drift.csv
```

Artifacts written locally:
- `models/mediwatch_model_<timestamp>.joblib`
- `models/latest_model.json`

### Run the API
```bash
uv run uvicorn mediwatch.api.app:app --reload
```

Endpoints:
- `GET /health`
- `POST /predict`
- `POST /predict-batch`

### Run the UI
```bash
uv run streamlit run src/mediwatch/ui/streamlit_app.py
```

### Simulate drift and monitor
Generate a shifted dataset:

```bash
uv run python -m mediwatch.pipelines.simulate_drift --output data/processed/simulated_drift.csv
```

By default this resamples the source data each run, so repeated executions can produce different drifted datasets. If you want the same generated drift dataset again for a reproducible demo, pass a seed:

```bash
uv run python -m mediwatch.pipelines.simulate_drift --output data/processed/simulated_drift.csv --seed 42
```

Run monitoring:

```bash
uv run python -m mediwatch.monitoring.drift --current-data data/processed/simulated_drift.csv --trigger-retraining
```

If you run that command from the `api` container, files written into bind-mounted folders such as `evidently/` use `LOCAL_UID` and `LOCAL_GID` from `.env` so they remain writable from your host shell.

Outputs:
- `evidently/drift_report.html`
- `evidently/drift_summary.json`

### Regenerate Airflow requirements
The Airflow image uses `requirements-airflow.txt`, which should be generated from `pyproject.toml` instead of edited by hand.

Regenerate it with:

```bash
uv run python scripts/generate_requirements_airflow.py
```

This script keeps ranged requirements from `pyproject.toml` and excludes API/UI-only packages from the Airflow runtime dependency file by default.

Run it again whenever the project dependencies change and before rebuilding the Airflow image.

### Airflow retraining
The DAG lives at `mediwatch_retrain_dag.py` (mediwatch-capstone/airflow/dags/mediwatch_retrain_dag.py:1). It accepts:
- `reference_data_path`
- `current_data_path`
- `drift_summary_path`
- `promotion_metric`

The retraining pipeline:
1. Combines reference and current data
2. Runs tuning on the combined data to choose the candidate hyperparameters
3. Retrains a candidate model with the best config
4. Compares the candidate `promotion_metric` against the incumbent
5. Promotes the candidate when it is better or equal

Important:
- on a fresh environment, run one baseline training job before triggering the DAG so that a baseline manifest/model state exists

## Deployment on EC2
This repo assumes that your EC2 instance has a role allowing it access to the S3 bucket specified in [`.env`](.env): `MEDIWATCH_S3_BUCKET`. IK has provided a document with steps on how to do this: [Misc1: IK <> Creating an IAM Role to Access S3 from EC2](https://docs.google.com/document/d/14HVH5vkPEaVlwsR5utBhuuvBgleHnKRld0AUuxXd9Zs/edit?tab=t.0).

## Suggested demo script
1. Show MLflow runs after training.
2. Open Swagger and score a patient payload.
3. Use the Streamlit UI for a prediction.
4. Generate simulated drift.
5. Run the Evidently monitoring command and open the report.
6. Trigger or show the Airflow DAG run.
7. Show the updated active model manifest after promotion.

## Submission checklist
- Colab or Jupyter notebook for EDA and model selection narrative
- Code repository zip
- Documentation PDF or markdown export
- Demo video link
- Architecture overview
- Setup and rerun instructions
- Future enhancements section
- Committed `uv.lock` for reproducible dependency setup

## Future enhancements
See [Future Scope](docs/architecture.md#future-scope).
