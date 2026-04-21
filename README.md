# MediWatch Capstone

MediWatch is an end-to-end MLOps capstone project for predicting patient readmission risk. This repository adapts the monitoring & continuous-training assignment structure into a full-fledged capstone with a training pipeline, MLflow experiment tracking, FastAPI inference service, Streamlit UI, Evidently drift detection, and Airflow-driven retraining.

## What is in this repo
- A reusable training pipeline with a single preprocessing-plus-model artifact
- MLflow logging and a small Ray Tune search space
- REST API with Swagger docs at `/docs`
- Streamlit UI for user interaction
- Evidently monitoring with simulated drift generation
- Airflow DAG for retraining and model promotion
- Docker Compose stack for reproducibility
- EDA notebook and project documentation

## Dataset assumption
This repository expects the Kaggle diabetes readmission dataset file to be available at:

`data/raw/diabetic_data.csv`

The code assumes the raw dataset contains the UCI-style `readmitted` column and converts it to a binary target:
- `1` when `readmitted != "NO"`
- `0` when `readmitted == "NO"`

If you want the stricter `<30` interpretation instead, adjust the target mapping in `dataset.py` (mediwatch-capstone/src/mediwatch/dataset.py:37).

## Quick start for local dev
NOTE: Deploying the Dockerfile requires an extra requirements-airflow.txt generation step!

1. Use Python `3.12`.
2. Create and sync the virtual environment with `uv`:

```bash
uv sync
```

4. Copy `.env.example` to `.env` and fill in AWS values if you want S3 uploads.
5. Place `diabetic_data.csv` (from [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes?resource=download)) in `data/raw/`.

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

And again, this part is not for local dev but if deploying to Docker (see [Regenerate Airflow requirements](#regenerate-airflow-requirements)):

```bash
uv run python scripts/generate_requirements_airflow.py
```

### Train a model
```bash
uv run python -m mediwatch.training.train
```

Optional tuning:

```bash
uv run python -m mediwatch.training.tune
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

Run monitoring:

```bash
uv run python -m mediwatch.monitoring.drift --current-data data/processed/simulated_drift.csv --trigger-retraining
```

Outputs:
- `evidently/drift_report.html`
- `evidently/drift_summary.json`

### Run the stack with Docker Compose
```bash
docker compose up --build
```

The container setup remains `uv`-based, but it installs from the project metadata during image build rather than running `uv sync` inside the container.

Recommended services:
- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- MLflow: `http://localhost:5001`
  - note: the dashboard defaults to the "GenAI" toggle in the left sidebar. toggle it to the "Model training" option.
- Airflow: `http://localhost:8080`

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
2. Retrains a candidate model
3. Compares candidate `f1` against the incumbent
4. Promotes the candidate when it is better or equal

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