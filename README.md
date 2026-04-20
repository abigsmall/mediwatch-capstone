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

## Quick start
1. Use Python `3.11` or `3.12`.
2. Create a virtual environment with `uv`.
3. Install dependencies with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in AWS values if you want S3 uploads.
5. Place `diabetic_data.csv` in `data/raw/`.

You can also sync directly from `pyproject.toml`:

```bash
uv pip install -e .
```

## Train a model
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

## Run the API
```bash
uv run uvicorn mediwatch.api.app:app --reload
```

Endpoints:
- `GET /health`
- `POST /predict`
- `POST /predict-batch`

## Run the UI
```bash
uv run streamlit run src/mediwatch/ui/streamlit_app.py
```

## Simulate drift and monitor
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

## Run the stack with Docker Compose
```bash
docker compose up --build
```

Recommended services:
- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`

## Airflow retraining
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

## Future enhancements
- Kubernetes or EKS deployment
- MWAA-managed orchestration
- richer feature store and data validation
- more sophisticated champion/challenger logic
