# MediWatch Architecture

## Core flow
- `data/raw/diabetic_data.csv` is the source dataset.
- `mediwatch.training.train` runs preprocessing, model training, MLflow logging, and artifact export.
- `models/latest_model.json` acts as the deployment manifest for the active model.
- `FastAPI` serves `/health`, `/predict`, and `/predict-batch`.
- `Streamlit` provides the evaluator-facing UI.
- `mediwatch.monitoring.drift` scores current data, runs Evidently drift checks, writes an HTML report, and optionally triggers Airflow.
- `Airflow` runs the retraining DAG and promotes a candidate model when it beats the incumbent on `f1`.

## Infrastructure
- Primary deployment target: `EC2 + S3 + Docker Compose`
- Services:
  - `api`
  - `ui`
  - `mlflow`
  - `postgres`
  - `airflow-webserver`
  - `airflow-scheduler`

## Future scope
- Replace Docker Compose orchestration with Kubernetes or EKS.
- Move Airflow to MWAA if more time is available.
- Add richer monitoring metrics and model performance dashboards.
