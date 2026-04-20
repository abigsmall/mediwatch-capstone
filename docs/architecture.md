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
- Dependency workflow: `uv` with a committed `uv.lock`
- Services:
  - `api`
  - `ui`
  - `mlflow`
  - `postgres`
  - `airflow-webserver`
  - `airflow-scheduler`

## Reasons for doing certain things
### Why `uv` instead of `pip`
I found out about `uv` while taking this course and fell in love with it. There were a couple times working on assignments in this SwitchUp when I ran into weird Python dependencies issues and having to learn about Python wheels and such stuff. Even knowing there’s `conda` environments and just multiple ways to set up Python rubbed me the wrong way (how is that Pythonic?) so when I learned about the speed and simplicity of `uv`, I decided to go all in on it.

### Why there is no Redis service
`docker-compose.yaml` uses Airflow with `LocalExecutor` instead of `CeleryExecutor`: reduces complexity to both make my development easier and also has a benefit in making the project easier to reproduce on a single EC2 instance for evaluation. Hooray!

With `LocalExecutor`:
- Airflow runs tasks locally
- Postgres is still used for Airflow metadata
- Redis is not required

Redis would be necessary if this project were using `CeleryExecutor` with separate Airflow workers, which is what you’d probably see in production. For this submission, the simpler `LocalExecutor` design keeps the orchestration part intact while lowering the risk of environment issues during evaluation (and for myself during development).

### Why `[tool.setuptools.packages.find]` in pyproject.toml
This is a cool feature (among many, many other things) that Codex let me know about: it allows users, once they’ve installed the Python dependencies, to run the Python scripts in this environment without having to specify either absolute path names or get the relative path correct.

## Future scope
- Replace Docker Compose orchestration with Kubernetes or EKS.
- Move Airflow to MWAA if more time is available.
- Add richer monitoring metrics and model performance dashboards.
- Listen to the `FutureWarning` from MLflow and log runs to a database like SQLite (currently it’s just logging to a local folder).
