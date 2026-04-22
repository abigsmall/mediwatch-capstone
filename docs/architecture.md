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
  - `evidently` (planned deployment service, not fully wired into the Compose stack yet)

## Reasons for doing certain things
### Why `uv` instead of `pip`
I found out about `uv` while taking this course and fell in love with it. There were a couple times working on assignments in this SwitchUp when I ran into weird Python dependencies issues and having to learn about Python wheels and such stuff. Even knowing there’s `conda` environments and just multiple ways to set up Python rubbed me the wrong way (how is that Pythonic?) so when I learned about the speed and simplicity of `uv`, I decided to go all in on it. Unfortunately, I ran into a lot of issues trying to get the Airflow Docker setup working with `uv`, partially due to misunderstandings from myself and Codex, but we got to it in the end :).

### Why two `Dockerfile`s (`Dockerfile.app` and `Dockerfile.airflow`)
The Airflow Docker image has its own conventions, from where its entrypoint is to [the permissions that are expected](https://airflow.apache.org/docs/docker-stack/build.html#:~:text=The%20production%20image%20of%20Airflow%20uses%20%E2%80%9Cairflow%E2%80%9D%20user%2C). There’s also an entire database setup process going on.

On the otherhand, the FastAPI, Streamlit, and Evidently apps are more “typical” Python app services, so the separation keeps the deployment/inference concerns apart from from orchestration concerns.

### Why there is no Redis service
`docker-compose.yaml` uses Airflow with `LocalExecutor` instead of `CeleryExecutor`: reduces complexity to both make my development easier and also has a benefit in making the project easier to reproduce on a single EC2 instance for evaluation. Hooray!

With `LocalExecutor`:
- Airflow runs tasks locally
- Postgres is still used for Airflow metadata
- Redis is not required

Redis would be necessary if this project were using `CeleryExecutor` with separate Airflow workers, which is what you’d probably see in production. For this submission, the simpler `LocalExecutor` design keeps the orchestration part intact while lowering the risk of environment issues during evaluation (and for myself during development).

### Why `t3` for the EC2 deployment
At first I was considering a `t2.xlarge` since that’s what the Monitoring & Continuous Training assignment setup instructed. However, since I’m way past the IK-allotted credits timeframe, I’m paying for AWS through my own personal account, and I’ve been noticing my monthly bill go up. One of these days I’ll get some of the data off the dormant EC2 instances I have, but I haven’t done that yet. But moreso than the dormant EC2 instances, it was working on the Monitoring & Continuous Training assignment for over a month that cost me more than I expected. Hence the thought of cost savings finally entered the picture for me.

That’s when I realized there’s something called `t3` and it’s basically the same thing but better and cheaper. Then I saw that there’s `t4g` and can be even cheaper, but that it uses ARM / Graviton. For this capstone, I would rather avoid introducing another potential source of compatibility issues across Docker images, Python dependencies, and ML-related libraries. Using `t3` keeps the deployment on `x86_64`.

### Why `[tool.setuptools.packages.find]` in pyproject.toml
This is a cool feature (among many, many other things) that Codex let me know about: it allows users, once they’ve installed the Python dependencies, to run the Python scripts in this environment without having to specify either absolute path names or get the relative path correct.

## Future scope
- Replace Docker Compose orchestration with Kubernetes or EKS.
- Move Airflow to MWAA (AWS-managed Airflow) or even MWAA Serverless.
- Rely on MLflow Models to serve the model inferencing endpoint instead of having our own separate FastAPI endpoint.
- Add randomization to `AIRFLOW__WEBSERVER__SECRET_KEY` as per the [Airflow docs](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html#:~:text=It%20should%20be%20as%20random%20as%20possible.).
- Hook into some Python-native way to update the requirements-airflow.txt when pyproject.toml is updated instead of having to call [`scripts/generate_requirements_airflow.py`](../scripts/generate_requirements_airflow.py).
- Even better, figure out how to get the `Dockerfile.airflow` setup not rely on a separate requirements-airflow.txt.
- For `Dockerfile.app`, use [the setup recommended](https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile) by Astral.
- Add richer monitoring metrics and model performance dashboards.
- Listen to the `FutureWarning` from MLflow and log runs to a database like SQLite (currently it’s just logging to a local folder).
