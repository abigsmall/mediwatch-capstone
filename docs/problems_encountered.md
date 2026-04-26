# Problems I Encountered

This document summarizes the major issues I ran into while building and validating the MediWatch capstone project, along with the root causes and the fixes or workarounds used.

The biggest headache so far has been with environment/dependency management and the libraries in Airflow not being able to see files/write to certain directories. For all my praise about [`uv`](https://docs.astral.sh/uv/), and I was doing everything with `uv` locally, I ran into so many issues trying to get it to work on the Docker side. Then the interaction between how I set up the Docker images (`Dockerfile.app` and `Dockerfile.airflow`), which at first sounded like a good idea since I was going to package the former into its own Kubernetes deployment, caused constant friction when MLflow or Ray Tune could not see something because it was prefaced with `/app` or the permissions for that file were not the same as the Airflow user, etc.

## General Lessons
The biggest recurring theme was environment consistency:
- local `uv` environment vs Docker image environment
- app container vs Airflow container
- host filesystem behavior vs container filesystem behavior

Another recurring theme was that a working artifact was not enough by itself:
- the model could train but still leak the target
- the drift report could generate but still be summarized incorrectly
- Airflow could start but still fail due to missing secrets, dependencies, or host filesystem issues

The project became more stable each time I reduced hidden assumptions and made the runtime contracts more explicit.

What follows is each issue I remembered to record in unnecessary detail.

## Environment and Setup Issues
### `uv` workflow and package installation confusion
I initially noticed that the repo still referenced `pip`-style workflows and a `requirements.txt`, even though I was primarily using [`uv`](https://docs.astral.sh/uv/).

What happened:
- I installed dependencies with `uv pip install -r requirements.txt`.
- I later realized that `requirements.txt` only contained `-e .`, so it was just acting as a wrapper around an editable install of the project.

Resolution:
- The repo was updated to be `uv`-first.
- `requirements.txt` was removed.
- The project now uses `uv sync` / `uv pip install -e .` locally.
- `uv.lock` and `pyproject.toml` are treated as the source of truth.

## Training and Data Pipeline Issues
### `pandas.NA` causing scikit-learn preprocessing failures

Training failed later inside preprocessing.

What happened:
- Missing values represented by `"?"` in the diabetes dataset were being converted to `pandas.NA`.
- scikit-learn's imputers and related preprocessing logic handle `numpy.nan` more reliably than `pandas.NA`.
- This led to the classic error:

```text
TypeError: boolean value of NA is ambiguous
```

Resolution:
- Replaced `pd.NA` with `np.nan` during dataset normalization.
- This made the missing values compatible with the scikit-learn pipeline.

### Target leakage causing unrealistically perfect model metrics

One training run completed successfully but returned suspiciously perfect metrics:
- `f1 = 1.0`
- `precision = 1.0`
- `recall = 1.0`
- `roc_auc = 1.0`

What happened:
- The dataset preprocessing logic created a derived target column named `readmitted_binary`.
- The original `readmitted` column was removed from the feature frame, but `readmitted_binary` was accidentally left in the feature set.
- The model was therefore learning directly from the target itself.

Resolution:
- Added `readmitted_binary` to the list of leaky columns excluded from features.
- Added a regression test to prevent target columns from accidentally appearing in training features again.

Outcome:
- Metrics dropped to a much more believable baseline:
  - `f1`: about `0.59`
  - `roc_auc`: about `0.67`

## Monitoring and Evidently Issues
### Evidently monitoring script was written against the wrong API shape

The initial `mediwatch.monitoring.drift` command crashed with an error claiming that Evidently was not installed or that the imports/API were wrong.

What happened:
- The capstone `drift.py` had drifted away from the Evidently API version actually installed in the environment.
- It used incompatible imports and method calls.
- In particular, it did not match the working style from the original monitoring assignment code.

Resolution:
- Compared it against the working assignment implementation in:
  - `ml-monitoring-and-continuous-training-assignment/assignment/test_and_create_reports_and_test_suite.py`
- Reworked `src/mediwatch/monitoring/drift.py` to use the newer Evidently pattern based on:
  - `DataDefinition`
  - `Dataset.from_pandas(...)`
  - `Report(...)`
  - `run = report.run(...)`
  - `run.dict()`

### Drift report generated successfully, but summary incorrectly showed no drift

After fixing the Evidently API usage, the report ran and output files were created, but the summary still said:
- `drift_detected: false`
- `share_of_drifted_columns: 0.0`

What happened:
- The JSON parsing logic in the summary code was reading the Evidently output incorrectly.
- The installed Evidently version emitted numeric drift metric values rather than the `drift_detected` structure the code expected.

Resolution:
- Updated the summary logic to inspect:
  - `metric["config"]["column"]`
  - `metric["config"]["threshold"]`
  - `metric["value"]`
- A column is now marked drifted when `value > threshold`.

Outcome:
- Drift detection correctly reported:
  - `drift_detected: true`
  - `share_of_drifted_columns: 0.5`
  - drift in `num_medications`, `number_inpatient`, `prediction_probability`, and `time_in_hospital`

## Docker and Compose Issues
### Docker Desktop was not running

I hit a Docker connection error before the stack could even start.

What happened:
- Docker Desktop was not running on macOS.

Resolution:
- Started Docker Desktop and reran the Compose commands.

### Port `5000` conflict with macOS Control Center

The Compose stack failed because MLflow could not bind to port `5000`.

What happened:
- On macOS, Control Center was already listening on port `5000` (`commplex-main`).
- This conflicted with the default MLflow port mapping.

Resolution:
- Changed the host-side MLflow port mapping from `5000:5000` to `5001:5000`.
- MLflow remains on port `5000` inside Docker, but is reached at `http://localhost:5001` on the host.

### Docker build issues caused by packaging metadata and editable installs

The Airflow image went through several build failures before stabilizing.

What happened:
- Early Dockerfiles tried to install the project in editable mode inside the image.
- This caused issues involving `.egg-info` metadata, file timestamps, and permissions.
- I also ran into mismatches between:
  - local `uv` expectations,
  - Airflow's runtime environment,
  - and how custom packages are layered into official Airflow images.

Resolution:
- Added a `.dockerignore` so local metadata and build artifacts would not contaminate the image build context.
- Treated the application image and Airflow image separately.
- For the Airflow image, moved back toward the assignment-style approach of installing dependencies directly into the Airflow image rather than forcing the same `uv` workflow everywhere.

## Airflow Issues
### Airflow webserver initially failed because the database was not ready

At one point only the scheduler came up, while the webserver exited with:

```text
You need to initialize the database. Please run `airflow db init`.
```

What happened:
- `airflow-init` succeeded, but `airflow-webserver` started before initialization had fully completed.
- This was effectively a startup-order race condition.

Resolution:
- Updated `docker-compose.yml` so `airflow-webserver` and `airflow-scheduler` wait for `airflow-init` to complete successfully.

### Airflow log viewing failed because components had different secret keys

After the webserver came up, task log viewing still failed with a `403 FORBIDDEN` error and a warning about the Airflow `secret_key`.

What happened:
- The Airflow components were not all using the same `AIRFLOW__WEBSERVER__SECRET_KEY`.
- The webserver could not validate log requests from the scheduler-side log server.

Resolution:
- Added a shared `AIRFLOW__WEBSERVER__SECRET_KEY` to the Compose environment and `.env.example`.
- Restarted the Airflow services so they all used the same key.

### Missing Python dependencies in Airflow tasks

Once task logs were visible, the DAG still failed because imports such as `joblib` and later `mlflow` were missing.

What happened:
- The Airflow runtime environment did not yet have all of the Python packages needed by the retraining task and the modules it imported.
- This required a different dependency strategy than the regular app image.

Resolution:
- Added an Airflow-specific requirements file.
- Later replaced the manually curated file with a generated one, produced from `pyproject.toml` using a helper script:
  - `scripts/generate_requirements_airflow.py`

### Manual DAG triggers failed because required config keys were missing

After the dependency issues were resolved, a manual DAG run failed with:

```text
KeyError: 'reference_data_path'
```

What happened:
- The DAG expected `dag_run.conf` to include keys such as:
  - `reference_data_path`
  - `current_data_path`
- Those keys were not provided when manually triggering the DAG from the UI.

Resolution:
- Added sensible defaults in `airflow/dags/mediwatch_retrain_dag.py` so the DAG can be triggered manually without custom JSON.

### `OSError: [Errno 35] Resource deadlock avoided` while reading CSVs in Airflow

Later, the retraining task failed while trying to read CSV files with pandas.

What happened:
- The project was being run from a Dropbox / CloudStorage-backed macOS folder.
- Docker bind mounts from cloud-synced directories on macOS can behave badly.
- The error:

```text
OSError: [Errno 35] Resource deadlock avoided
```

strongly suggested that the problem was with the host filesystem mount, not with pandas itself.

Resolution / workaround:
- Planned to push the current work to GitHub and re-clone it into a normal local directory outside Dropbox/CloudStorage.
- This should provide a more stable filesystem for Docker and Airflow.

## Kubernetes and EC2 Deployment Issues
### Minikube was not available through `apt`

On a fresh EC2 instance, the first attempt to start Minikube failed because the `minikube` command was not installed.

What happened:
- Running `minikube start --driver=docker --ports=30080:30080` returned `Command 'minikube' not found`.
- Trying `sudo apt install minikube` then failed with:

```text
E: Unable to locate package minikube
```

Resolution:
- Installed Minikube from the official Linux Debian package URL instead of relying on Ubuntu's default apt repositories:

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
minikube version
```

### Minikube Docker driver failed due to Docker permissions

After Minikube was installed, the Docker driver still could not start cleanly at first.

What happened:
- `minikube start --driver=docker --ports=30080:30080` needs access to the local Docker daemon.
- The EC2 user did not yet have permission to use Docker without `sudo`.
- This commonly appears as a permission-denied error while connecting to the Docker socket.

Resolution:
- Added the current user to the `docker` group and refreshed the shell session:

```bash
sudo usermod -aG docker $USER
newgrp docker
docker ps
```

- If the permission change does not take effect immediately, log out of SSH and log back in.
    - This may not be necessary, but I actually had to restart the EC2 instance entirely before `groups` finally showed `docker` in its result list.
- After that, Minikube should be started as the normal user, not with `sudo`.
