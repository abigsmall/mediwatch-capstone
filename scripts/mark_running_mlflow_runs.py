import argparse

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from mediwatch.settings import settings


def mark_running_runs(experiment_name: str | None, status: str) -> list[str]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        experiments = [experiment] if experiment is not None else []
    else:
        experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)

    updated_run_ids = []
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        for run in runs:
            client.set_terminated(run.info.run_id, status=status)
            updated_run_ids.append(run.info.run_id)

    return updated_run_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=settings.mlflow_experiment_name)
    parser.add_argument("--status", choices=["KILLED", "FAILED"], default="KILLED")
    args = parser.parse_args()

    updated_run_ids = mark_running_runs(args.experiment_name, args.status)
    print(f"Updated {len(updated_run_ids)} RUNNING MLflow run(s) to {args.status}.")
    for run_id in updated_run_ids:
        print(run_id)


if __name__ == "__main__":
    main()
