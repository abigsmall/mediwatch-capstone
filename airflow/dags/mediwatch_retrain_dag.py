from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="mediwatch_retraining",
    schedule=None,
    start_date=datetime(2026, 4, 20),
    catchup=False,
    tags=["mediwatch", "retraining"],
)
def mediwatch_retraining():
    @task
    def retrain(dag_run=None):
        from pathlib import Path

        from mediwatch.pipelines.retrain import retrain
        from mediwatch.settings import settings

        config = dag_run.conf or {}
        result = retrain(
            reference_data_path=Path(
                # default so no error is thrown when DAG is manually triggered
                config.get("reference_data_path", str(settings.raw_dataset_path))
            ),
            current_data_path=Path(
                config.get(
                    "current_data_path",
                    # default so no error is thrown when DAG is manually triggered
                    str(settings.processed_dir / "simulated_drift.csv"),
                )
            ),
        )
        print(result)
        return result

    retrain()


mediwatch_retraining()
