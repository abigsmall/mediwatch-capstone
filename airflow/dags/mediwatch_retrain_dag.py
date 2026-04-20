from __future__ import annotations

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
        from mediwatch.pipelines.retrain import retrain

        config = dag_run.conf or {}
        result = retrain(
            reference_data_path=config["reference_data_path"],
            current_data_path=config["current_data_path"],
        )
        print(result)
        return result

    retrain()


mediwatch_retraining()
