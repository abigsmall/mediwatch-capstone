import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from ray import tune

from mediwatch.dataset import DatasetBundle, load_raw_dataset, make_dataset_bundle
from mediwatch.settings import settings
from mediwatch.training.pipeline import train_and_log


LOGISTIC_REGRESSION_SEARCH = [
    {
        "model_type": "logistic_regression",
        "c_value": c_value,
        "class_weight": class_weight,
    }
    for c_value, class_weight in product([0.25, 0.5, 1.0, 2.0], ["balanced", None])
]

RANDOM_FOREST_SEARCH = [
    {
        "model_type": "random_forest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
    }
    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        [200, 400],
        [None, 12],
        [1, 3],
        ["balanced", None],
    )
]

EXTRA_TREES_SEARCH = [
    {
        "model_type": "extra_trees",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
    }
    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        [200, 400],
        [None, 12],
        [1, 3],
        ["balanced", None],
    )
]

SEARCH_SPACE = LOGISTIC_REGRESSION_SEARCH + RANDOM_FOREST_SEARCH + EXTRA_TREES_SEARCH


# the `*` means everything after has to be passed by keyword, not position,
# which is useful here for clarity since the arguments are mutually exclusive
def _bundle_from_source(
    *,
    bundle: DatasetBundle | None = None,
    dataframe: pd.DataFrame | None = None,
    data_path: Path | None = None,
) -> DatasetBundle:
    if bundle is not None:
        return bundle
    if dataframe is not None:
        return make_dataset_bundle(dataframe)
    return make_dataset_bundle(load_raw_dataset(data_path))


def trainable(
    config: dict[str, Any],
    bundle: DatasetBundle,
    parent_run_id: str | None = None,
) -> dict[str, float]:
    model_config = dict(config["model_config"])
    result = train_and_log(
        bundle,
        model_config=model_config,
        register_as_best=False,
        parent_run_id=parent_run_id,
        extra_tags={
            "run_type": "tuning_trial",
            "model_type": model_config["model_type"],
        },
        run_name_prefix=f"mediwatch-tune-trial-{model_config['model_type']}",
    )
    return {
        "average_precision": result.average_precision,
        "f1": result.f1,
        "precision": result.precision,
        "recall": result.recall,
        "roc_auc": result.roc_auc,
        "positive_rate": result.positive_rate,
    }


def run_tuning(
    *,
    samples: int = 1,
    data_path: Path | None = None,
    dataframe: pd.DataFrame | None = None,
    bundle: DatasetBundle | None = None,
) -> dict[str, Any]:
    dataset_bundle = _bundle_from_source(bundle=bundle, dataframe=dataframe, data_path=data_path)
    promotion_metric = settings.promotion_metric
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    tuning_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with mlflow.start_run(run_name=f"mediwatch-tune-{tuning_started_at}") as parent_run:
        mlflow.set_tags(
            {
                "run_type": "tuning_campaign",
                "backend": "ray-tune",
                "promotion_metric": promotion_metric,
            }
        )
        mlflow.log_params(
            {
                "num_samples": samples,
                "max_concurrent_trials": settings.tune_max_concurrent_trials,
                "search_space_size": len(SEARCH_SPACE),
            }
        )

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    trainable,
                    bundle=dataset_bundle,
                    parent_run_id=parent_run.info.run_id,
                ),
                {"cpu": 1},
            ),
            tune_config=tune.TuneConfig(
                metric=promotion_metric,
                mode="max",
                num_samples=samples,
                max_concurrent_trials=settings.tune_max_concurrent_trials,
            ),
            param_space={
                "model_config": tune.grid_search(SEARCH_SPACE),
            },
        )
        results = tuner.fit()
        best = results.get_best_result(metric=promotion_metric, mode="max")
        best_config = best.config["model_config"]
        numeric_best_metrics = {
            key: value
            for key, value in best.metrics.items()
            if isinstance(value, int | float)
        }
        mlflow.log_metrics(numeric_best_metrics)
        mlflow.log_dict(best_config, "best_config.json")

        return {
            "backend": "ray-tune",
            "best_config": best_config,
            "best_metrics": best.metrics,
            "promotion_metric": promotion_metric,
            "parent_run_id": parent_run.info.run_id,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--data-path")
    args = parser.parse_args()

    result = run_tuning(
        samples=args.samples,
        data_path=Path(args.data_path) if args.data_path else None,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
