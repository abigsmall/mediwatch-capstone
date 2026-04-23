import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from ray import tune

from mediwatch.dataset import DatasetBundle, load_raw_dataset, make_dataset_bundle
from mediwatch.training.pipeline import train_and_log


SEARCH_SPACE = {
    "c_value": [0.25, 0.5, 1.0, 2.0],
    "class_weight": ["balanced", None],
}


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


def trainable(config: dict[str, Any], bundle: DatasetBundle) -> dict[str, float]:
    result = train_and_log(
        bundle,
        c_value=float(config["c_value"]),
        class_weight=config["class_weight"],
        register_as_best=False,
    )
    return {
        "average_precision": result.average_precision,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
    }


def run_tuning(
    *,
    samples: int = 1,
    data_path: Path | None = None,
    dataframe: pd.DataFrame | None = None,
    bundle: DatasetBundle | None = None,
) -> dict[str, Any]:
    dataset_bundle = _bundle_from_source(bundle=bundle, dataframe=dataframe, data_path=data_path)

    def _trainable(config: dict[str, Any]) -> dict[str, float]:
        return trainable(config, dataset_bundle)

    tuner = tune.Tuner(
        tune.with_resources(_trainable, {"cpu": 1}),
        tune_config=tune.TuneConfig(metric="average_precision", mode="max", num_samples=samples),
        param_space={
            "c_value": tune.grid_search(SEARCH_SPACE["c_value"]),
            "class_weight": tune.grid_search(SEARCH_SPACE["class_weight"]),
        },
    )
    results = tuner.fit()
    best = results.get_best_result(metric="average_precision", mode="max")
    return {
        "backend": "ray-tune",
        "best_config": best.config,
        "best_metrics": best.metrics,
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
