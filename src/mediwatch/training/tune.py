import argparse
import json
from typing import Any

try:
    from ray import tune
except Exception:  # pragma: no cover - optional dependency fallback
    tune = None

from mediwatch.dataset import load_raw_dataset, make_dataset_bundle
from mediwatch.training.pipeline import train_and_log


SEARCH_SPACE = {
    "c_value": [0.25, 0.5, 1.0, 2.0],
    "class_weight": ["balanced", None],
}


def trainable(config: dict[str, Any]) -> dict[str, float]:
    bundle = make_dataset_bundle(load_raw_dataset())
    result = train_and_log(
        bundle,
        c_value=float(config["c_value"]),
        class_weight=config["class_weight"],
        register_as_best=False,
    )
    return {"f1": result.f1, "roc_auc": result.roc_auc}


def run_without_ray() -> dict[str, Any]:
    best_result = None
    for c_value in SEARCH_SPACE["c_value"]:
        for class_weight in SEARCH_SPACE["class_weight"]:
            metrics = trainable({"c_value": c_value, "class_weight": class_weight})
            candidate = {
                "c_value": c_value,
                "class_weight": class_weight,
                **metrics,
            }
            if best_result is None or candidate["f1"] > best_result["f1"]:
                best_result = candidate
    return best_result or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    if tune is None:
        print(json.dumps({"backend": "fallback-grid-search", "best": run_without_ray()}, indent=2))
        return

    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 1}),
        tune_config=tune.TuneConfig(metric="f1", mode="max", num_samples=args.samples),
        param_space={
            "c_value": tune.grid_search(SEARCH_SPACE["c_value"]),
            "class_weight": tune.grid_search(SEARCH_SPACE["class_weight"]),
        },
    )
    results = tuner.fit()
    best = results.get_best_result(metric="f1", mode="max")
    print(
        json.dumps(
            {
                "backend": "ray-tune",
                "best_config": best.config,
                "best_metrics": best.metrics,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
