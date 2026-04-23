import argparse
import json
from pathlib import Path

import pandas as pd

from mediwatch.artifacts import load_model, read_manifest, upload_file_to_s3, write_manifest
from mediwatch.dataset import normalize_raw_dataframe, make_dataset_bundle
from mediwatch.settings import settings
from mediwatch.training.pipeline import evaluate_pipeline, train_and_log
from mediwatch.training.tune import run_tuning


def retrain(reference_data_path: Path, current_data_path: Path) -> dict:
    reference_df = normalize_raw_dataframe(pd.read_csv(reference_data_path))
    current_df = normalize_raw_dataframe(pd.read_csv(current_data_path))
    combined_df = pd.concat([reference_df, current_df], ignore_index=True)

    bundle = make_dataset_bundle(combined_df)
    tuning_result = run_tuning(bundle=bundle)
    best_config = tuning_result.get("best_config", {})
    candidate = train_and_log(
        bundle,
        c_value=float(best_config.get("c_value", 1.0)),
        class_weight=best_config.get("class_weight", "balanced"),
        register_as_best=False,
    )
    incumbent_manifest = read_manifest()
    incumbent_model = load_model()
    # the `f1` metric from the manifest does not account for the new (`current`) data, so evaluate it on data that includes that
    incumbent_metrics = evaluate_pipeline(incumbent_model, bundle.test_x, bundle.test_y)
    incumbent_metric = incumbent_metrics["f1"]

    promoted = candidate.f1 >= incumbent_metric
    if promoted:
        model_key = f"{settings.s3_prefix}/models/{Path(candidate.model_path).name}"
        s3_model_uri = upload_file_to_s3(Path(candidate.model_path), model_key)
        new_manifest = {
            "project": incumbent_manifest.get("project", "MediWatch"),
            "model_version": candidate.model_version,
            "model_path": candidate.model_path,
            "s3_model_uri": s3_model_uri,
            "promotion_metric": "f1",
            "metrics": {
                "f1": candidate.f1,
                "precision": candidate.precision,
                "recall": candidate.recall,
                "roc_auc": candidate.roc_auc,
            },
            "params": candidate.params,
            "mlflow_run_id": candidate.run_id,
            "candidate_from_retraining": True,
        }
        write_manifest(new_manifest)

    return {
        "tuning_backend": tuning_result.get("backend"),
        "best_config": best_config,
        "candidate_version": candidate.model_version,
        "candidate_f1": candidate.f1,
        "incumbent_f1": incumbent_metric,
        "promoted": promoted,
        "candidate_model_path": candidate.model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", required=True)
    parser.add_argument("--current-data", required=True)
    args = parser.parse_args()

    result = retrain(Path(args.reference_data), Path(args.current_data))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
