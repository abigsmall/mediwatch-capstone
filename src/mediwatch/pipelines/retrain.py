from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mediwatch.artifacts import read_manifest, write_manifest
from mediwatch.dataset import build_feature_frame, normalize_raw_dataframe, make_dataset_bundle
from mediwatch.training.pipeline import train_and_log


def retrain(reference_data_path: Path, current_data_path: Path) -> dict:
    reference_df = normalize_raw_dataframe(pd.read_csv(reference_data_path))
    current_df = normalize_raw_dataframe(pd.read_csv(current_data_path))
    combined_df = pd.concat([reference_df, current_df], ignore_index=True)

    bundle = make_dataset_bundle(combined_df)
    candidate = train_and_log(bundle, c_value=1.0, class_weight="balanced", register_as_best=False)
    incumbent_manifest = read_manifest()
    incumbent_metric = float(incumbent_manifest["metrics"]["f1"])

    promoted = candidate.f1 >= incumbent_metric
    if promoted:
        new_manifest = read_manifest(Path(candidate.local_manifest_path)) if Path(candidate.local_manifest_path).exists() else {}
        new_manifest = {
            "project": incumbent_manifest.get("project", "MediWatch"),
            "model_version": candidate.model_version,
            "local_model_path": candidate.model_path,
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
