from pathlib import Path

import pandas as pd

from mediwatch.training.pipeline import TrainingResult


def _write_dataset(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_retrain_compares_candidate_and_incumbent_on_same_combined_test_split(
    tmp_path, monkeypatch
):
    from mediwatch.pipelines.retrain import retrain

    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"

    base_rows = [
        {
            "encounter_id": str(index),
            "patient_nbr": str(index),
            "readmitted": "NO" if index % 2 == 0 else "<30",
            "time_in_hospital": 3 + index,
            "num_lab_procedures": 10 + index,
            "num_procedures": 1,
            "num_medications": 5 + index,
            "number_outpatient": index % 3,
            "number_emergency": index % 2,
            "number_inpatient": index % 4,
            "number_diagnoses": 4 + (index % 5),
            "age": "[60-70)",
        }
        for index in range(10)
    ]

    _write_dataset(reference_path, base_rows[:6])
    _write_dataset(current_path, base_rows[6:])

    candidate_bundle = {}
    incumbent_bundle = {}
    tuned_config = {"c_value": 2.0, "class_weight": None}

    def fake_train_and_log(bundle, c_value, class_weight, register_as_best):
        candidate_bundle["test_x"] = bundle.test_x
        candidate_bundle["test_y"] = bundle.test_y
        candidate_bundle["c_value"] = c_value
        candidate_bundle["class_weight"] = class_weight
        return TrainingResult(
            model_version="candidate-v1",
            model_path=str(tmp_path / "candidate.joblib"),
            f1=0.60,
            precision=0.61,
            recall=0.62,
            roc_auc=0.63,
            average_precision=0.64,
            positive_rate=0.50,
            local_manifest_path=str(tmp_path / "manifest.json"),
            s3_model_uri="",
            s3_manifest_uri="",
            run_id="run-123",
            params={"C": c_value, "class_weight": class_weight},
        )

    def fake_evaluate_pipeline(model, test_x, test_y):
        incumbent_bundle["test_x"] = test_x
        incumbent_bundle["test_y"] = test_y
        return {
            "f1": 0.70,
            "precision": 0.71,
            "recall": 0.72,
            "roc_auc": 0.73,
            "average_precision": 0.74,
            "positive_rate": 0.40,
        }

    monkeypatch.setattr(
        "mediwatch.pipelines.retrain.run_tuning",
        lambda bundle: {"backend": "fallback-grid-search", "best_config": tuned_config},
    )
    monkeypatch.setattr("mediwatch.pipelines.retrain.train_and_log", fake_train_and_log)
    monkeypatch.setattr("mediwatch.pipelines.retrain.read_manifest", lambda: {"metrics": {"f1": 0.10}})
    monkeypatch.setattr("mediwatch.pipelines.retrain.load_model", lambda: object())
    monkeypatch.setattr("mediwatch.pipelines.retrain.evaluate_pipeline", fake_evaluate_pipeline)

    wrote_manifest = {"called": False}

    def fake_write_manifest(payload):
        wrote_manifest["called"] = True

    monkeypatch.setattr("mediwatch.pipelines.retrain.write_manifest", fake_write_manifest)

    result = retrain(reference_path, current_path)

    assert result["tuning_backend"] == "fallback-grid-search"
    assert result["best_config"] == tuned_config
    assert result["promotion_metric"] == "average_precision"
    assert result["candidate_metric"] == 0.64
    assert result["incumbent_metric"] == 0.74
    assert result["promoted"] is False
    assert wrote_manifest["called"] is False
    assert candidate_bundle["c_value"] == 2.0
    assert candidate_bundle["class_weight"] is None
    assert candidate_bundle["test_x"].equals(incumbent_bundle["test_x"])
    assert candidate_bundle["test_y"].equals(incumbent_bundle["test_y"])
