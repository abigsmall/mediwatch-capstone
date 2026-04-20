from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mediwatch.artifacts import save_model, upload_file_to_s3, upload_json_to_s3, write_manifest
from mediwatch.dataset import DatasetBundle, infer_categorical_columns, infer_numeric_columns
from mediwatch.settings import settings


@dataclass
class TrainingResult:
    model_version: str
    model_path: str
    f1: float
    precision: float
    recall: float
    roc_auc: float
    positive_rate: float
    local_manifest_path: str
    s3_model_uri: str
    s3_manifest_uri: str
    run_id: str
    params: dict[str, Any]


def build_pipeline(train_x: pd.DataFrame, c_value: float, class_weight: str | None) -> Pipeline:
    categorical_columns = infer_categorical_columns(train_x)
    numeric_columns = infer_numeric_columns(train_x)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )

    classifier = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=1500,
        solver="liblinear",
        random_state=settings.random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_pipeline(model: Pipeline, test_x: pd.DataFrame, test_y: pd.Series) -> dict[str, float]:
    predicted = model.predict(test_x)
    probabilities = model.predict_proba(test_x)[:, 1]
    return {
        "f1": float(f1_score(test_y, predicted)),
        "precision": float(precision_score(test_y, predicted)),
        "recall": float(recall_score(test_y, predicted)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "positive_rate": float(predicted.mean()),
    }


def _model_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def train_and_log(
    bundle: DatasetBundle,
    c_value: float = 1.0,
    class_weight: str | None = "balanced",
    register_as_best: bool = True,
) -> TrainingResult:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    params = {"C": c_value, "class_weight": class_weight or "none"}
    model_version = _model_version()

    with mlflow.start_run(run_name=f"mediwatch-train-{model_version}") as run:
        mlflow.log_params(params)

        model = build_pipeline(bundle.train_x, c_value=c_value, class_weight=class_weight)
        model.fit(bundle.train_x, bundle.train_y)
        metrics = evaluate_pipeline(model, bundle.test_x, bundle.test_y)
        mlflow.log_metrics(metrics)

        sample_input = bundle.test_x.head(5)
        sample_output = model.predict_proba(sample_input)[:, 1]
        signature = infer_signature(sample_input, sample_output)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=sample_input,
        )

        model_path = save_model(model, model_version=model_version)
        mlflow.log_artifact(str(model_path), artifact_path="exported")

        manifest = {
            "project": settings.project_name,
            "model_version": model_version,
            "local_model_path": str(model_path),
            "promotion_metric": settings.promotion_metric,
            "metrics": metrics,
            "params": params,
            "mlflow_run_id": run.info.run_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        s3_model_uri = ""
        s3_manifest_uri = ""
        if register_as_best:
            model_key = f"{settings.s3_prefix}/models/{model_path.name}"
            manifest_key = f"{settings.s3_prefix}/models/latest_model.json"
            s3_model_uri = upload_file_to_s3(model_path, model_key)
            manifest["s3_model_uri"] = s3_model_uri
            s3_manifest_uri = upload_json_to_s3(manifest, manifest_key)
            write_manifest(manifest)

        result = TrainingResult(
            model_version=model_version,
            model_path=str(model_path),
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics["roc_auc"],
            positive_rate=metrics["positive_rate"],
            local_manifest_path=str(settings.manifest_path),
            s3_model_uri=s3_model_uri,
            s3_manifest_uri=s3_manifest_uri,
            run_id=run.info.run_id,
            params=params,
        )
        mlflow.log_dict(asdict(result), "training_result.json")
        return result
