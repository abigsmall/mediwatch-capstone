from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mediwatch.artifacts import save_model, upload_file_to_s3, upload_json_to_s3, write_manifest
from mediwatch.dataset import DatasetBundle, infer_categorical_columns, infer_numeric_columns
from mediwatch.settings import settings

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "model_type": "logistic_regression",
}


@dataclass
class TrainingResult:
    model_version: str
    model_path: str
    f1: float
    precision: float
    recall: float
    roc_auc: float
    average_precision: float
    positive_rate: float
    local_manifest_path: str
    s3_model_uri: str
    s3_manifest_uri: str
    run_id: str
    params: dict[str, Any]


def _normalize_model_config(
    model_config: dict[str, Any] | None = None,
    *,
    c_value: float = 1.0,
    class_weight: str | None = "balanced",
) -> dict[str, Any]:
    if model_config is None:
        model_config = {
            "model_type": "logistic_regression",
            "c_value": c_value,
            "class_weight": class_weight,
        }

    normalized = {**DEFAULT_MODEL_CONFIG, **model_config}
    if normalized["model_type"] == "logistic_regression":
        normalized.setdefault("c_value", c_value)
        normalized.setdefault("class_weight", class_weight)
    if normalized.get("class_weight") == "none":
        normalized["class_weight"] = None
    return normalized


def _build_classifier(model_config: dict[str, Any]):
    model_type = model_config["model_type"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=float(model_config["c_value"]),
            class_weight=model_config.get("class_weight"),
            max_iter=int(model_config.get("max_iter", 1500)),
            solver=str(model_config.get("solver", "liblinear")),
            random_state=settings.random_state,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 300)),
            max_depth=model_config.get("max_depth"),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            class_weight=model_config.get("class_weight"),
            n_jobs=int(model_config.get("n_jobs", 1)),
            random_state=settings.random_state,
        )

    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(model_config.get("n_estimators", 300)),
            max_depth=model_config.get("max_depth"),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            class_weight=model_config.get("class_weight"),
            n_jobs=int(model_config.get("n_jobs", 1)),
            random_state=settings.random_state,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _loggable_params(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "none" if value is None else value
        for key, value in model_config.items()
    }


def _loggable_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {
        key: "none" if value is None else str(value)
        for key, value in tags.items()
    }


def build_pipeline(
    train_x: pd.DataFrame,
    model_config: dict[str, Any] | None = None,
    *,
    c_value: float = 1.0,
    class_weight: str | None = "balanced",
) -> Pipeline:
    model_config = _normalize_model_config(
        model_config,
        c_value=c_value,
        class_weight=class_weight,
    )
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

    classifier = _build_classifier(model_config)

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
        "average_precision": float(average_precision_score(test_y, probabilities)),
        "positive_rate": float(predicted.mean()),
    }


def _model_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def train_and_log(
    bundle: DatasetBundle,
    c_value: float = 1.0,
    class_weight: str | None = "balanced",
    model_config: dict[str, Any] | None = None,
    register_as_best: bool = True,
    parent_run_id: str | None = None,
    extra_tags: dict[str, Any] | None = None,
    run_name_prefix: str = "mediwatch-train",
    log_model_artifacts: bool = True,
) -> TrainingResult:
    if register_as_best and not log_model_artifacts:
        raise ValueError("register_as_best=True requires log_model_artifacts=True.")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    normalized_model_config = _normalize_model_config(
        model_config,
        c_value=c_value,
        class_weight=class_weight,
    )
    params = _loggable_params(normalized_model_config)
    model_version = _model_version()

    with mlflow.start_run(
        run_name=f"{run_name_prefix}-{model_version}",
        nested=mlflow.active_run() is not None,
    ) as run:
        tags = {
            "run_type": "training",
            "model_type": normalized_model_config["model_type"],
            **(extra_tags or {}),
        }
        if parent_run_id:
            tags["mlflow.parentRunId"] = parent_run_id
        mlflow.set_tags(_loggable_tags(tags))
        mlflow.log_params(params)

        model = build_pipeline(bundle.train_x, model_config=normalized_model_config)
        model.fit(bundle.train_x, bundle.train_y)
        metrics = evaluate_pipeline(model, bundle.test_x, bundle.test_y)
        mlflow.log_metrics(metrics)

        model_path: Path | None = None
        s3_model_uri = ""
        s3_manifest_uri = ""
        if log_model_artifacts:
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
                "model_path": str(model_path),
                "promotion_metric": settings.promotion_metric,
                "metrics": metrics,
                "params": params,
                "mlflow_run_id": run.info.run_id,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            if register_as_best:
                model_key = f"{settings.s3_prefix}/models/{model_path.name}"
                manifest_key = f"{settings.s3_prefix}/models/latest_model.json"
                s3_model_uri = upload_file_to_s3(model_path, model_key)
                manifest["s3_model_uri"] = s3_model_uri
                s3_manifest_uri = upload_json_to_s3(manifest, manifest_key)
                write_manifest(manifest)

        result = TrainingResult(
            model_version=model_version,
            model_path=str(model_path) if model_path else "",
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics["roc_auc"],
            average_precision=metrics["average_precision"],
            positive_rate=metrics["positive_rate"],
            local_manifest_path=str(settings.manifest_path),
            s3_model_uri=s3_model_uri,
            s3_manifest_uri=s3_manifest_uri,
            run_id=run.info.run_id,
            params=params,
        )
        mlflow.log_dict(asdict(result), "training_result.json")
        return result
