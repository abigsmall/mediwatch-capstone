import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


@dataclass(frozen=True)
class Settings:
    project_name: str = "MediWatch"
    raw_dataset_path: Path = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
    sample_payload_path: Path = PROJECT_ROOT / "data" / "samples" / "sample_patient.json"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    evidently_dir: Path = PROJECT_ROOT / "evidently"
    mlruns_dir: Path = PROJECT_ROOT / "mlruns"
    manifest_path: Path = PROJECT_ROOT / "models" / "latest_model.json"
    reference_predictions_path: Path = PROJECT_ROOT / "data" / "processed" / "reference_predictions.csv"
    current_predictions_path: Path = PROJECT_ROOT / "data" / "processed" / "current_predictions.csv"
    drift_summary_path: Path = PROJECT_ROOT / "evidently" / "drift_summary.json"
    s3_bucket: str = os.getenv("MEDIWATCH_S3_BUCKET", "")
    s3_prefix: str = os.getenv("MEDIWATCH_S3_PREFIX", "mediwatch")
    airflow_api_url: str = os.getenv(
        "MEDIWATCH_AIRFLOW_API_URL",
        "http://localhost:8080/api/v1/dags/mediwatch_retraining/dagRuns",
    )
    airflow_username: str = os.getenv("MEDIWATCH_AIRFLOW_USERNAME", "airflow")
    airflow_password: str = os.getenv("MEDIWATCH_AIRFLOW_PASSWORD", "airflow")
    airflow_project_root: Path = Path(
        os.getenv("MEDIWATCH_AIRFLOW_PROJECT_ROOT", "/opt/airflow/project")
    )
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file://{PROJECT_ROOT / 'mlruns'}",
    )
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "mediwatch-readmission"
    )
    promotion_metric: str = os.getenv("MEDIWATCH_PROMOTION_METRIC", "f1")
    random_state: int = int(os.getenv("MEDIWATCH_RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("MEDIWATCH_TEST_SIZE", "0.2"))
    drift_share_threshold: float = float(
        os.getenv("MEDIWATCH_DRIFT_SHARE_THRESHOLD", "0.3")
    )

    def to_airflow_path(self, path: Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            return self.airflow_project_root / candidate

        try:
            relative = candidate.resolve(strict=False).relative_to(
                PROJECT_ROOT.resolve(strict=False)
            )
        except ValueError:
            return candidate

        return self.airflow_project_root / relative


settings = Settings()
