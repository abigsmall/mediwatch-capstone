import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import joblib

from mediwatch.settings import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# In Airflow, mounted repo directories may be readable but not writable,
# so fall back to /tmp for things like staging location for models before upload to S3.
def _writable_directory(preferred: Path, fallback_name: str) -> Path:
    if preferred.exists() and os.access(preferred, os.W_OK | os.X_OK):
        return preferred

    if preferred.parent.exists() and os.access(preferred.parent, os.W_OK | os.X_OK):
        return preferred

    fallback = Path("/tmp/mediwatch") / fallback_name
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def write_manifest(payload: dict[str, Any], path: Path | None = None) -> Path:
    manifest_path = path or settings.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def read_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or settings.manifest_path
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_model(model: Any, model_version: str) -> Path:
    output_dir = _writable_directory(settings.models_dir, "models")
    output_path = output_dir / f"mediwatch_model_{model_version}.joblib"
    joblib.dump(model, output_path)
    return output_path


def load_model(model_path: str | Path | None = None) -> Any:
    if model_path is None:
        manifest = read_manifest()
        model_path = manifest["local_model_path"]
    return joblib.load(model_path)


def upload_file_to_s3(file_path: Path, object_key: str) -> str:
    if not settings.s3_bucket:
        return ""
    client = boto3.client("s3")
    client.upload_file(str(file_path), settings.s3_bucket, object_key)
    return f"s3://{settings.s3_bucket}/{object_key}"


def upload_json_to_s3(payload: dict[str, Any], object_key: str) -> str:
    if not settings.s3_bucket:
        return ""
    client = boto3.client("s3")
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=object_key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{settings.s3_bucket}/{object_key}"
