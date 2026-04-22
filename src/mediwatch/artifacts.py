import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


def _manifest_s3_key() -> str:
    return f"{settings.s3_prefix}/models/latest_model.json"


def _s3_client():
    return boto3.client("s3")


def _read_manifest_from_s3() -> dict[str, Any] | None:
    if not settings.s3_bucket:
        return None

    try:
        response = _s3_client().get_object(Bucket=settings.s3_bucket, Key=_manifest_s3_key())
    except Exception:
        return None

    return json.loads(response["Body"].read().decode("utf-8"))


def _write_local_manifest(payload: dict[str, Any], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Unsupported S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_s3_file(uri: str, fallback_name: str) -> Path:
    bucket, key = _parse_s3_uri(uri)
    download_dir = _writable_directory(settings.models_dir, fallback_name)
    target_path = download_dir / Path(key).name
    _s3_client().download_file(bucket, key, str(target_path))
    return target_path


def write_manifest(payload: dict[str, Any], path: Path | None = None) -> Path:
    manifest_path = path or settings.manifest_path
    if settings.s3_bucket:
        upload_json_to_s3(payload, _manifest_s3_key())

    try:
        return _write_local_manifest(payload, manifest_path)
    except PermissionError:
        fallback_dir = _writable_directory(manifest_path.parent, "manifests")
        return _write_local_manifest(payload, fallback_dir / manifest_path.name)


def read_manifest(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        s3_manifest = _read_manifest_from_s3()
        if s3_manifest is not None:
            return s3_manifest

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
        model_path_value = manifest.get("model_path") or manifest.get("local_model_path")
        if not model_path_value:
            raise KeyError("Manifest must contain either 'model_path' or 'local_model_path'.")
        resolved_model_path = Path(model_path_value)
        if resolved_model_path.exists():
            model_path = resolved_model_path
        elif manifest.get("s3_model_uri"):
            model_path = _download_s3_file(manifest["s3_model_uri"], "models")
        else:
            model_path = resolved_model_path
    return joblib.load(model_path)


def upload_file_to_s3(file_path: Path, object_key: str) -> str:
    if not settings.s3_bucket:
        return ""
    client = _s3_client()
    client.upload_file(str(file_path), settings.s3_bucket, object_key)
    return f"s3://{settings.s3_bucket}/{object_key}"


def upload_json_to_s3(payload: dict[str, Any], object_key: str) -> str:
    if not settings.s3_bucket:
        return ""
    client = _s3_client()
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=object_key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{settings.s3_bucket}/{object_key}"
