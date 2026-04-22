import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mediwatch.artifacts import load_model
from mediwatch.dataset import (
    build_feature_frame,
    infer_categorical_columns,
    infer_numeric_columns,
    load_raw_dataset,
    normalize_raw_dataframe,
)
from mediwatch.settings import settings

EVIDENTLY_IMPORT_ERROR = None
try:
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift
except Exception as exc:  # pragma: no cover - dependency/version fallback
    EVIDENTLY_IMPORT_ERROR = exc
    DataDefinition = None
    Dataset = None
    Report = None
    ValueDrift = None


MONITORED_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "prediction_probability",
]


@dataclass
class DriftSummary:
    report_path: str
    drift_summary_path: str
    drift_detected: bool
    share_of_drifted_columns: float
    drifted_columns: list[str]
    monitored_rows: int
    triggered_retraining: bool


def _prepare_scored_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_raw_dataframe(raw_df)
    feature_df, target = build_feature_frame(normalized)
    model = load_model()
    scored = normalized.copy()
    scored["prediction_probability"] = model.predict_proba(feature_df)[:, 1]
    scored["prediction_label"] = (scored["prediction_probability"] >= 0.5).astype(int)
    scored["actual_label"] = target
    return scored


def _build_definition(df: pd.DataFrame) -> Any:
    numeric_columns = infer_numeric_columns(df)
    categorical_columns = infer_categorical_columns(df)
    return DataDefinition(
        numerical_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _collect_value_drift_results(obj: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        metric_id = str(obj.get("metric_id", obj.get("type", "")))
        if "ValueDrift" in metric_id:
            results.append(obj)
        for value in obj.values():
            results.extend(_collect_value_drift_results(value))
    elif isinstance(obj, list):
        for value in obj:
            results.extend(_collect_value_drift_results(value))
    return results


def _extract_result_payload(metric_node: dict[str, Any]) -> dict[str, Any]:
    result = metric_node.get("result")
    if isinstance(result, dict):
        return result
    return metric_node


def generate_drift_report(
    current_data_path: Path,
    reference_data_path: Path | None = None,
    trigger_retraining: bool = False,
) -> DriftSummary:
    reference_df = _prepare_scored_frame(load_raw_dataset(reference_data_path))
    current_df = _prepare_scored_frame(pd.read_csv(current_data_path))

    if Report is None:
        raise RuntimeError(
            "Evidently could not be imported with the expected API. "
            f"Original import error: {EVIDENTLY_IMPORT_ERROR!r}"
        )

    monitored_columns = [column for column in MONITORED_COLUMNS if column in reference_df.columns]
    definition = _build_definition(reference_df)
    reference_data = Dataset.from_pandas(reference_df, data_definition=definition)
    current_data = Dataset.from_pandas(current_df, data_definition=definition)

    report = Report(
        [ValueDrift(column=column) for column in monitored_columns],
        include_tests=False,
    )
    run = report.run(current_data=current_data, reference_data=reference_data)
    run_dict = run.dict()

    report_path = settings.evidently_dir / "drift_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(run_dict, indent=2), encoding="utf-8")

    value_drift_nodes = run_dict.get("metrics", [])
    drifted_columns = []
    for node in value_drift_nodes:
        payload = _extract_result_payload(node)
        config = payload.get("config", {})
        column_name = config.get("column")
        threshold = config.get("threshold")
        value = payload.get("value")
        if (
            column_name
            and isinstance(threshold, (int, float))
            and isinstance(value, (int, float))
            and value > threshold
        ):
            drifted_columns.append(str(column_name))

    drifted_columns = sorted(set(drifted_columns))
    share_of_drifted_columns = (
        len(drifted_columns) / len(monitored_columns) if monitored_columns else 0.0
    )
    drift_detected = share_of_drifted_columns >= settings.drift_share_threshold

    summary = DriftSummary(
        report_path=str(report_path),
        drift_summary_path=str(settings.drift_summary_path),
        drift_detected=drift_detected,
        share_of_drifted_columns=share_of_drifted_columns,
        drifted_columns=drifted_columns,
        monitored_rows=len(current_df),
        triggered_retraining=False,
    )

    settings.drift_summary_path.parent.mkdir(parents=True, exist_ok=True)
    settings.drift_summary_path.write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )

    if trigger_retraining and drift_detected:
        response = requests.post(
            settings.airflow_api_url,
            auth=(settings.airflow_username, settings.airflow_password),
            json={
                "conf": {
                    "reference_data_path": str(reference_data_path or settings.raw_dataset_path),
                    "current_data_path": str(current_data_path),
                    "drift_summary_path": str(settings.drift_summary_path),
                    "promotion_metric": settings.promotion_metric,
                }
            },
            timeout=30,
        )
        response.raise_for_status()
        summary.triggered_retraining = True
        settings.drift_summary_path.write_text(
            json.dumps(asdict(summary), indent=2), encoding="utf-8"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-data", required=True)
    parser.add_argument("--reference-data")
    parser.add_argument("--trigger-retraining", action="store_true")
    args = parser.parse_args()

    summary = generate_drift_report(
        current_data_path=Path(args.current_data),
        reference_data_path=Path(args.reference_data) if args.reference_data else None,
        trigger_retraining=args.trigger_retraining,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
