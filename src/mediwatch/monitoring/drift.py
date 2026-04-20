import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

from mediwatch.artifacts import ensure_directories, load_model
from mediwatch.dataset import load_raw_dataset, normalize_raw_dataframe, build_feature_frame
from mediwatch.settings import settings

try:
    from evidently import Report
    from evidently.metrics import DatasetDriftMetric, ValueDrift
except Exception:  # pragma: no cover - optional dependency during import
    Report = None
    DatasetDriftMetric = None
    ValueDrift = None


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
    scored["prediction_label"] = (
        scored["prediction_probability"] >= 0.5
    ).astype(int)
    scored["actual_label"] = target
    return scored


def generate_drift_report(
    current_data_path: Path,
    reference_data_path: Path | None = None,
    trigger_retraining: bool = False,
) -> DriftSummary:
    ensure_directories()
    reference_df = _prepare_scored_frame(load_raw_dataset(reference_data_path))
    current_df = _prepare_scored_frame(pd.read_csv(current_data_path))

    if Report is None:
        raise RuntimeError(
            "Evidently is not installed. Install project dependencies before running monitoring."
        )

    report = Report(
        metrics=[
            DatasetDriftMetric(),
            ValueDrift(column_name="prediction_probability"),
        ]
    )
    report.run(reference_data=reference_df, current_data=current_df)

    result = report.as_dict()
    drift_metric = next(
        metric for metric in result["metrics"] if metric["metric"] == "DatasetDriftMetric"
    )
    drifted_columns = sorted(drift_metric["result"]["drift_by_columns"].keys())
    share_of_drifted_columns = float(drift_metric["result"]["share_of_drifted_columns"])
    drift_detected = bool(drift_metric["result"]["dataset_drift"])

    report_path = settings.evidently_dir / "drift_report.html"
    report.save_html(str(report_path))

    summary = DriftSummary(
        report_path=str(report_path),
        drift_summary_path=str(settings.drift_summary_path),
        drift_detected=drift_detected,
        share_of_drifted_columns=share_of_drifted_columns,
        drifted_columns=drifted_columns,
        monitored_rows=len(current_df),
        triggered_retraining=False,
    )

    settings.drift_summary_path.write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )

    if trigger_retraining and drift_detected and share_of_drifted_columns >= settings.drift_share_threshold:
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
