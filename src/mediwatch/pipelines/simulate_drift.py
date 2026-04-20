import argparse
from pathlib import Path

import pandas as pd

from mediwatch.dataset import load_raw_dataset


def simulate_drift(output_path: Path, sample_size: int = 500) -> Path:
    df = load_raw_dataset().sample(n=sample_size, replace=True, random_state=42).copy()

    if "time_in_hospital" in df.columns:
        df["time_in_hospital"] = (pd.to_numeric(df["time_in_hospital"], errors="coerce").fillna(1) + 4).clip(1, 14)
    if "num_medications" in df.columns:
        df["num_medications"] = (pd.to_numeric(df["num_medications"], errors="coerce").fillna(1) + 10).clip(lower=1)
    if "number_inpatient" in df.columns:
        df["number_inpatient"] = (pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0) + 2).clip(lower=0)
    if "age" in df.columns:
        df["age"] = "[80-90)"
    if "admission_type_id" in df.columns:
        df["admission_type_id"] = 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/processed/simulated_drift.csv",
    )
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    simulate_drift(Path(args.output), sample_size=args.sample_size)


if __name__ == "__main__":
    main()
