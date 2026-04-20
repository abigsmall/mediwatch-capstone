from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from mediwatch.settings import settings


TARGET_COLUMN = "readmitted"
BINARY_TARGET_COLUMN = "readmitted_binary"
ID_COLUMNS = ["encounter_id", "patient_nbr"]
LEAKY_COLUMNS = ["readmitted", "encounter_id", "patient_nbr"]
OPTIONAL_DROP_COLUMNS = [
    "weight",
    "payer_code",
    "medical_specialty",
]
NUMERIC_CANDIDATES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


class DatasetNotFoundError(FileNotFoundError):
    pass


@dataclass
class DatasetBundle:
    full_df: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    train_x: pd.DataFrame
    test_x: pd.DataFrame
    train_y: pd.Series
    test_y: pd.Series


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or settings.raw_dataset_path
    if not dataset_path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {dataset_path}. Download the Kaggle diabetes "
            "readmission dataset and place diabetic_data.csv in data/raw/."
        )
    df = pd.read_csv(dataset_path)
    return normalize_raw_dataframe(df)


def normalize_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace("?", pd.NA)
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned[BINARY_TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN]
        .fillna("NO")
        .astype(str)
        .str.upper()
        .ne("NO")
        .astype(int)
    )
    for column in ID_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str)
    for column in NUMERIC_CANDIDATES:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected {TARGET_COLUMN} to be present in dataset.")

    feature_df = df.drop(columns=[column for column in LEAKY_COLUMNS if column in df.columns])
    feature_df = feature_df.drop(
        columns=[column for column in OPTIONAL_DROP_COLUMNS if column in feature_df.columns]
    )
    target = df[BINARY_TARGET_COLUMN]
    return feature_df, target


def make_dataset_bundle(df: pd.DataFrame) -> DatasetBundle:
    x, y = build_feature_frame(df)
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )
    return DatasetBundle(
        full_df=df,
        features=x,
        target=y,
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        test_y=test_y,
    )


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if df[column].dtype == "object"]


def infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in infer_categorical_columns(df)]


def frame_from_payloads(payloads: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(payloads))
    if frame.empty:
        raise ValueError("Prediction payloads were empty.")
    return frame
