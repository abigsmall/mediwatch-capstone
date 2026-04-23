from pathlib import Path

from mediwatch.settings import PROJECT_ROOT, Settings


def test_to_airflow_path_rewrites_repo_absolute_paths():
    settings = Settings(airflow_project_root=Path("/opt/airflow/project"))

    rewritten = settings.to_airflow_path(PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv")

    assert rewritten == Path("/opt/airflow/project/data/raw/diabetic_data.csv")


def test_to_airflow_path_rewrites_relative_paths():
    settings = Settings(airflow_project_root=Path("/opt/airflow/project"))

    rewritten = settings.to_airflow_path(Path("data/processed/simulated_drift.csv"))

    assert rewritten == Path("/opt/airflow/project/data/processed/simulated_drift.csv")


def test_to_airflow_path_leaves_external_absolute_paths_unchanged():
    settings = Settings(airflow_project_root=Path("/opt/airflow/project"))

    external = Path("/mnt/shared/custom.csv")

    assert settings.to_airflow_path(external) == external
