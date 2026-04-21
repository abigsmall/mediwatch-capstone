#!/usr/bin/env python3
"""Generate requirements-airflow.txt from direct project dependencies.

This script reads the direct dependencies declared in pyproject.toml and writes
an Airflow-focused requirements file with ranged specifiers instead of the
fully pinned lockfile export.

By default it excludes packages that are only needed for the API/UI or local
testing, while keeping the runtime dependencies used by the Airflow DAGs.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "requirements-airflow.txt"

# These are direct project dependencies that are not needed in the Airflow
# runtime for the current DAGs. They remain in pyproject.toml for the app/dev
# environments but are excluded from requirements-airflow.txt by default.
DEFAULT_EXCLUDES = {
    "fastapi",
    "httpx",
    "pytest",
    "python-multipart",
    "ray",
    "streamlit",
    "uvicorn",
}


def load_project_dependencies(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return list(data["project"]["dependencies"])


def filter_dependencies(
    dependency_strings: list[str],
    excluded_names: set[str],
    included_names: set[str],
) -> list[str]:
    filtered: list[str] = []

    for dependency_string in dependency_strings:
        requirement = Requirement(dependency_string)
        canonical_name = canonicalize_name(requirement.name)

        if canonical_name in included_names:
            filtered.append(dependency_string)
            continue

        if canonical_name in excluded_names:
            continue

        filtered.append(dependency_string)

    return filtered


def write_requirements(output_path: Path, requirement_lines: list[str]) -> None:
    header = [
        "# Generated from pyproject.toml by scripts/generate_requirements_airflow.py",
        "# Edit pyproject.toml or the script's include/exclude lists, then rerun the script.",
        "",
    ]
    output_path.write_text(
        "\n".join(header + requirement_lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate requirements-airflow.txt from pyproject.toml."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=PYPROJECT_PATH,
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output requirements file path",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional package name to exclude from Airflow requirements.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Package name to force-include even if excluded by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dependency_strings = load_project_dependencies(args.pyproject)
    excluded_names = set(DEFAULT_EXCLUDES) | {
        canonicalize_name(name) for name in args.exclude
    }
    included_names = {canonicalize_name(name) for name in args.include}

    requirement_lines = filter_dependencies(
        dependency_strings=dependency_strings,
        excluded_names=excluded_names,
        included_names=included_names,
    )

    write_requirements(args.output, requirement_lines)
    print(f"Wrote {len(requirement_lines)} requirements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
