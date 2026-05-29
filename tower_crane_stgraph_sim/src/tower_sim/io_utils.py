from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    """Formal output layout for one generated dataset run."""

    root: Path
    tables: Path
    windows: Path
    quality: Path
    plots: Path
    config_used: Path
    metadata: Path
    data_dictionary: Path
    readme: Path


def dataset_paths(root: str | Path) -> DatasetPaths:
    """Return the formal dataset output layout rooted at ``root``."""

    root_path = Path(root)
    return DatasetPaths(
        root=root_path,
        tables=root_path / "tables",
        windows=root_path / "windows",
        quality=root_path / "quality",
        plots=root_path / "quality" / "plots",
        config_used=root_path / "config_used.yaml",
        metadata=root_path / "metadata.json",
        data_dictionary=root_path / "data_dictionary.md",
        readme=root_path / "README.md",
    )


def ensure_dataset_dirs(paths: DatasetPaths) -> DatasetPaths:
    """Create the dataset output directories and return the paths object."""

    for directory in [paths.root, paths.tables, paths.windows, paths.quality, paths.plots]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def dataclass_records(items: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert dataclass instances to dictionaries."""

    rows: list[dict[str, Any]] = []
    for item in items:
        if not is_dataclass(item):
            raise TypeError(f"Expected dataclass item, got {type(item)!r}")
        rows.append(asdict(item))
    return rows


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV with stable UTF-8 encoding."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def write_optional_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write parquet when pyarrow is installed, otherwise silently skip."""

    try:
        import pyarrow  # noqa: F401

        df.to_parquet(path, index=False)
    except Exception:
        return


def split_scenario_ids(
    scenario_ids: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    add_generalization_test: bool = False,
) -> dict[int, str]:
    """Assign scenario ids to train/val/test and optionally generalization."""

    if not scenario_ids:
        return {}
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    train_ratio /= total
    val_ratio /= total
    sorted_ids = sorted(set(int(x) for x in scenario_ids))
    generalization_ids: list[int] = []
    if add_generalization_test and len(sorted_ids) >= 5:
        gen_count = max(1, int(round(len(sorted_ids) * 0.1)))
        generalization_ids = sorted_ids[-gen_count:]
        sorted_ids = sorted_ids[:-gen_count]
    n = len(sorted_ids)
    train_end = max(1, int(round(n * train_ratio))) if n >= 3 else max(1, n - 2)
    val_count = max(1, int(round(n * val_ratio))) if n >= 3 else (1 if n >= 2 else 0)
    val_end = min(n, train_end + val_count)
    if val_end >= n and n >= 3:
        val_end = n - 1
    assignments: dict[int, str] = {}
    for idx, scenario_id in enumerate(sorted_ids):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        assignments[scenario_id] = split
    for scenario_id in generalization_ids:
        assignments[scenario_id] = "generalization"
    return assignments
