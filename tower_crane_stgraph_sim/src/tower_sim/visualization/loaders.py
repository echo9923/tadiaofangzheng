from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tower_sim.ids import scenario_index_from_id
from tower_sim.visualization.schemas import REQUIRED_TABLES, ScenarioData, ViewMode


@dataclass(frozen=True)
class RunLayout:
    root: Path
    tables_dir: Path
    windows_dir: Path
    quality_dir: Path
    plots_dir: Path
    metadata_path: Path
    config_path: Path


def resolve_run_layout(run_root: str | Path) -> RunLayout:
    """Resolve formal and legacy run layouts."""

    root = Path(run_root)
    tables_dir = root / "tables" if (root / "tables").exists() else root
    windows_dir = root / "windows" if (root / "windows").exists() else root
    quality_dir = root / "quality" if (root / "quality").exists() else root
    plots_dir = quality_dir / "plots" if (quality_dir / "plots").exists() else root / "plots"
    return RunLayout(
        root=root,
        tables_dir=tables_dir,
        windows_dir=windows_dir,
        quality_dir=quality_dir,
        plots_dir=plots_dir,
        metadata_path=root / "metadata.json",
        config_path=root / "config_used.yaml",
    )


def _read_table_file(tables_dir: Path, name: str, columns: list[str] | None = None, prefer_format: str = "parquet") -> pd.DataFrame:
    csv_path = tables_dir / f"{name}.csv"
    parquet_path = tables_dir / f"{name}.parquet"
    candidates = [parquet_path, csv_path] if prefer_format == "parquet" else [csv_path, parquet_path]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path, columns=columns)
        return pd.read_csv(path, usecols=columns)
    raise FileNotFoundError(f"Missing table {name!r}: expected {parquet_path} or {csv_path}")


def _scenario_index_value(value: Any) -> int | None:
    try:
        return scenario_index_from_id(value)
    except Exception:
        return None


def filter_scenario(df: pd.DataFrame, scenario_id: Any) -> pd.DataFrame:
    """Filter a table to one scenario while accepting string ids and numeric indexes."""

    if df.empty:
        return df.copy()
    scenario_index = _scenario_index_value(scenario_id)
    if "scenario_index" in df.columns and scenario_index is not None:
        result = df[pd.to_numeric(df["scenario_index"], errors="coerce") == scenario_index]
        if not result.empty:
            return result.copy()
    if "scenario_id" in df.columns:
        values = df["scenario_id"]
        result = df[values.astype(str) == str(scenario_id)]
        if not result.empty:
            return result.copy()
        if scenario_index is not None:
            numeric = pd.to_numeric(values, errors="coerce")
            result = df[numeric == scenario_index]
            if not result.empty:
                return result.copy()
    if "scenario_uid" in df.columns:
        result = df[df["scenario_uid"].astype(str) == str(scenario_id)]
        if not result.empty:
            return result.copy()
    return df.iloc[0:0].copy()


def select_scenario_row(scenario_table: pd.DataFrame, scenario_id: Any) -> pd.Series:
    match = filter_scenario(scenario_table, scenario_id)
    if match.empty:
        raise KeyError(f"Scenario not found: {scenario_id!r}")
    return match.iloc[0]


class RunDataRepository:
    """Lazy reader for one generated tower-crane dataset run."""

    def __init__(self, run_root: str | Path, prefer_format: str = "parquet") -> None:
        self.layout = resolve_run_layout(run_root)
        self.prefer_format = prefer_format
        self._table_cache: dict[tuple[str, tuple[str, ...] | None], pd.DataFrame] = {}

    @property
    def run_root(self) -> Path:
        return self.layout.root

    @property
    def tables_dir(self) -> Path:
        return self.layout.tables_dir

    @property
    def windows_dir(self) -> Path:
        return self.layout.windows_dir

    @property
    def quality_dir(self) -> Path:
        return self.layout.quality_dir

    def validate(self, required_tables: list[str] | None = None) -> list[str]:
        """Return missing required artifact descriptions."""

        missing: list[str] = []
        for table in required_tables or REQUIRED_TABLES:
            csv_path = self.tables_dir / f"{table}.csv"
            parquet_path = self.tables_dir / f"{table}.parquet"
            if not csv_path.exists() and not parquet_path.exists():
                missing.append(f"{table}.csv or {table}.parquet")
        if not self.layout.config_path.exists():
            missing.append("config_used.yaml")
        return missing

    def load_metadata(self) -> dict[str, Any]:
        if self.layout.metadata_path.exists():
            return json.loads(self.layout.metadata_path.read_text(encoding="utf-8"))
        scenario_count = 0
        try:
            scenario_count = int(self.load_scenario_table()["scenario_id"].nunique())
        except Exception:
            pass
        return {
            "project": "tower_crane_stgraph_sim",
            "run_root": str(self.run_root),
            "num_scenarios": scenario_count,
            "metadata_missing": True,
        }

    def load_config(self) -> dict[str, Any]:
        if not self.layout.config_path.exists():
            raise FileNotFoundError(f"Missing config file: {self.layout.config_path}")
        with self.layout.config_path.open("r", encoding="utf-8") as fh:
            value = yaml.safe_load(fh) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Config file must contain a mapping: {self.layout.config_path}")
        return value

    def load_table(self, name: str, columns: list[str] | None = None, use_cache: bool = True) -> pd.DataFrame:
        key = (name, tuple(columns) if columns is not None else None)
        if use_cache and key in self._table_cache:
            return self._table_cache[key].copy()
        df = _read_table_file(self.tables_dir, name, columns=columns, prefer_format=self.prefer_format)
        if use_cache:
            self._table_cache[key] = df.copy()
        return df

    def load_scenario_table(self) -> pd.DataFrame:
        return self.load_table("scenario_table")

    def list_scenarios(self) -> pd.DataFrame:
        return self.load_scenario_table().copy()

    def load_scenario_data(
        self,
        scenario_id: Any,
        *,
        view_mode: str | ViewMode = ViewMode.DEBUG,
        include_labels: bool | None = None,
    ) -> ScenarioData:
        mode = ViewMode.parse(view_mode)
        should_include_labels = mode is not ViewMode.INPUT if include_labels is None else include_labels
        scenario_table = self.load_scenario_table()
        scenario_row = select_scenario_row(scenario_table, scenario_id)
        canonical_id = scenario_row.get("scenario_id", scenario_id)

        edge_future_label: pd.DataFrame | None
        if should_include_labels:
            edge_future_label = filter_scenario(self.load_table("edge_future_label"), canonical_id)
        else:
            edge_future_label = None
        return ScenarioData(
            scenario_id=canonical_id,
            scenario_row=scenario_row,
            crane_static=filter_scenario(self.load_table("crane_static"), canonical_id),
            task_table=filter_scenario(self.load_table("task_table"), canonical_id),
            state_true=filter_scenario(self.load_table("state_true"), canonical_id),
            state_obs=filter_scenario(self.load_table("state_obs"), canonical_id),
            geometry=filter_scenario(self.load_table("geometry_table"), canonical_id),
            edge_current=filter_scenario(self.load_table("edge_current"), canonical_id),
            edge_future_label=edge_future_label,
        )

    def load_window_npz(self, split: str) -> dict[str, Any]:
        import numpy as np

        path = self.windows_dir / f"{split}_windows.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing window file: {path}")
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}

    def table_path(self, name: str) -> Path | None:
        parquet_path = self.tables_dir / f"{name}.parquet"
        csv_path = self.tables_dir / f"{name}.csv"
        if self.prefer_format == "parquet":
            return parquet_path if parquet_path.exists() else csv_path if csv_path.exists() else None
        return csv_path if csv_path.exists() else parquet_path if parquet_path.exists() else None
