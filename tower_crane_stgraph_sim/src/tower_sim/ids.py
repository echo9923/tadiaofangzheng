from __future__ import annotations

from typing import Any

import pandas as pd


def scenario_id_from_index(index: int) -> str:
    return f"scenario_{int(index):06d}"


def crane_id_from_index(index: int) -> str:
    return f"crane_{int(index):02d}"


def task_id_from_index(crane_index: int, task_index: int) -> str:
    return f"task_{int(crane_index):02d}_{int(task_index):04d}"


def _parse_prefixed_int(value: Any, prefix: str) -> int:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(prefix):
            return int(text.removeprefix(prefix))
        return int(text)
    return int(value)


def scenario_index_from_id(value: Any) -> int:
    return _parse_prefixed_int(value, "scenario_")


def crane_index_from_id(value: Any) -> int:
    return _parse_prefixed_int(value, "crane_")


def task_index_from_id(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("task_"):
            return int(text.rsplit("_", 1)[-1])
        return int(text)
    return int(value)


def scenario_key(row: pd.Series) -> int:
    if "scenario_index" in row and not pd.isna(row["scenario_index"]):
        return int(row["scenario_index"])
    return scenario_index_from_id(row["scenario_id"])


def crane_key(row: pd.Series, id_column: str = "crane_id", index_column: str = "crane_index") -> int:
    if index_column in row and not pd.isna(row[index_column]):
        return int(row[index_column])
    return crane_index_from_id(row[id_column])


def task_key(row: pd.Series, id_column: str = "task_id", index_column: str = "task_index") -> int:
    if index_column in row and not pd.isna(row[index_column]):
        return int(row[index_column])
    return task_index_from_id(row[id_column])
