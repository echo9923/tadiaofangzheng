from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tower_sim.visualization.loaders import RunDataRepository, filter_scenario
from tower_sim.visualization.schemas import RISK_TYPE_SPECS


@dataclass(frozen=True)
class BrowserResult:
    table_name: str
    data: pd.DataFrame
    warning: str | None = None


def label_table_warning(table_name: str) -> str | None:
    if table_name == "edge_future_label":
        return "edge_future_label 包含未来标签，不能作为模型输入。"
    return None


def filter_table(
    df: pd.DataFrame,
    *,
    scenario_id: Any | None = None,
    crane_id: Any | None = None,
    step: int | None = None,
    timestamp: float | None = None,
    crane_i: Any | None = None,
    crane_j: Any | None = None,
    risk_type: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if scenario_id is not None:
        result = filter_scenario(result, scenario_id)
    if crane_id is not None and "crane_id" in result.columns:
        result = result[result["crane_id"].astype(str) == str(crane_id)]
    if step is not None and "step" in result.columns:
        result = result[pd.to_numeric(result["step"], errors="coerce") == int(step)]
    if timestamp is not None and "timestamp" in result.columns:
        result = result[pd.to_numeric(result["timestamp"], errors="coerce") == float(timestamp)]
    if crane_i is not None and "crane_i" in result.columns:
        result = result[result["crane_i"].astype(str) == str(crane_i)]
    if crane_j is not None and "crane_j" in result.columns:
        result = result[result["crane_j"].astype(str) == str(crane_j)]
    if risk_type is not None:
        spec = RISK_TYPE_SPECS[risk_type]
        if spec.risk_column in result.columns:
            result = result[pd.to_numeric(result[spec.risk_column], errors="coerce").fillna(0) > 0]
    return result


def browse_table(repository: RunDataRepository, table_name: str, **filters: Any) -> BrowserResult:
    data = filter_table(repository.load_table(table_name), **filters)
    return BrowserResult(table_name=table_name, data=data, warning=label_table_warning(table_name))
