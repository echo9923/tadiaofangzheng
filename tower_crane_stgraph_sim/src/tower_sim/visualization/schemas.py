from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class ViewMode(str, Enum):
    """Visualization view modes with explicit future-label isolation."""

    INPUT = "input"
    LABEL = "label"
    DEBUG = "debug"

    @classmethod
    def parse(cls, value: str | "ViewMode") -> "ViewMode":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "input_view": cls.INPUT,
            "input": cls.INPUT,
            "label_view": cls.LABEL,
            "label": cls.LABEL,
            "debug_view": cls.DEBUG,
            "debug": cls.DEBUG,
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown visualization view mode: {value!r}")
        return aliases[normalized]


REQUIRED_TABLES = [
    "scenario_table",
    "crane_static",
    "task_table",
    "state_true",
    "state_obs",
    "geometry_table",
    "edge_current",
    "edge_future_label",
]

OPTIONAL_WINDOW_SPLITS = ["train", "val", "test", "generalization"]

INPUT_LABEL_TOKEN_EXCEPTIONS = {"same_height_risk_zone"}
FORBIDDEN_INPUT_SUBSTRINGS = ("future_min_d", "ttc_label", "future_risk", "label_risk", "risk_", "future", "label")
FORBIDDEN_INPUT_PREFIXES = ("risk_", "future_", "label_")


@dataclass(frozen=True)
class RiskTypeSpec:
    risk_type: str
    display_name: str
    risk_column: str
    current_distance_column: str
    future_distance_column: str
    ttc_label_column: str
    threshold_config_key: str


RISK_TYPE_SPECS: dict[str, RiskTypeSpec] = {
    "arm_arm": RiskTypeSpec(
        risk_type="arm_arm",
        display_name="臂-臂",
        risk_column="risk_arm_arm",
        current_distance_column="d_arm_arm",
        future_distance_column="future_min_d_arm_arm",
        ttc_label_column="ttc_label_arm_arm",
        threshold_config_key="d_safe_arm_arm_m",
    ),
    "arm_hook_i_to_j": RiskTypeSpec(
        risk_type="arm_hook_i_to_j",
        display_name="臂-钩 i->j",
        risk_column="risk_arm_hook_i_to_j",
        current_distance_column="d_arm_hook_i_to_j",
        future_distance_column="future_min_d_arm_hook_i_to_j",
        ttc_label_column="ttc_label_arm_hook",
        threshold_config_key="d_safe_arm_hook_m",
    ),
    "arm_hook_j_to_i": RiskTypeSpec(
        risk_type="arm_hook_j_to_i",
        display_name="臂-钩 j->i",
        risk_column="risk_arm_hook_j_to_i",
        current_distance_column="d_arm_hook_j_to_i",
        future_distance_column="future_min_d_arm_hook_j_to_i",
        ttc_label_column="ttc_label_arm_hook",
        threshold_config_key="d_safe_arm_hook_m",
    ),
    "hook_hook": RiskTypeSpec(
        risk_type="hook_hook",
        display_name="钩-钩",
        risk_column="risk_hook_hook",
        current_distance_column="d_hook_hook",
        future_distance_column="future_min_d_hook_hook",
        ttc_label_column="ttc_label_hook_hook",
        threshold_config_key="d_safe_hook_hook_m",
    ),
}


@dataclass(frozen=True)
class ScenarioData:
    scenario_id: Any
    scenario_row: pd.Series
    crane_static: pd.DataFrame
    task_table: pd.DataFrame
    state_true: pd.DataFrame
    state_obs: pd.DataFrame
    geometry: pd.DataFrame
    edge_current: pd.DataFrame
    edge_future_label: pd.DataFrame | None


@dataclass(frozen=True)
class AnimationFrame:
    scenario_id: Any
    step: int
    timestamp: float
    view_mode: ViewMode
    state: pd.DataFrame
    geometry: pd.DataFrame
    edges: pd.DataFrame
    labels: pd.DataFrame | None
    tasks: pd.DataFrame
    static: pd.DataFrame


def future_label_offenders(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Return input column names that look like future labels."""

    offenders: list[str] = []
    for column in columns:
        name = str(column)
        lowered = name.lower()
        if lowered in INPUT_LABEL_TOKEN_EXCEPTIONS:
            continue
        if any(token in lowered for token in FORBIDDEN_INPUT_SUBSTRINGS):
            offenders.append(name)
            continue
        if any(lowered.startswith(prefix) for prefix in FORBIDDEN_INPUT_PREFIXES):
            offenders.append(name)
    return offenders


def assert_input_columns_safe(columns: list[str] | tuple[str, ...], context: str = "input features") -> None:
    offenders = future_label_offenders(columns)
    if offenders:
        raise ValueError(f"Future-label leakage in {context}: {offenders}")


def risk_any_frame(labels: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a risk_any column when label risk columns are present."""

    if labels.empty:
        result = labels.copy()
        result["risk_any"] = pd.Series(dtype=float)
        return result
    columns = [spec.risk_column for spec in RISK_TYPE_SPECS.values() if spec.risk_column in labels.columns]
    result = labels.copy()
    result["risk_any"] = result[columns].max(axis=1) if columns else 0
    return result
