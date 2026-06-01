from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from tower_sim.visualization.i18n_zh import field_label, risk_type_label
from tower_sim.visualization.schemas import RISK_TYPE_SPECS, RiskTypeSpec


@dataclass(frozen=True)
class RiskEvent:
    scenario_id: Any
    step: int
    timestamp: float
    horizon_s: float
    crane_i: Any
    crane_j: Any
    risk_type: str
    future_min_distance: float
    ttc_label: float
    risk_column: str
    threshold: float | None = None
    scenario_index: int | None = None
    crane_i_index: int | None = None
    crane_j_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(row: pd.Series, key: str) -> int | None:
    if key not in row or pd.isna(row[key]):
        return None
    return int(row[key])


def _stable_id(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def _threshold_for_spec(spec: RiskTypeSpec, config: dict[str, Any] | None) -> float | None:
    if not config:
        return None
    value = config.get("risk_thresholds", {}).get(spec.threshold_config_key)
    return None if value is None else float(value)


def extract_risk_events(
    edge_future_label: pd.DataFrame,
    *,
    risk_types: list[str] | None = None,
    config: dict[str, Any] | None = None,
    include_zero_risk: bool = False,
) -> list[RiskEvent]:
    """Split edge_future_label rows into one event per risk type."""

    if edge_future_label.empty:
        return []
    specs = [RISK_TYPE_SPECS[name] for name in (risk_types or list(RISK_TYPE_SPECS))]
    events: list[RiskEvent] = []
    for _, row in edge_future_label.iterrows():
        for spec in specs:
            if spec.risk_column not in row:
                continue
            risk_value = float(row[spec.risk_column])
            if risk_value <= 0 and not include_zero_risk:
                continue
            events.append(
                RiskEvent(
                    scenario_id=_stable_id(row.get("scenario_id", row.get("scenario_uid"))),
                    scenario_index=_optional_int(row, "scenario_index"),
                    step=int(row["step"]),
                    timestamp=float(row.get("timestamp", row["step"])),
                    horizon_s=float(row.get("horizon_s", 0.0)),
                    crane_i=_stable_id(row.get("crane_i", row.get("crane_i_uid"))),
                    crane_j=_stable_id(row.get("crane_j", row.get("crane_j_uid"))),
                    crane_i_index=_optional_int(row, "crane_i_index"),
                    crane_j_index=_optional_int(row, "crane_j_index"),
                    risk_type=spec.risk_type,
                    future_min_distance=float(row.get(spec.future_distance_column, float("nan"))),
                    ttc_label=float(row.get(spec.ttc_label_column, -1.0)),
                    risk_column=spec.risk_column,
                    threshold=_threshold_for_spec(spec, config),
                )
            )
    return sorted(events, key=lambda event: (str(event.scenario_id), event.step, event.horizon_s, str(event.crane_i), str(event.crane_j), event.risk_type))


def events_to_frame(events: list[RiskEvent]) -> pd.DataFrame:
    return pd.DataFrame([event.to_dict() for event in events])


def _pair_filter(df: pd.DataFrame, event: RiskEvent) -> pd.DataFrame:
    result = df
    if event.scenario_index is not None and "scenario_index" in result.columns:
        result = result[pd.to_numeric(result["scenario_index"], errors="coerce") == event.scenario_index]
    elif "scenario_id" in result.columns:
        result = _filter_id_value(result, "scenario_id", event.scenario_id)
    if event.crane_i_index is not None and "crane_i_index" in result.columns:
        result = result[pd.to_numeric(result["crane_i_index"], errors="coerce") == event.crane_i_index]
    elif "crane_i" in result.columns:
        result = _filter_id_value(result, "crane_i", event.crane_i)
    if event.crane_j_index is not None and "crane_j_index" in result.columns:
        result = result[pd.to_numeric(result["crane_j_index"], errors="coerce") == event.crane_j_index]
    elif "crane_j" in result.columns:
        result = _filter_id_value(result, "crane_j", event.crane_j)
    return result.copy()


def _filter_id_value(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    numeric = pd.to_numeric(df[column], errors="coerce")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value is not None and not numeric.isna().all():
        matched = df[numeric == numeric_value]
        if not matched.empty:
            return matched
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return df[df[column].astype(str).isin({str(value), text})]


def edge_series_for_event(edge_current: pd.DataFrame, event: RiskEvent, *, include_horizon: bool = True) -> pd.DataFrame:
    """Return the current-distance time series for an event's directed pair."""

    result = _pair_filter(edge_current, event)
    if result.empty or not include_horizon or "timestamp" not in result.columns:
        return result.sort_values("step") if "step" in result.columns else result
    start = event.timestamp - event.horizon_s
    end = event.timestamp + event.horizon_s
    return result[(pd.to_numeric(result["timestamp"], errors="coerce") >= start) & (pd.to_numeric(result["timestamp"], errors="coerce") <= end)].sort_values("timestamp")


def current_edge_row_for_event(edge_current: pd.DataFrame, event: RiskEvent) -> pd.Series | None:
    result = _pair_filter(edge_current, event)
    if result.empty or "step" not in result.columns:
        return None
    current = result[pd.to_numeric(result["step"], errors="coerce") == event.step]
    if current.empty:
        return None
    return current.iloc[0]


def explain_risk_event(event: RiskEvent, current_edge_row: pd.Series | None = None) -> str:
    """Generate a compact Chinese Markdown explanation for one event."""

    spec = RISK_TYPE_SPECS[event.risk_type]
    current_text = "当前距离不可用。"
    if current_edge_row is not None and spec.current_distance_column in current_edge_row:
        current_distance = float(current_edge_row[spec.current_distance_column])
        current_text = f"当前{field_label(spec.current_distance_column)}为 {current_distance:.3f} m。"
    threshold_text = "" if event.threshold is None else f"配置的安全阈值为 {event.threshold:.3f} m。"
    if event.ttc_label >= 0:
        ttc_text = f"首次进入阈值发生在 t+{event.ttc_label:.3f} s。"
    else:
        ttc_text = "该事件没有记录首次进入阈值时间。"
    return (
        f"在场景 `{event.scenario_id}` 的第 `{event.step}` 步（t={event.timestamp:.3f} s），"
        f"`{event.crane_i}` -> `{event.crane_j}` 在 {event.horizon_s:.3f} s 未来预测窗口内被标记为"
        f"`{risk_type_label(event.risk_type)}`风险。{current_text}"
        f"未来窗口内最小距离为 {event.future_min_distance:.3f} m。{threshold_text}{ttc_text} "
        "该标签来自 `edge_future_label`，不能作为模型输入特征。"
    )
